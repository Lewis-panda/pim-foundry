"""Typed graph execution across native CIM timing models and software models.

Backend interpreters produce the tensors consumed by subsequent nodes. A
separate scalar execution checks every node. The wall-clock time of simulation
is never used as modeled device latency. All transfers and packing are explicit.
"""

from collections import Counter
from dataclasses import asdict
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import struct

from . import typed
from .config import read_json
from .dcc_simulator import file_sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _positive(value, name, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(name + " must be a finite " + ("nonnegative" if zero else "positive") + " number")
    return value


def _matrix_args(node, tensors, values):
    k, n = tensors[node.inputs[1]].shape
    a = values[node.inputs[0]]
    b = values[node.inputs[1]]
    return [list(a[i:i + k]) for i in range(0, len(a), k)], [list(b[i:i + n]) for i in range(0, len(b), n)]


class AnalyticalBackend:
    """An explicit GPU/NPU/CPU cost profile with scalar functional execution."""
    name = "analytical"

    def __init__(self, options):
        self.options = options
        if set(options) != {"macs_per_ns", "elements_per_ns", "local_bytes_per_ns", "launch_ns", "mac_energy_pj", "element_energy_pj", "local_energy_pj_per_byte"}:
            raise ValueError("invalid analytical backend options")
        for key in ("macs_per_ns", "elements_per_ns", "local_bytes_per_ns"):
            _positive(options.get(key), key)
        for key in ("launch_ns", "mac_energy_pj", "element_energy_pj", "local_energy_pj_per_byte"):
            _positive(options.get(key), key, zero=True)

    def validate(self, node, tensors):
        typed.validate_node(node, tensors)

    def run(self, node, tensors, values, directory, available_bytes):
        count = typed.node_operations(node, tensors)
        linear = node.op in ("matvec", "matmul")
        data_bytes = sum(tensors[name].nbytes for name in dict.fromkeys(node.inputs)) + tensors[node.output].nbytes
        rate = self.options["macs_per_ns" if linear else "elements_per_ns"]
        duration = self.options["launch_ns"] + max(count / rate, data_bytes / self.options["local_bytes_per_ns"])
        energy = count * self.options["mac_energy_pj" if linear else "element_energy_pj"] + data_bytes * self.options["local_energy_pj_per_byte"]
        result = {"values": list(typed.execute_node(node, tensors, values)), "duration_ns": duration,
                  "energy_pj": energy, "workspace_bytes": 0, "evidence": "analytical_cost_with_scalar_functional_execution",
                  "instruction_lowering_verified": False, "native_arithmetic": False,
                  "profile_provenance": "supplied_uncalibrated_analytical_profile"}
        directory.mkdir(parents=True)
        save(directory / "kernel.json", result)
        return result


class SRAMBackend:
    name = "cimflow-sram"

    def __init__(self, options):
        self.options = options
        if set(options) != {"executable", "packing_bytes_per_ns", "packing_energy_pj_per_byte"}:
            raise ValueError("invalid SRAM backend options")
        if not isinstance(options.get("executable"), str):
            raise ValueError("CIMFlow executable path is required")
        for key in ("packing_bytes_per_ns", "packing_energy_pj_per_byte"):
            _positive(options.get(key), key, zero=key.endswith("per_byte"))

    def validate(self, node, tensors):
        reasons = typed.BackendCapabilities(self.name, ("matvec", "matmul"), ("int8",), ("int32",), ("int32",)).check(node, tensors)
        if reasons or node.arithmetic.overflow != "error" or node.arithmetic.requantize is not None:
            raise ValueError(node.name + ": SRAM needs INT8 inputs, INT32 output, explicit error-on-overflow linear arithmetic")
        k, n = tensors[node.inputs[1]].shape
        m = 1 if node.op == "matvec" else tensors[node.inputs[0]].shape[0]
        if m > 16 or k > 128 or n > 128:
            raise ValueError(node.name + ": SRAM compiler budget is M<=16, K/N<=128")

    def run(self, node, tensors, values, directory, available_bytes):
        from . import cimflow
        a, b = _matrix_args(node, tensors, values)
        manifest = cimflow.compile_matmul(a, b, directory)
        workspace = manifest["memory"]["cim_size"] + manifest["memory"]["staging_size"]
        if workspace > available_bytes:
            raise ValueError(node.name + ": packed SRAM workspace exceeds available device capacity")
        report = cimflow.execute(directory, Path(self.options["executable"]))
        packing = (sum(tensors[name].nbytes for name in dict.fromkeys(node.inputs))
                   + manifest["memory"]["staging_size"] + 2 * tensors[node.output].nbytes)
        return {"values": [v for row in report["output"] for v in row],
                "duration_ns": report["native_timing_ns"] + packing / self.options["packing_bytes_per_ns"],
                "energy_pj": report["native_energy_pj"] + packing * self.options["packing_energy_pj_per_byte"],
                "native_timing_ns": report["native_timing_ns"], "packing_bytes": packing,
                "packing_ns": packing / self.options["packing_bytes_per_ns"], "workspace_bytes": workspace,
                "evidence": report["evidence_kind"], "instruction_lowering_verified": True,
                "native_arithmetic": False, "profile_provenance": report["profile_provenance"],
                "native_report": str((directory / "execution.json").resolve())}


class HBMBackend:
    name = "hbm-simd16"

    def __init__(self, options):
        self.options = options
        required = {"runtime_descriptor", "packing_bytes_per_ns", "packing_energy_pj_per_byte"}
        if not required <= set(options) or set(options) - required - {"variant", "native_energy_pj_per_cycle"}:
            raise ValueError("invalid HBM backend options")
        if options.get("variant", "release") not in ("release", "asan"):
            raise ValueError("invalid HBM runtime variant")
        if not isinstance(options.get("runtime_descriptor"), str):
            raise ValueError("patched HBM runtime descriptor is required")
        for key in ("packing_bytes_per_ns", "packing_energy_pj_per_byte"):
            _positive(options.get(key), key, zero=key.endswith("per_byte"))
        if "native_energy_pj_per_cycle" in options:
            _positive(options["native_energy_pj_per_cycle"], "native_energy_pj_per_cycle", zero=True)

    def validate(self, node, tensors):
        reasons = typed.BackendCapabilities(self.name, ("matvec", "matmul"), ("float16",), ("float16",), ("float32",)).check(node, tensors)
        if reasons or node.arithmetic != typed.Arithmetic("float32"):
            raise ValueError(node.name + ": HBM SIMD16 requires FP16 inputs/output and sequential FP32 accumulation")
        if typed.node_operations(node, tensors) > 2_000_000:
            raise ValueError(node.name + ": HBM compiler exceeds the 2000000-MAC budget")

    def run(self, node, tensors, values, directory, available_bytes):
        from . import banked, native_hbm
        a, b = _matrix_args(node, tensors, values)
        manifest = banked.compile_matmul(a, b, directory)
        workspace = manifest["reserved_bytes"]
        if workspace > available_bytes:
            raise ValueError(node.name + ": packed HBM workspace exceeds available device capacity")
        report = native_hbm.execute(directory, Path(self.options["runtime_descriptor"]), directory / "native",
                                    variant=self.options.get("variant", "release"))
        packing = (sum(tensors[name].nbytes for name in dict.fromkeys(node.inputs))
                   + manifest["packed_initialization_bytes"] + 2 * tensors[node.output].nbytes)
        packing_energy = packing * self.options["packing_energy_pj_per_byte"]
        energy = (report["statistics"]["memory_system_cycles"] * self.options["native_energy_pj_per_cycle"]
                  + packing_energy) if "native_energy_pj_per_cycle" in self.options else None
        return {"values": [v for row in report["outputs"] for v in row],
                "duration_ns": report["native_timing_ns"] + packing / self.options["packing_bytes_per_ns"],
                "energy_pj": energy, "known_packing_energy_pj": packing_energy,
                "energy_evidence": "supplied analytical pJ-per-memory-cycle coefficient" if energy is not None else "unavailable",
                "native_timing_ns": report["native_timing_ns"], "packing_bytes": packing,
                "packing_ns": packing / self.options["packing_bytes_per_ns"], "workspace_bytes": workspace,
                "evidence": "instruction_functional_with_patched_DCC_request_timing_projection",
                "instruction_lowering_verified": True, "native_arithmetic": False,
                "profile_provenance": "Foundry SIMD16 ISA; supplied uncalibrated PIM completion costs",
                "native_report": str((directory / "native/report.json").resolve())}


_BACKENDS = {backend.name: backend for backend in (AnalyticalBackend, SRAMBackend, HBMBackend)}


def register_backend(name, factory):
    """Register an explicit Python backend factory; configuration never imports code."""
    if not isinstance(name, str) or not name or name in _BACKENDS or not callable(factory):
        raise ValueError("backend name must be new and factory callable")
    _BACKENDS[name] = factory


def validate_config(config):
    if (not isinstance(config, dict) or set(config) != {"schema_version", "devices", "links"}
            or type(config["schema_version"]) is not int or config["schema_version"] != 1):
        raise ValueError("invalid unified platform schema")
    devices, backends = {}, {}
    if not isinstance(config["devices"], list) or not 1 <= len(config["devices"]) <= 16:
        raise ValueError("platform must have 1..16 devices")
    for device in config["devices"]:
        if not isinstance(device, dict) or set(device) != {"name", "kind", "backend", "capacity_bytes", "idle_power_w", "options"}:
            raise ValueError("invalid device fields")
        name = device["name"]
        if not isinstance(name, str) or not name or name == "host" or name in devices:
            raise ValueError("device names must be unique and distinct from host")
        if (not isinstance(device["backend"], str) or device["kind"] not in ("sram", "hbm", "gpu", "npu", "cpu")
                or device["backend"] not in _BACKENDS):
            raise ValueError("unknown device kind or backend")
        if type(device["capacity_bytes"]) is not int or device["capacity_bytes"] <= 0:
            raise ValueError("capacity must be a positive integer")
        _positive(device["idle_power_w"], "idle power", zero=True)
        if not isinstance(device["options"], dict):
            raise ValueError("backend options must be an object")
        devices[name] = device
        backends[name] = _BACKENDS[device["backend"]](device["options"])
    links = {}
    if not isinstance(config["links"], list) or len(config["links"]) > 272:
        raise ValueError("invalid link list")
    for link in config["links"]:
        if not isinstance(link, dict) or set(link) != {"source", "destination", "bytes_per_ns", "latency_ns", "energy_pj_per_byte"}:
            raise ValueError("invalid link fields")
        key = (link["source"], link["destination"])
        if any(not isinstance(name, str) or name not in {*devices, "host"} for name in key) or key[0] == key[1] or key in links:
            raise ValueError("invalid or duplicate link endpoints")
        for item in ("bytes_per_ns", "latency_ns", "energy_pj_per_byte"):
            _positive(link[item], item, zero=item != "bytes_per_ns")
        links[key] = link
    return devices, backends, links


def _route(source, target, links):
    queue = [(source,)]
    seen = {source}
    for path in queue:
        if path[-1] == target:
            return path
        for a, b in sorted(links):
            if a == path[-1] and b not in seen:
                queue.append((*path, b))
                seen.add(b)
    raise ValueError(f"no transfer route from {source} to {target}")


def run(graph, inputs, config, placement, output: Path):
    """Cold, serial typed graph co-simulation with explicit fabric/packing costs."""
    graph.validate()
    devices, backends, links = validate_config(config)
    tensors = graph.tensor_map
    if type(placement) is not dict or set(placement) != {node.name for node in graph.nodes}:
        raise ValueError("placement must assign every node exactly once")
    for node in graph.nodes:
        target = placement[node.name]
        if target not in devices:
            raise ValueError("placement refers to an unknown device")
        backends[target].validate(node, tensors)
    values = typed.prepare_inputs(graph, inputs)
    canonical_inputs = {name: list(values[name]) for name in graph.inputs}
    oracle = typed.evaluate(graph, inputs)
    runtime_bindings = {}
    for name, device in devices.items():
        options = device["options"]
        for key in ("executable", "runtime_descriptor"):
            if key in options:
                runtime_bindings[name + ":" + key] = file_sha256(Path(options[key]))
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output already exists")
    output.mkdir(parents=True)
    save(output / "graph.json", typed.graph_to_dict(graph))
    save(output / "inputs.json", canonical_inputs)
    save(output / "platform.json", config)
    save(output / "placement.json", placement)
    homes = {name: "host" for name in values}
    resident = {name: set() for name in devices}
    peaks = {name: 0 for name in devices}
    uses = Counter(name for node in graph.nodes for name in node.inputs)
    events, observations = [], []
    now, fabric_energy = 0.0, 0.0
    formats = {"int8": "b", "int32": "i", "float16": "e", "float32": "f"}

    def memory(device, extra=0):
        total = sum(tensors[name].nbytes for name in resident[device]) + extra
        peaks[device] = max(peaks[device], total)
        if total > devices[device]["capacity_bytes"]:
            raise ValueError(f"{device}: requires {total} bytes, exceeding configured capacity")
        return total

    def move(name, target, consumer):
        nonlocal now, fabric_energy
        origin = homes[name]
        if target != "host" and name in resident[target]:
            return
        route = _route(origin, target, links)
        tensor = tensors[name]
        packed = struct.pack("<" + formats[tensor.dtype] * len(values[name]), *values[name])
        for a, b in zip(route, route[1:]):
            if b != "host":
                memory(b, tensor.nbytes if name not in resident[b] else 0)
            link = links[(a, b)]
            duration = link["latency_ns"] + len(packed) / link["bytes_per_ns"]
            energy = len(packed) * link["energy_pj_per_byte"]
            events.append({"kind": "transfer", "tensor": name, "consumer": consumer,
                           "source": a, "destination": b, "bytes": len(packed),
                           "payload_sha256": hashlib.sha256(packed).hexdigest(),
                           "start_ns": now, "end_ns": now + duration, "energy_pj": energy})
            now += duration
            fabric_energy += energy
        # The destination consumes the serialized and reconstructed backend value.
        values[name] = struct.unpack("<" + formats[tensor.dtype] * len(values[name]), packed)
        if target != "host":
            resident[target].add(name)

    try:
        for index, node in enumerate(graph.nodes):
            target = placement[node.name]
            for name in dict.fromkeys(node.inputs):
                move(name, target, node.name)
            resident[target].add(node.output)
            base_memory = memory(target)
            available = devices[target]["capacity_bytes"] - base_memory
            result = backends[target].run(node, tensors, values, output / f"node-{index:03d}", available)
            if not isinstance(result, dict):
                raise ValueError("backend must return a result object")
            if type(result.get("workspace_bytes")) is not int or result["workspace_bytes"] < 0:
                raise ValueError("backend workspace must be a nonnegative integer")
            if result.get("energy_pj") is not None:
                _positive(result["energy_pj"], "kernel energy", zero=True)
            if "known_packing_energy_pj" in result:
                _positive(result["known_packing_energy_pj"], "packing energy", zero=True)
            duration = _positive(result["duration_ns"], "kernel duration", zero=True)
            actual = tuple(result["values"])
            if len(actual) != math.prod(tensors[node.output].shape):
                raise ValueError(node.name + ": backend output shape mismatch")
            if any(typed.storage_value(value, tensors[node.output].dtype) != value for value in actual):
                raise ValueError(node.name + ": backend output violates storage precision")
            if actual != oracle[node.output]:
                raise ValueError(node.name + ": instruction result differs from independent typed reference")
            memory(target, result["workspace_bytes"])
            values[node.output] = actual
            homes[node.output] = target
            events.append({"kind": "kernel", "node": node.name, "device": target,
                           "start_ns": now, "end_ns": now + duration})
            now += duration
            observations.append({**result, "node": node.name, "device": target, "backend": devices[target]["backend"],
                                 "matches_scalar_reference": True})
            for name in node.inputs:
                uses[name] -= 1
                if uses[name] == 0 and name not in graph.outputs:
                    for buffers in resident.values():
                        buffers.discard(name)
            if uses[node.output] == 0 and node.output not in graph.outputs:
                resident[target].discard(node.output)
        for name in graph.outputs:
            move(name, "host", "graph_output")
    except Exception as error:
        save(output / "failure.json", {"status": "failed", "error": str(error),
                                        "events": events, "completed_nodes": len(observations)})
        raise
    expected = {name: oracle[name] for name in graph.outputs}
    actual = {name: values[name] for name in graph.outputs}
    quality = typed.compare_outputs(graph, actual, expected)
    idle = sum(device["idle_power_w"] for device in devices.values()) * now * 1000
    energy_known = all(item["energy_pj"] is not None for item in observations)
    result = {
        "schema_version": 1, "status": "typed_heterogeneous_simulation_verified",
        "workload_sha256": digest({"graph": typed.graph_to_dict(graph), "inputs": canonical_inputs}),
        "platform_sha256": digest({"config": config, "runtime_bindings": runtime_bindings}),
        "runtime_bindings": runtime_bindings, "placement": placement,
        "outputs": {name: list(value) for name, value in actual.items()}, "quality": quality,
        "modeled_latency_ns": now,
        "modeled_energy_pj": fabric_energy + idle + sum(item["energy_pj"] for item in observations) if energy_known else None,
        "known_energy_pj": fabric_energy + idle + sum(item["energy_pj"] if item["energy_pj"] is not None else item.get("known_packing_energy_pj", 0) for item in observations),
        "energy_complete": energy_known, "fabric_energy_pj": fabric_energy, "idle_energy_pj": idle,
        "peak_reserved_bytes": peaks, "logical_transfer_bytes": sum(event.get("bytes", 0) for event in events),
        "nodes": observations, "events": events,
        "evidence": "typed_backend_execution_with_declared_cost_models",
        "hardware_calibrated": False, "hardware_execution": False, "performance_comparison_eligible": False,
        "protocol": {"name": "foundry-typed-device-graph/v1", "initial_state": "cold", "schedule": "serial",
                     "scope": "device_graph_with_fabric_packing_compute_and_output", "numeric": "explicit_per_node_policy",
                     "costs": ["input", "weight", "interdevice", "packing", "compute", "synchronization", "output", "all_device_idle"],
                     "excludes": ["compilation_wall_time", "host_application_preprocessing", "host_idle_power"],
                     "energy_missing_components": [item["node"] for item in observations if item["energy_pj"] is None]},
    }
    save(output / "result.json", result)
    (output / "report.html").write_text(render(result))
    (output / "reproduce.py").write_text(
        "import json, sys\nfrom pathlib import Path\nsys.dont_write_bytecode = True\n"
        "root = Path(__file__).resolve().parent\nsys.path.insert(0, str(root / 'engine'))\n"
        "from pim_foundry.unified import replay\n"
        "destination = Path(sys.argv[1]) if len(sys.argv) == 2 else root.with_name(root.name + '-replay')\n"
        "print(json.dumps(replay(root, destination), indent=2))\n")
    engine = output / "engine/pim_foundry"
    shutil.copytree(Path(__file__).parent, engine, ignore=shutil.ignore_patterns("__pycache__"))
    manifest = {str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    save(output / "manifest.json", {"schema_version": 1, "files": manifest})
    return result


def verify(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("invalid unified manifest")
    expected = manifest["files"]
    required = {"graph.json", "inputs.json", "platform.json", "placement.json", "result.json", "report.html", "engine/pim_foundry/unified.py"}
    if not required <= set(expected):
        raise ValueError("manifest is missing required experiment artifacts")
    for name, sha in expected.items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe manifest path")
        path = directory / name
        chain = [directory.joinpath(*Path(name).parts[:i]) for i in range(1, len(Path(name).parts) + 1)]
        if (any(part.is_symlink() for part in chain) or not path.resolve().is_relative_to(directory)
                or not path.is_file() or file_sha256(path) != sha):
            raise ValueError("missing or modified artifact: " + str(name))
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and p != directory / "manifest.json"}
    if actual != set(expected):
        raise ValueError("manifest does not cover all artifacts")
    return {"passed": True, "files_verified": len(expected), "external_runtime_binaries_bundled": False}


def replay(directory, output):
    directory = Path(directory).resolve()
    verify(directory)
    expected = read_json(directory / "result.json")
    config = read_json(directory / "platform.json")
    bindings = {}
    for device in config["devices"]:
        for key in ("executable", "runtime_descriptor"):
            if key in device["options"]:
                bindings[device["name"] + ":" + key] = file_sha256(Path(device["options"][key]))
    if bindings != expected["runtime_bindings"]:
        raise ValueError("external runtime binding changed; do not silently replay a different model")
    actual = run(typed.graph_from_dict(read_json(directory / "graph.json")),
                 read_json(directory / "inputs.json"), config,
                 read_json(directory / "placement.json"), Path(output))
    fields = ("outputs", "quality", "modeled_latency_ns", "modeled_energy_pj", "known_energy_pj",
              "peak_reserved_bytes", "logical_transfer_bytes", "workload_sha256", "platform_sha256")
    matches = {key: expected[key] == actual[key] for key in fields}
    return {"passed": all(matches.values()), "checks": matches, "output": str(Path(output).resolve()),
            "comparison": "numerical outputs and declared model results, not wall-clock logs or hardware calibration"}


def compare(left, right, *, metric="latency"):
    if metric not in ("latency", "energy"):
        raise ValueError("metric must be latency or energy")
    reasons = []
    for side, result in (("left", left), ("right", right)):
        if not isinstance(result, dict):
            raise ValueError("comparison requires result objects")
        if result.get("status") != "typed_heterogeneous_simulation_verified":
            reasons.append(side + ":invalid_status")
        for key in ("workload_sha256", "platform_sha256"):
            value = result.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                reasons.append(side + ":invalid_" + key)
        if not isinstance(result.get("protocol"), dict) or result["protocol"].get("name") != "foundry-typed-device-graph/v1":
            reasons.append(side + ":invalid_protocol")
        value = result.get("modeled_latency_ns")
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            reasons.append(side + ":invalid_latency")
    for key in ("workload_sha256", "platform_sha256"):
        if left.get(key) != right.get(key):
            reasons.append("mismatch:" + key)
    protocols = [dict(result["protocol"]) if isinstance(result.get("protocol"), dict) else {} for result in (left, right)]
    if metric == "latency":
        for protocol in protocols:
            protocol.pop("energy_missing_components", None)
    if protocols[0] != protocols[1]:
        reasons.append("mismatch:protocol")
    if not all(isinstance(result.get("quality"), dict) and result["quality"].get("passed") is True for result in (left, right)):
        reasons.append("numerical_quality_failed")
    if metric == "energy" and not all(result.get("energy_complete") is True
                and type(result.get("modeled_energy_pj")) in (int, float)
                and math.isfinite(result["modeled_energy_pj"]) and result["modeled_energy_pj"] >= 0 for result in (left, right)):
        reasons.append("energy_incomplete_or_invalid")
    field = "modeled_latency_ns" if metric == "latency" else "modeled_energy_pj"
    return {"same_model_comparable": not reasons, "reasons": reasons,
            "metric": metric,
            "modeled_latency_ratio": left["modeled_latency_ns"] / right["modeled_latency_ns"] if not reasons and right["modeled_latency_ns"] > 0 else None,
            "modeled_ratio": left[field] / right[field] if not reasons and right[field] > 0 else None,
            "hardware_performance_comparison_eligible": False,
            "meaning": "conditional predictions under identical workload, numeric, platform and cost assumptions; not hardware speedup"}


def render(result):
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
        item["node"], item["device"], item["backend"], round(item["duration_ns"], 3), item["matches_scalar_reference"], item["evidence"])) + "</tr>" for item in result["nodes"])
    return """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>CIM Foundry — 共用後端實驗</title>
<style>body{font:16px/1.7 system-ui;margin:40px auto;max-width:1200px;padding:0 24px;background:#f6f7f9;color:#17232d}h1{font-size:28px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}small{color:#52606d}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<h1>SRAM／HBM／GPU／NPU 共用圖實驗</h1><p>數值驗證通過；每個節點的實際後端輸出繼續傳給下一節點。時間為明示模型下的預測，未經硬體校準。</p>""" + (
        f"<p>模型總時間：<b>{result['modeled_latency_ns']:.3f} ns</b>　搬移：{result['logical_transfer_bytes']} bytes　完整能量：{result['energy_complete']}</p>"
        "<table><thead><tr><th>節點</th><th>裝置</th><th>後端</th><th>模型 ns</th><th>數值相符</th><th>證據</th></tr></thead><tbody>" + rows + "</tbody></table>"
        "<p>SRAM/HBM 的數值由指令與記憶體解譯器驗證，原生模擬器提供時序；GPU/NPU 使用明示分析模型。搬移、資料整理、同步與輸出成本皆保留。</p>"
        "<p><a href='result.json'>完整結果與時間軸</a> · <a href='platform.json'>平台假設</a> · <a href='graph.json'>精度與計算圖</a> · <a href='manifest.json'>完整性清單</a></p>"
        "<details><summary>輸出數值</summary><pre>" + html.escape(json.dumps(result["outputs"], indent=2)) + "</pre></details></html>")

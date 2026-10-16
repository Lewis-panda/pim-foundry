"""Concrete n-D tensors on shared CIM backends, with resource event accounting.

Functional execution is sequential on the host. The dependency/resource schedule
is a separate prediction, never a measurement of simultaneous device execution.
No precision conversion or device fallback is implicit.
"""

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path

from . import typed, unified


def _numpy():
    import numpy as np
    return np


def default_platform(kind="cpu"):
    """An explicit, uncalibrated cost profile; the kind is not execution evidence."""
    return {"schema_version": 1, "devices": [{
        "name": kind, "kind": kind, "backend": "analytical",
        "capacity_bytes": 4 * 1024**3, "idle_power_w": 0,
        "options": {"macs_per_ns": 16, "elements_per_ns": 16,
                    "local_bytes_per_ns": 64, "launch_ns": 10,
                    "mac_energy_pj": 0, "element_energy_pj": 0,
                    "local_energy_pj_per_byte": 0}}],
        "links": [{"source": a, "destination": b, "bytes_per_ns": 16,
                   "latency_ns": 5, "energy_pj_per_byte": 0}
                  for a, b in (("host", kind), (kind, "host"))]}


def _placement(graph, devices, placement):
    if placement is None:
        choices = [name for name, d in devices.items() if d["backend"] == "analytical"]
        if not choices:
            raise ValueError("native-only platforms require explicit placement")
        placement = {n["name"]: choices[0] for n in graph["nodes"]}
    if (not isinstance(placement, dict) or set(placement) != {n["name"] for n in graph["nodes"]}
            or any(d not in devices for d in placement.values())):
        raise ValueError("placement must assign every node to a declared device exactly once")
    return dict(placement)


def _native(node, arrays, spec, backend, directory, capacity):
    np = _numpy()
    if node["op"] != "matmul" or len(arrays) != 2 or any(x.ndim < 2 for x in arrays):
        raise ValueError(node["name"] + ": native backend requires rank >= 2 matmul")
    a, b = arrays
    if node.get("attrs", {}).get("transpose_a", False):
        a = np.swapaxes(a, -1, -2)
    if node.get("attrs", {}).get("transpose_b", False):
        b = np.swapaxes(b, -1, -2)
    sram = backend.name == "cimflow-sram"
    input_dtype, output_dtype = ("int8", "int32") if sram else ("float16", "float16")
    if str(a.dtype) != input_dtype or str(b.dtype) != input_dtype or spec["dtype"] != output_dtype:
        raise ValueError(node["name"] + ": explicit " + input_dtype + " inputs and " + output_dtype + " output required")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("native matmul reduction dimensions differ")
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
    if not k or not m or not n:
        raise ValueError("native matmul requires nonempty dimensions")
    if sram and k > 128:
        raise ValueError("native SRAM K > 128 requires an explicit cross-tile reduction graph")
    if sram and k * int(np.max(np.abs(a.astype(np.int64)))) * int(np.max(np.abs(b.astype(np.int64)))) > 2**31 - 1:
        raise ValueError("native INT32 accumulator cannot guarantee overflow-free reduction")
    aa = np.broadcast_to(a, (*batch, m, k)).reshape((-1, m, k))
    bb = np.broadcast_to(b, (*batch, k, n)).reshape((-1, k, n))
    out = np.zeros((len(aa), m, n), dtype=output_dtype)
    records, duration, energy, peak, count = [], 0.0, 0.0, 0, 0
    tile_m, tile_n, tile_k = (16, 128, 128) if sram else (min(m, 16), min(n, 128), k)
    if tile_m * tile_n * tile_k > 2_000_000:
        raise ValueError("native HBM tile exceeds operation budget")
    for bi in range(len(aa)):
        for mi in range(0, m, tile_m):
            for ni in range(0, n, tile_n):
                total = None
                for ki in range(0, k, tile_k):
                    x, y = aa[bi, mi:mi+tile_m, ki:ki+tile_k], bb[bi, ki:ki+tile_k, ni:ni+tile_n]
                    tensor_map = {
                        "a": typed.Tensor("a", tuple(x.shape), input_dtype),
                        "b": typed.Tensor("b", tuple(y.shape), input_dtype),
                        "c": typed.Tensor("c", (x.shape[0], y.shape[1]), output_dtype)}
                    kernel = typed.Node("tile", "matmul", ("a", "b"), "c", typed.Arithmetic("int32" if sram else "float32"))
                    backend.validate(kernel, tensor_map)
                    values = {"a": tuple(x.reshape(-1).tolist()), "b": tuple(y.reshape(-1).tolist())}
                    live = sum(t.nbytes for t in tensor_map.values())
                    if live >= capacity:
                        raise ValueError("native tensor tile exceeds device capacity")
                    record = backend.run(kernel, tensor_map, values, directory / ("tile-%05d" % count), capacity - live)
                    observed = np.asarray(record["values"], dtype=output_dtype).reshape((x.shape[0], y.shape[1]))
                    expected = np.asarray(typed.execute_node(kernel, tensor_map, values), dtype=output_dtype).reshape(observed.shape)
                    if not np.array_equal(observed, expected):
                        raise ValueError(node["name"] + ": native instruction output differs from scalar arithmetic contract")
                    total = observed.astype(np.int64) if total is None and sram else observed if total is None else total + observed
                    duration += record["duration_ns"]
                    energy = None if energy is None or record["energy_pj"] is None else energy + record["energy_pj"]
                    peak = max(peak, record["workspace_bytes"])
                    records.append({k: v for k, v in record.items() if k != "values"})
                    count += 1
                if sram and (np.any(total < -(2**31)) or np.any(total > 2**31 - 1)):
                    raise ValueError("INT32 output overflow")
                out[bi, mi:mi+tile_m, ni:ni+tile_n] = total
    return out.reshape((*batch, m, n)), {
        "backend": backend.name, "duration_ns": duration, "energy_pj": energy,
        "workspace_bytes": peak, "tiles": records, "numerical_check": "passed",
        "numeric_contract": "INT32 exact" if sram else "sequential separate FP32 MAC, FP16 output",
        "hardware_execution": False}


def _cost(node, arrays, result, device):
    opts = device["options"]
    op = node["op"]
    macs, elements, workspace = 0, 0, 0
    if op == "matmul":
        macs = int(result.size * arrays[0].shape[-2 if node.get("attrs", {}).get("transpose_a", False) else -1])
    elif op == "conv2d":
        macs = int(result.size * math.prod(arrays[1].shape[1:]))
    elif op == "causal_attention":
        q, k, v = arrays
        scores = q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2]
        macs = int(scores * (q.shape[3] + v.shape[3]))
        elements = int(8 * scores)
        workspace = int(3 * scores * 4)
    else:
        elements = int(sum(a.size for a in arrays) + result.size)
        if op in ("max_pool2d", "avg_pool2d"):
            elements += int(result.size * math.prod(node["attrs"]["kernel"]))
    data_bytes = int(sum(a.nbytes for a in arrays) + result.nbytes)
    return {"backend": "analytical", "duration_ns": opts["launch_ns"] + max(
        macs / opts["macs_per_ns"] + elements / opts["elements_per_ns"], data_bytes / opts["local_bytes_per_ns"]),
        "energy_pj": macs * opts["mac_energy_pj"] + elements * opts["element_energy_pj"] + data_bytes * opts["local_energy_pj_per_byte"],
        "workspace_bytes": workspace, "mac_count": macs, "element_count": elements,
        "cost_model": "contraction MACs and explicit attention scores; tensor-element proxy for nonlinear operators",
        "profile_provenance": "supplied_uncalibrated_analytical_profile", "hardware_execution": False}


class _Calendars:
    def __init__(self):
        self.busy = defaultdict(list)

    def reserve(self, resources, ready, duration):
        start = ready
        while True:
            conflicts = [end for resource in resources for begin, end in self.busy[resource]
                         if begin < start + duration and end > start]
            if not conflicts:
                break
            start = max(conflicts)
        end = start + duration
        if duration:
            for resource in resources:
                self.busy[resource].append((start, end))
                self.busy[resource].sort()
        return start, end


def _validate_profile(profile, devices):
    required = {"nodes", "sizes", "sources", "outputs", "initializer_hashes"}
    if not isinstance(profile, dict) or set(profile) != required:
        raise ValueError("invalid execution profile fields")
    if (not isinstance(profile["sizes"], dict) or not profile["sizes"]
            or any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in profile["sizes"].items())):
        raise ValueError("profile tensor sizes must be nonnegative integer bytes")
    for key in ("sources", "outputs"):
        names = profile[key]
        if not isinstance(names, list) or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names) or set(names) - set(profile["sizes"]):
            raise ValueError("invalid profile " + key)
    if not profile["outputs"] or not isinstance(profile["nodes"], list):
        raise ValueError("profile must declare nodes and outputs")
    hashes = profile["initializer_hashes"]
    if not isinstance(hashes, dict) or hashes.keys() - set(profile["sources"]):
        raise ValueError("profile initializer hashes must identify sources")
    import re
    if any(not isinstance(v, str) or re.fullmatch(r"[a-f0-9]{64}", v) is None for v in hashes.values()):
        raise ValueError("invalid initializer hash")
    available, names = set(profile["sources"]), set()
    for node in profile["nodes"]:
        if not isinstance(node, dict) or not {"name", "inputs", "output", "device", "duration_ns", "energy_pj", "workspace_bytes"} <= node.keys():
            raise ValueError("invalid profile node")
        if (not isinstance(node["name"], str) or node["name"] in names or node["device"] not in devices
                or not isinstance(node["inputs"], list) or any(n not in available for n in node["inputs"])
                or node["output"] in available or node["output"] not in profile["sizes"]):
            raise ValueError("profile must have topological SSA nodes on declared devices")
        unified._positive(node["duration_ns"], "node duration", zero=True)
        if node["energy_pj"] is not None:
            unified._positive(node["energy_pj"], "node energy", zero=True)
        if type(node["workspace_bytes"]) is not int or node["workspace_bytes"] < 0:
            raise ValueError("workspace must be nonnegative integer bytes")
        names.add(node["name"])
        available.add(node["output"])
    if set(profile["outputs"]) - available:
        raise ValueError("profile output has no producer")


def schedule(profiles, config, *, arrivals=None, max_inflight=1, state="cold"):
    """List scheduling with compute, link and endpoint DMA exclusion.

    Each profile describes an executed graph. Warm mode shares byte-identical
    initializers; graph inputs (including KV) remain request-owned. Device memory
    includes live copies, output buffers and native packing workspaces.
    """
    devices, _, links = unified.validate_config(config)
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("at least one execution profile is required")
    for profile in profiles:
        _validate_profile(profile, devices)
    arrivals = [0] * len(profiles) if arrivals is None else arrivals
    if (not isinstance(arrivals, list) or len(arrivals) != len(profiles)
            or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in arrivals)
            or arrivals != sorted(arrivals)):
        raise ValueError("arrivals must be nondecreasing finite nonnegative times")
    if type(max_inflight) is not int or max_inflight < 1 or state not in ("cold", "warm"):
        raise ValueError("invalid inflight limit or state")
    calendar, events, requests, allocations = _Calendars(), [], [], []
    resident = {}
    energy = 0.0

    def allocation(device, size, start):
        item = {"device": device, "bytes": size, "start_ns": start, "end_ns": start}
        allocations.append(item)
        return item

    for request_id, profile in enumerate(profiles):
        admitted = max(arrivals[request_id], requests[request_id-max_inflight]["finish_ns"] if request_id >= max_inflight else 0)
        issued_finish = admitted
        location = {name: {"host": (admitted, None)} for name in profile["sources"]}
        last_use = Counter(name for n in profile["nodes"] for name in set(n["inputs"]))
        for name in profile["outputs"]:
            last_use[name] += 1

        def transfer(name, target):
            nonlocal energy
            copies = location[name]
            if target in copies:
                return copies[target][0]
            weight_key = (profile["initializer_hashes"].get(name), target)
            if state == "warm" and weight_key[0] and weight_key in resident:
                copies[target] = resident[weight_key]
                return max(admitted, copies[target][0])
            routes = []
            for source, (ready, _) in copies.items():
                try:
                    routes.append((ready, unified._route(source, target, links), source))
                except ValueError:
                    pass
            if not routes:
                raise ValueError("no transfer route for " + name + " to " + target)
            ready, route, source = min(routes, key=lambda x: (x[0], len(x[1]), x[1]))
            size = profile["sizes"][name]
            for a, b in zip(route, route[1:]):
                link = links[(a, b)]
                resources = ["link:"+a+":"+b, "dma:"+a, "dma:"+b]
                start, end = calendar.reserve(resources, max(admitted, ready), link["latency_ns"] + size / link["bytes_per_ns"])
                previous = copies[a][1]
                if previous:
                    previous["end_ns"] = max(previous["end_ns"], end)
                memory = allocation(b, size, start) if b != "host" else None
                copies[b] = (end, memory)
                events.append({"kind": "transfer", "request": request_id, "tensor": name,
                               "source": a, "destination": b, "bytes": size,
                               "start_ns": start, "end_ns": end, "resources": resources})
                if energy is not None:
                    energy += size * link["energy_pj_per_byte"]
                ready = end
            if state == "warm" and weight_key[0]:
                resident[weight_key] = copies[target]
            return ready

        def release(name, at):
            last_use[name] -= 1
            for _, memory in location[name].values():
                if memory:
                    memory["end_ns"] = max(memory["end_ns"], at)
            # Intervals are extended by subsequent uses; no allocation is
            # removed before its final graph consumer or output transfer.

        for node in profile["nodes"]:
            target = node["device"]
            ready = max([admitted] + [transfer(n, target) for n in dict.fromkeys(node["inputs"])])
            start, end = calendar.reserve(["compute:"+target], ready, node["duration_ns"])
            issued_finish = max(issued_finish, end)
            output_memory = allocation(target, profile["sizes"][node["output"]], start)
            output_memory["end_ns"] = end
            location[node["output"]] = {target: (end, output_memory)}
            if node["workspace_bytes"]:
                work = allocation(target, node["workspace_bytes"], start)
                work["end_ns"] = end
            for name in dict.fromkeys(node["inputs"]):
                release(name, end)
            events.append({"kind": "compute", "request": request_id, "node": node["name"], "device": target,
                           "start_ns": start, "end_ns": end, "resources": ["compute:"+target]})
            energy = None if energy is None or node["energy_pj"] is None else energy + node["energy_pj"]
        finish = issued_finish
        for name in profile["outputs"]:
            ready = transfer(name, "host")
            release(name, ready)
            finish = max(finish, ready)
        requests.append({"request": request_id, "arrival_ns": arrivals[request_id], "admitted_ns": admitted,
                         "finish_ns": finish, "latency_ns": finish-arrivals[request_id]})
    observation_end = max(r["finish_ns"] for r in requests)
    makespan = observation_end - arrivals[0]
    for _, memory in resident.values():
        if memory:
            memory["end_ns"] = observation_end
    peaks = {}
    for device in devices:
        changes = []
        for item in allocations:
            if item["device"] == device and item["end_ns"] > item["start_ns"]:
                changes.extend([(item["start_ns"], item["bytes"]), (item["end_ns"], -item["bytes"])])
        used = peak = 0
        # Release before allocate at an equal boundary.
        for _, delta in sorted(changes):
            used += delta
            peak = max(peak, used)
        peaks[device] = peak
        if peak > devices[device]["capacity_bytes"]:
            raise ValueError(device + ": scheduled live tensors and workspace exceed capacity")
    if energy is not None:
        energy += sum(d["idle_power_w"] for d in devices.values()) * makespan * 1000
    return {"schema": "pim-foundry.tensor-schedule.v1", "modeled_latency_ns": makespan,
            "observation_start_ns": arrivals[0], "observation_end_ns": observation_end,
            "modeled_energy_pj": energy, "requests": requests, "events": sorted(events, key=lambda e: (e["start_ns"], e["kind"])),
            "peak_device_bytes": peaks, "state": state, "max_inflight": max_inflight,
            "scheduler": "deterministic request-order list scheduling with earliest resource gaps",
            "memory_policy": "no eviction; live copies and persistent warm initializers",
            "hardware_execution": False, "hardware_calibrated": False}


def run(graph, inputs, config, placement=None, output=None):
    """Execute actual tensors, then predict their explicit heterogeneous schedule."""
    from .tensor_ir import validate_graph
    from .tensor_executor import execute
    np = _numpy()
    graph = validate_graph(graph)
    devices, backends, _ = unified.validate_config(config)
    placement = _placement(graph, devices, placement)
    native = any(devices[d]["backend"] != "analytical" for d in placement.values())
    if native and output is None:
        raise ValueError("native tensor execution requires a persistent output directory")
    directory = Path(output) if output is not None else None
    if directory:
        directory.mkdir(parents=True, exist_ok=False)
    metadata = {}

    def callback(node, arrays, spec, context):
        device = devices[placement[node["name"]]]
        if device["backend"] == "analytical":
            return None
        value, record = _native(node, arrays, spec, backends[device["name"]],
                                directory / ("node-%05d" % len(metadata)), device["capacity_bytes"])
        metadata[node["name"]] = record
        return {"value": value, "metadata": record}

    result = execute(graph, inputs, node_callback=callback)
    values = dict(result["intermediates"])
    for name, value in graph.get("initializers", {}).items():
        values.setdefault(name, np.asarray(value, dtype=graph["values"][name]["dtype"]))
    for name, value in inputs.items():
        values.setdefault(name, np.asarray(value, dtype=graph["values"][name]["dtype"]))
    sizes = {name: int(value.nbytes) for name, value in values.items()}
    nodes = []
    for node in graph["nodes"]:
        target = placement[node["name"]]
        record = metadata.get(node["name"]) or _cost(node, [values[n] for n in node["inputs"]], values[node["output"]], devices[target])
        nodes.append({"name": node["name"], "op": node["op"], "inputs": node["inputs"], "output": node["output"],
                      "device": target, **record})
    hashes = {name: hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
              for name in graph.get("initializers", {}) for value in [values[name]]}
    profile = {"nodes": nodes, "sizes": sizes, "sources": list(dict.fromkeys([*graph["inputs"], *graph.get("initializers", {})])),
               "outputs": graph["outputs"], "initializer_hashes": hashes}
    modeled = schedule([profile], config)
    from .dcc_simulator import file_sha256
    bindings = {name+":"+key: file_sha256(Path(device["options"][key]))
                for name, device in devices.items() for key in ("executable", "runtime_descriptor") if key in device["options"]}
    modeled.update({"nodes": nodes, "profile": profile, "platform_identity": {"config": config, "runtime_bindings": bindings},
                    "runtime_bindings": bindings, "placement": placement,
                    "functional_execution": "actual tensors; native ISA interpreters for explicitly placed linear nodes"})
    result["platform_execution"] = modeled
    if directory:
        unified.save(directory / "platform-execution.json", modeled)
        np.savez(directory / "outputs.npz", **result["outputs"])
    return result

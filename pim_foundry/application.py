"""Application contracts, explicit adapters, and scoped experiment comparison.

An application owns semantics and quality requirements. Experiments own devices,
implementation variants, and mapping. Original paper artifacts are a separate
execution mode from graphs actually lowered to the shared device backends.
"""

from dataclasses import dataclass
import copy
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

from . import typed, unified
from .config import read_json
from .dcc_simulator import file_sha256

APP_SCHEMA = "pim-foundry.application.v1"
EXPERIMENT_SCHEMA = "pim-foundry.experiment.v1"
RESULT_SCHEMA = "pim-foundry.application-result.v1"
STATELESS = {"kind": "stateless_single_batch", "initial_state": "cold", "state_policy": "reset_each_run"}
_ASSET_ROLES = {"model", "inputs", "preprocessing", "quality", "implementation"}


def _fields(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(label + " requires exactly: " + ", ".join(sorted(fields)))


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _finite(value, label, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(label + " must be finite and " + ("positive" if positive else "nonnegative"))


def asset(path, role, format="json"):
    """Create a content-bound asset descriptor; absolute paths are accepted."""
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path), "format": format, "role": role}


def load_manifest(path):
    path = Path(path).resolve()
    manifest = read_json(path)
    _fields(manifest, {"schema", "id", "adapter", "assets", "parameters", "scenario", "quality"}, "application")
    if manifest["schema"] != APP_SCHEMA or not isinstance(manifest["id"], str) or not manifest["id"]:
        raise ValueError("invalid application schema or id")
    if not isinstance(manifest["adapter"], str) or manifest["adapter"] not in _ADAPTERS:
        raise ValueError("unknown application adapter: " + str(manifest["adapter"]))
    if not isinstance(manifest["assets"], dict) or not manifest["assets"]:
        raise ValueError("application must declare content-bound assets")
    for name in ("parameters", "scenario", "quality"):
        if not isinstance(manifest[name], dict):
            raise ValueError(name + " must be an object")
    assets = {}
    for name, spec in manifest["assets"].items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("unsafe asset name")
        _fields(spec, {"path", "sha256", "format", "role"}, "asset " + name)
        if (not isinstance(spec["path"], str) or not spec["path"] or not _sha(spec["sha256"])
                or not isinstance(spec["format"], str) or not spec["format"] or spec["role"] not in _ASSET_ROLES):
            raise ValueError("invalid asset descriptor: " + name)
        resolved = (path.parent / spec["path"]).resolve()
        if not resolved.is_file() or file_sha256(resolved) != spec["sha256"]:
            raise ValueError("missing or modified application asset: " + name)
        assets[name] = resolved
    return manifest, assets


def _experiment(value):
    _fields(value, {"schema", "platform", "mapping", "variant"}, "experiment")
    if value["schema"] != EXPERIMENT_SCHEMA:
        raise ValueError("unsupported experiment schema")
    if any(not isinstance(value[key], dict) for key in ("platform", "mapping", "variant")):
        raise ValueError("platform, mapping and variant must be objects")
    return value


def task_identity(manifest):
    # Paths and display ids are deliberately excluded. Methods/budgets belong
    # to execution identity, while model/data/scenario/quality define the task.
    assets = {name: {key: value for key, value in spec.items() if key != "path"}
              for name, spec in manifest["assets"].items() if spec["role"] != "implementation"}
    return {"adapter_contract": manifest["adapter"], "assets": assets,
            "parameters": manifest["parameters"], "scenario": manifest["scenario"], "quality": manifest["quality"]}


def plan_graph(graph, platform, mapping):
    """Bounded deterministic feasibility search, with no precision conversion."""
    graph.validate()
    devices, backends, links = unified.validate_config(platform)
    strategy = mapping.get("strategy")
    if strategy == "first_feasible":
        _fields(mapping, {"strategy", "device_preference", "max_candidates"}, "mapping")
        preference, limit = mapping["device_preference"], mapping["max_candidates"]
        if (not isinstance(preference, list) or any(not isinstance(x, str) for x in preference)
                or len(set(preference)) != len(preference) or set(preference) - set(devices)):
            raise ValueError("device_preference must contain unique platform device names")
        if type(limit) is not int or not 1 <= limit <= 100000:
            raise ValueError("max_candidates must be an integer in 1..100000")
        order = preference + sorted(set(devices) - set(preference))
        explicit = None
    elif strategy == "explicit":
        _fields(mapping, {"strategy", "placement"}, "mapping")
        explicit, limit, order = mapping["placement"], max(1, len(graph.nodes)), sorted(devices)
        if not isinstance(explicit, dict) or set(explicit) != {n.name for n in graph.nodes} or any(d not in devices for d in explicit.values()):
            raise ValueError("explicit placement must bind every node to a platform device")
    else:
        raise ValueError("unknown graph mapping strategy")
    tensors, candidates, diagnostics = graph.tensor_map, {}, {}
    sources = set(graph.inputs) | {t.name for t in graph.tensors if t.data is not None}
    def reachable(source, target):
        try:
            unified._route(source, target, links)
            return True
        except ValueError:
            return False
    for node in graph.nodes:
        checks = {}
        lower_bound = sum(tensors[t].nbytes for t in set(node.inputs) | {node.output})
        for name in order:
            reasons = []
            try:
                backends[name].validate(node, tensors)
            except (ValueError, TypeError) as exc:
                reasons.append("unsupported_operation_or_numeric_contract: " + str(exc))
            if lower_bound > devices[name]["capacity_bytes"]:
                reasons.append("node_storage_lower_bound_exceeds_capacity")
            if any(t in sources for t in node.inputs) and not reachable("host", name):
                reasons.append("missing_host_input_route")
            if node.output in graph.outputs and not reachable(name, "host"):
                reasons.append("missing_output_host_route")
            checks[name] = {"supported": not reasons, "reasons": reasons, "minimum_bytes": lower_bound,
                            "workspace_check": "deferred_to_compile", "live_tensor_capacity_check": "deferred_to_execution"}
        candidates[node.name] = [name for name in order if checks[name]["supported"] and (explicit is None or explicit[node.name] == name)]
        diagnostics[node.name] = checks
    chosen, locations, tried, exhausted = {}, {name: "host" for name in sources}, 0, False
    def search(index):
        nonlocal tried, exhausted
        if index == len(graph.nodes):
            return True
        node = graph.nodes[index]
        for destination in candidates[node.name]:
            if tried >= limit:
                exhausted = True
                return False
            tried += 1
            if not all(reachable(locations[name], destination) for name in node.inputs):
                continue
            chosen[node.name], locations[node.output] = destination, destination
            if search(index + 1):
                return True
            chosen.pop(node.name)
            locations.pop(node.output)
        return False
    feasible = search(0)
    return {"passed": feasible, "placement": dict(chosen) if feasible else None, "nodes": diagnostics,
            "strategy": strategy, "candidate_order": order, "assignments_examined": tried,
            "search_limit_reached": exhausted, "optimality_claim": False,
            "reason": None if feasible else ("search_budget_exhausted" if exhausted else "no_placement_satisfies_checked_constraints"),
            "capacity_validation": "per_node_lower_bound_only; exact resident and packed memory checked during execution"}


@dataclass
class PreparedGraph:
    manifest: dict
    graph: object
    inputs: dict
    imported: object = None
    raw_inputs: object = None


class GraphAdapter:
    def __init__(self, onnx=False):
        self.onnx = onnx

    def prepare(self, manifest, assets):
        if manifest["scenario"] != STATELESS:
            raise ValueError("graph adapters require cold stateless_single_batch; KV state and decode loops are not supported")
        quality = manifest["quality"]
        _fields(quality, {"kind", "reference", "atol", "rtol"}, "numerical quality policy")
        reference = "onnx.reference.ReferenceEvaluator" if self.onnx else "typed_scalar/v1"
        if quality["kind"] != "numerical_equivalence" or quality["reference"] != reference:
            raise ValueError("this graph adapter requires numerical_equivalence against " + reference)
        for key in ("atol", "rtol"):
            _finite(quality[key], key)
        model_key = "model" if self.onnx else "graph"
        if set(assets) != {model_key, "inputs"}:
            raise ValueError("graph application requires exactly " + model_key + " and inputs assets")
        if manifest["assets"][model_key]["role"] != "model" or manifest["assets"]["inputs"]["role"] != "inputs":
            raise ValueError("model and inputs must use semantic asset roles")
        if (manifest["assets"][model_key]["format"] != ("onnx" if self.onnx else "pim-foundry.typed-graph.v1")
                or manifest["assets"]["inputs"]["format"] != "json"):
            raise ValueError("graph asset format does not match the adapter")
        raw = read_json(assets["inputs"])
        if self.onnx:
            from .onnx_frontend import load_model
            if set(manifest["parameters"]) - {"input_shapes"}:
                raise ValueError("ONNX parameters accept only input_shapes")
            imported = load_model(assets["model"], input_shapes=manifest["parameters"].get("input_shapes"))
            return PreparedGraph(manifest, imported.graph, imported.prepare_inputs(raw), imported, raw)
        if manifest["parameters"]:
            raise ValueError("typed graph parameters must be empty")
        graph = typed.graph_from_dict(read_json(assets["graph"]))
        values = typed.prepare_inputs(graph, raw)
        return PreparedGraph(manifest, graph, {name: list(values[name]) for name in graph.inputs})

    def inspect(self, prepared, experiment):
        if experiment["variant"]:
            raise ValueError("graph implementation is bound by its model; variant must be empty")
        return plan_graph(prepared.graph, experiment["platform"], experiment["mapping"])

    def execute(self, prepared, experiment, output):
        diagnostics = self.inspect(prepared, experiment)
        if not diagnostics["passed"]:
            raise ValueError("application placement failed: " + diagnostics["reason"])
        result = unified.run(prepared.graph, prepared.inputs, experiment["platform"], diagnostics["placement"], output)
        numerical = result["quality"]
        if prepared.imported is not None:
            policy = prepared.manifest["quality"]
            numerical = prepared.imported.compare_reference(prepared.raw_inputs, result["outputs"], atol=policy["atol"], rtol=policy["rtol"])
        metrics = {"latency": {"value": result["modeled_latency_ns"], "unit": "ns", "scope": result["protocol"]["scope"],
                               "aggregation": "one_cold_batch_serial_execution", "origin": "declared_device_and_fabric_models"}}
        if result["energy_complete"]:
            metrics["energy"] = {**metrics["latency"], "value": result["modeled_energy_pj"], "unit": "pJ"}
        contract = {key: value for key, value in result["protocol"].items() if key != "energy_missing_components"}
        return {"execution_mode": "shared_graph_backends", "metrics": metrics,
                "quality": {"numerical": {"status": "passed" if numerical["passed"] else "failed", "observation": numerical},
                            "task": {"status": "not_evaluated", "reason": "numerical_equivalence_is_not_dataset_task_accuracy"}},
                "platform_identity": {"platform_sha256": result["platform_sha256"]}, "cost_contract": contract,
                "execution_identity": {"workload_sha256": result["workload_sha256"], "placement": result["placement"],
                                       "runtime_bindings": result["runtime_bindings"],
                                       "frontend": prepared.imported.report if prepared.imported is not None else "typed_graph/v1"},
                "diagnostics": diagnostics, "outputs": result["outputs"], "hardware_execution": False}


class StarcAdapter:
    def prepare(self, manifest, assets):
        from . import starc_adapter
        return starc_adapter.prepare(manifest, assets)

    def inspect(self, prepared, experiment):
        from . import starc_adapter
        return starc_adapter.inspect(prepared, experiment)

    def execute(self, prepared, experiment, output):
        from . import starc_adapter
        return starc_adapter.execute(prepared, experiment, output)


class PimbenchAdapter:
    def prepare(self, manifest, assets):
        from . import pimbench_adapter
        return pimbench_adapter.prepare(manifest, assets)

    def inspect(self, prepared, experiment):
        from . import pimbench_adapter
        return pimbench_adapter.inspect(prepared, experiment)

    def execute(self, prepared, experiment, output):
        from . import pimbench_adapter
        return pimbench_adapter.execute(prepared, experiment, output)


class NeuPimsAdapter:
    def prepare(self, manifest, assets):
        from . import neupims_adapter
        return neupims_adapter.prepare(manifest, assets)

    def inspect(self, prepared, experiment):
        from . import neupims_adapter
        return neupims_adapter.inspect(prepared, experiment)

    def execute(self, prepared, experiment, output):
        from . import neupims_adapter
        return neupims_adapter.execute(prepared, experiment, output)


class CimflowApplicationAdapter:
    def prepare(self, manifest, assets):
        from . import cimflow_application
        return cimflow_application.prepare(manifest, assets)

    def inspect(self, prepared, experiment):
        from . import cimflow_application
        return cimflow_application.inspect(prepared, experiment)

    def execute(self, prepared, experiment, output):
        from . import cimflow_application
        return cimflow_application.execute(prepared, experiment, output)


class TransformerApplicationAdapter:
    def prepare(self, manifest, assets):
        from . import transformer_application
        return transformer_application.prepare(manifest, assets)

    def inspect(self, prepared, experiment):
        from . import transformer_application
        return transformer_application.inspect(prepared, experiment)

    def execute(self, prepared, experiment, output):
        from . import transformer_application
        return transformer_application.execute(prepared, experiment, output)


class TensorApplicationAdapter:
    def prepare(self, manifest, assets):
        from . import tensor_application
        return tensor_application.prepare(manifest, assets)

    def inspect(self, prepared, experiment):
        from . import tensor_application
        return tensor_application.inspect(prepared, experiment)

    def execute(self, prepared, experiment, output):
        from . import tensor_application
        return tensor_application.execute(prepared, experiment, output)


_ADAPTERS = {"typed-graph/v1": GraphAdapter(), "onnx-graph/v1": GraphAdapter(onnx=True),
             "starc-artifact/v1": StarcAdapter(), "pimbench-artifact/v1": PimbenchAdapter(),
             "neupims-artifact/v1": NeuPimsAdapter(), "cimflow-artifact/v1": CimflowApplicationAdapter(),
             "transformer-graph/v1": TransformerApplicationAdapter(), "tensor-graph/v2": TensorApplicationAdapter(),
             "onnx-tensor/v2": TensorApplicationAdapter()}


def register_adapter(name, adapter):
    """Explicit trusted Python registration. JSON cannot import arbitrary code."""
    if not isinstance(name, str) or not name or name in _ADAPTERS or any(not callable(getattr(adapter, method, None)) for method in ("prepare", "inspect", "execute")):
        raise ValueError("adapter needs a new name and prepare/inspect/execute methods")
    _ADAPTERS[name] = adapter


def inspect(application_path, experiment_path):
    manifest, assets = load_manifest(application_path)
    experiment = _experiment(read_json(experiment_path))
    adapter = _ADAPTERS[manifest["adapter"]]
    prepared = adapter.prepare(manifest, assets)
    return {"task_sha256": unified.digest(task_identity(manifest)), "adapter": manifest["adapter"],
            "diagnostics": adapter.inspect(prepared, experiment)}


def _engine_identity(directory):
    return {str(p.relative_to(directory)): file_sha256(p) for p in sorted(directory.rglob("*.py"))}


def _observations(result):
    if not isinstance(result.get("metrics"), dict) or not result["metrics"]:
        raise ValueError("adapter must return scoped metrics")
    for name, metric in result["metrics"].items():
        _fields(metric, {"value", "unit", "scope", "aggregation", "origin"}, "metric " + name)
        _finite(metric["value"], name)
        if any(not isinstance(metric[k], str) or not metric[k] for k in ("unit", "scope", "aggregation", "origin")):
            raise ValueError("metrics need explicit unit, scope, aggregation and origin")
    _fields(result.get("quality"), {"numerical", "task"}, "quality observations")
    for observation in result["quality"].values():
        if not isinstance(observation, dict) or observation.get("status") not in {"passed", "failed", "not_evaluated", "not_applicable"}:
            raise ValueError("invalid quality observation")


def run(application_path, experiment_path, output):
    manifest, assets = load_manifest(application_path)
    experiment = _experiment(read_json(experiment_path))
    adapter = _ADAPTERS[manifest["adapter"]]
    prepared = adapter.prepare(manifest, assets)
    diagnostics = adapter.inspect(prepared, experiment)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    unified.save(output / "diagnostics.json", diagnostics)
    if diagnostics.get("passed") is not True:
        unified.save(output / "failure.json", {"status": "preflight_failed", "diagnostics": diagnostics})
        raise ValueError("application preflight failed; see " + str(output / "diagnostics.json"))
    stored = copy.deepcopy(manifest)
    (output / "assets").mkdir()
    for name, path in assets.items():
        destination = output / "assets" / (name + path.suffix)
        shutil.copyfile(path, destination)
        if file_sha256(destination) != manifest["assets"][name]["sha256"]:
            raise ValueError("asset changed while snapshotting: " + name)
        stored["assets"][name]["path"] = str(destination.relative_to(output))
    unified.save(output / "application.json", stored)
    unified.save(output / "experiment.json", experiment)
    engine = output / "engine/pim_foundry"
    shutil.copytree(Path(__file__).parent, engine, ignore=shutil.ignore_patterns("__pycache__"))
    # Execute against the snapshot's assets, closing a source-file mutation race.
    stored, bound_assets = load_manifest(output / "application.json")
    prepared = adapter.prepare(stored, bound_assets)
    try:
        observation = adapter.execute(prepared, experiment, output / "execution")
        _observations(observation)
        identity = {"adapter": manifest["adapter"], "engine": _engine_identity(engine),
                    "implementation": observation["execution_identity"], "experiment": experiment}
        failed = any(q["status"] == "failed" for q in observation["quality"].values())
        result = {**observation, "schema": RESULT_SCHEMA, "status": "quality_failed" if failed else "completed", "application_id": manifest["id"],
                  "adapter": manifest["adapter"], "task_sha256": unified.digest(task_identity(manifest)),
                  "execution_sha256": unified.digest(identity), "platform_sha256": unified.digest(observation["platform_identity"]),
                  "cost_contract_sha256": unified.digest(observation["cost_contract"]),
                  "hardware_performance_comparison_eligible": False,
                  "quality_preserving_comparison_eligible": not failed and observation["quality"]["task"]["status"] == "passed"}
        unified.save(output / "result.json", result)
        (output / "report.html").write_text(render(result))
        (output / "reproduce.py").write_text(
            "import json, sys\nfrom pathlib import Path\nsys.dont_write_bytecode = True\n"
            "root = Path(__file__).resolve().parent\nsys.path.insert(0, str(root / 'engine'))\n"
            "from pim_foundry.application import replay\n"
            "out = Path(sys.argv[1]) if len(sys.argv) == 2 else root.with_name(root.name + '-replay')\n"
            "print(json.dumps(replay(root, out), indent=2))\n")
        _seal(output)
        return result
    except Exception as exc:
        unified.save(output / "failure.json", {"status": "failed", "error": str(exc)})
        raise


def _seal(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("application bundle may not contain symlinks: " + str(path))
        if path.is_file():
            files[str(path.relative_to(directory))] = file_sha256(path)
    unified.save(directory / "manifest.json", {"schema": "pim-foundry.application-bundle.v1", "files": files})


def verify(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    _fields(manifest, {"schema", "files"}, "application bundle manifest")
    if manifest["schema"] != "pim-foundry.application-bundle.v1" or not isinstance(manifest["files"], dict):
        raise ValueError("invalid application bundle manifest")
    files = manifest["files"]
    required = {"application.json", "experiment.json", "diagnostics.json", "result.json", "report.html", "reproduce.py", "engine/pim_foundry/application.py"}
    if not required <= set(files) or not any(x.startswith("execution/") for x in files):
        raise ValueError("incomplete application bundle")
    for name, expected in files.items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or not _sha(expected):
            raise ValueError("invalid application manifest entry")
        path = directory / name
        parents = [directory.joinpath(*Path(name).parts[:i]) for i in range(1, len(Path(name).parts) + 1)]
        if any(p.is_symlink() for p in parents) or not path.is_file() or file_sha256(path) != expected:
            raise ValueError("missing or modified artifact: " + name)
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and p != directory / "manifest.json"}
    if actual != set(files):
        raise ValueError("application manifest does not cover all files")
    app, _ = load_manifest(directory / "application.json")
    result = read_json(directory / "result.json")
    if result["task_sha256"] != unified.digest(task_identity(app)):
        raise ValueError("task identity disagrees with application")
    return {"passed": True, "files_verified": len(files), "integrity_is_not_external_attestation": True}


def replay(directory, output):
    directory = Path(directory).resolve()
    verify(directory)
    if _engine_identity(Path(__file__).parent) != _engine_identity(directory / "engine/pim_foundry"):
        # Public CLI replay must preserve the same engine guarantee as the
        # bundle's reproduce.py, even after the installed package is upgraded.
        process = subprocess.run([sys.executable, str(directory / "reproduce.py"), str(Path(output).resolve())],
                                 check=True, capture_output=True, text=True)
        return json.loads(process.stdout)
    before = read_json(directory / "result.json")
    after = run(directory / "application.json", directory / "experiment.json", output)
    fields = ("status", "task_sha256", "execution_sha256", "platform_sha256", "cost_contract_sha256", "metrics", "quality", "execution_identity")
    checks = {name: before[name] == after[name] for name in fields}
    return {"passed": all(checks.values()), "checks": checks, "output": str(Path(output).resolve())}


def compare_results(left, right, metric="latency"):
    """Conditional same-platform comparison; never certifies physical speedup."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("application comparison requires two result objects")
    reasons = []
    for side, result in (("left", left), ("right", right)):
        if result.get("schema") != RESULT_SCHEMA or result.get("status") != "completed":
            reasons.append(side + ":unsuccessful_application")
        for key in ("task_sha256", "platform_sha256", "cost_contract_sha256", "execution_sha256"):
            if not _sha(result.get(key)):
                reasons.append(side + ":invalid_" + key)
        try:
            _observations(result)
        except (ValueError, TypeError):
            reasons.append(side + ":invalid_observations")
        quality = result.get("quality") if isinstance(result.get("quality"), dict) else {}
        if any(q.get("status") == "failed" for q in quality.values() if isinstance(q, dict)):
            reasons.append(side + ":quality_failed")
    for key in ("task_sha256", "platform_sha256", "cost_contract_sha256", "execution_mode"):
        if left.get(key) != right.get(key):
            reasons.append("mismatch:" + key)
    metrics = [r.get("metrics") if isinstance(r.get("metrics"), dict) else {} for r in (left, right)]
    a, b = metrics[0].get(metric), metrics[1].get(metric)
    if not isinstance(a, dict) or not isinstance(b, dict):
        reasons.append("metric_unavailable:" + metric)
    else:
        if {k: v for k, v in a.items() if k != "value"} != {k: v for k, v in b.items() if k != "value"}:
            reasons.append("mismatch:metric_scope_unit_or_origin")
        if type(b.get("value")) not in (int, float) or b["value"] <= 0:
            reasons.append("nonpositive_denominator")
    quality_passed = all(isinstance(r.get("quality"), dict) and isinstance(r["quality"].get("task"), dict)
                         and r["quality"]["task"].get("status") == "passed" for r in (left, right))
    return {"same_model_comparable": not reasons, "reasons": reasons, "metric": metric,
            "left_over_right_ratio": a["value"] / b["value"] if not reasons else None,
            "quality_preserving_comparison_eligible": not reasons and quality_passed,
            "quality_limit": None if quality_passed else "task_quality_not_evaluated_or_not_passed",
            "hardware_performance_comparison_eligible": False,
            "meaning": "conditional modeled ratio for the same semantic task, scenario, platform and accounting boundary"}


def compare(left_directory, right_directory, metric="latency"):
    verify(left_directory)
    verify(right_directory)
    return compare_results(read_json(Path(left_directory) / "result.json"), read_json(Path(right_directory) / "result.json"), metric)


def render(result):
    rows = "".join("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in (name, value["value"], value["unit"], value["scope"], value["aggregation"])) + "</tr>" for name, value in result["metrics"].items())
    return ("<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><title>Application experiment</title>"
            "<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#182837;background:#f5f7fa}"
            "table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
            "<h1>" + html.escape(result["application_id"]) + "</h1><p>Execution mode: " + html.escape(result["execution_mode"]) + "</p>"
            "<p>以下數值是指定模型與計量範圍下的結果。數值一致性、任務品質與硬體效能分開記錄。</p>"
            "<table><tr><th>Metric</th><th>Value</th><th>Unit</th><th>Scope</th><th>Aggregation</th></tr>" + rows + "</table>"
            "<h2>品質觀測</h2><pre>" + html.escape(json.dumps(result["quality"], indent=2, ensure_ascii=False)) + "</pre>"
            "<p><a href='application.json'>Application</a> · <a href='experiment.json'>Experiment</a> · <a href='diagnostics.json'>Capabilities</a> · "
            "<a href='result.json'>Result / provenance</a> · <a href='manifest.json'>Integrity manifest</a></p></html>")

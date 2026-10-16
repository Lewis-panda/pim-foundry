"""Application adapter for externally authored tensor graphs and reference data."""
from pathlib import Path

from . import unified
from .config import read_json


def prepare(manifest, assets):
    import numpy as np
    from .tensor_ir import validate_graph
    onnx = manifest["adapter"] == "onnx-tensor/v2"
    model_key = "model" if onnx else "graph"
    if not {model_key, "inputs", "reference"} <= set(assets) or set(assets) - {model_key, "inputs", "reference", "weights"}:
        raise ValueError("tensor application requires graph, inputs, reference and optional weights assets")
    roles = {model_key: "model", "inputs": "inputs", "reference": "quality", "weights": "model"}
    if any(manifest["assets"][k]["role"] != roles[k] for k in assets):
        raise ValueError("tensor assets must preserve their semantic roles")
    if (not onnx and manifest["parameters"]) or (onnx and set(manifest["parameters"]) - {"input_shapes"}):
        raise ValueError("tensor adapter parameters must be empty")
    if onnx:
        from . import tensor_onnx
        graph = tensor_onnx.load_model(assets["model"], input_shapes=manifest["parameters"].get("input_shapes"))
    else:
        graph = read_json(assets["graph"])
    if "weights" in assets:
        with np.load(assets["weights"], allow_pickle=False) as data:
            if set(data.files) & set(graph.get("initializers", {})):
                raise ValueError("weights and inline initializers overlap")
            graph.setdefault("initializers", {}).update({name: data[name] for name in data.files})
    graph = validate_graph(graph)
    with np.load(assets["inputs"], allow_pickle=False) as data:
        inputs = {name: data[name] for name in data.files}
    if onnx:
        inputs = tensor_onnx.prepare_inputs(graph, inputs)
    with np.load(assets["reference"], allow_pickle=False) as data:
        reference = {name: data[name] for name in data.files}
    if onnx:
        names = graph["metadata"]["onnx"]["output_names"]
        if set(reference) != set(names):
            raise ValueError("ONNX reference must use every original output name")
        reference = {names[name]: value for name, value in reference.items()}
        for name, value in reference.items():
            if value.dtype == np.int64 and graph["values"][name]["dtype"] == "int32":
                if value.size and (np.min(value) < -(2**31) or np.max(value) > 2**31-1):
                    raise ValueError("ONNX integer reference exceeds checked INT32 contract")
                reference[name] = value.astype(np.int32)
    if set(reference) != set(graph["outputs"]):
        raise ValueError("reference must contain every graph output exactly once")
    if any(not np.isfinite(x).all() for x in reference.values()):
        raise ValueError("reference outputs must be finite")
    quality = manifest["quality"]
    if (set(quality) != {"kind", "reference", "atol", "rtol"} or quality["kind"] != "numerical_equivalence"
            or not isinstance(quality["reference"], str) or not quality["reference"]):
        raise ValueError("explicit numerical reference identity and tolerances are required")
    for key in ("atol", "rtol"):
        unified._positive(quality[key], key, zero=True)
    scenario = manifest["scenario"]
    if (set(scenario) != {"kind", "arrivals_ns", "max_inflight", "initial_state"}
            or scenario["kind"] != "tensor_requests" or scenario["initial_state"] not in ("cold", "warm")):
        raise ValueError("tensor_requests scenario requires arrivals_ns, max_inflight and initial_state")
    arrivals = scenario["arrivals_ns"]
    if not isinstance(arrivals, list) or not 1 <= len(arrivals) <= 10000:
        raise ValueError("tensor scenario requires 1..10000 arrivals")
    for x in arrivals:
        unified._positive(x, "arrival", zero=True)
    if arrivals != sorted(arrivals) or type(scenario["max_inflight"]) is not int or scenario["max_inflight"] < 1:
        raise ValueError("invalid arrival ordering or max_inflight")
    return {"manifest": manifest, "graph": graph, "inputs": inputs, "reference": reference}


def inspect(prepared, experiment):
    from .tensor_platform import _placement
    devices, _, _ = unified.validate_config(experiment["platform"])
    if experiment["variant"]:
        raise ValueError("precision changes must be explicit in the model; variant must be empty")
    mapping = experiment["mapping"]
    if mapping == {"strategy": "default"}:
        requested = None
    elif set(mapping) == {"strategy", "placement"} and mapping["strategy"] == "explicit":
        requested = mapping["placement"]
    else:
        raise ValueError("tensor mapping must be default or explicit placement")
    placement = _placement(prepared["graph"], devices, requested)
    return {"passed": True, "placement": placement, "shape_and_live_memory_checks": "at_execution",
            "hardware_performance_claim": False}


def execute(prepared, experiment, output):
    import numpy as np
    from . import tensor_platform
    diagnostics = inspect(prepared, experiment)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = tensor_platform.run(prepared["graph"], prepared["inputs"], experiment["platform"],
                                 placement=diagnostics["placement"], output=output / "functional")
    policy = prepared["manifest"]["quality"]
    checks = {}
    for name, expected in prepared["reference"].items():
        observed = result["outputs"][name]
        shape = observed.shape == expected.shape and observed.dtype == expected.dtype
        checks[name] = {"passed": bool(shape and np.allclose(observed, expected, atol=policy["atol"], rtol=policy["rtol"])),
                        "shape_and_dtype_match": shape,
                        "max_abs_error": float(np.max(np.abs(observed.astype(np.float64)-expected.astype(np.float64)), initial=0)) if shape else None}
    scenario = prepared["manifest"]["scenario"]
    # Identical requests share a validated operation profile, not fabricated
    # independent observations. Every request's predicted latency is retained.
    modeled = tensor_platform.schedule([result["platform_execution"]["profile"]] * len(scenario["arrivals_ns"]),
                                       experiment["platform"], arrivals=scenario["arrivals_ns"],
                                       max_inflight=scenario["max_inflight"], state=scenario["initial_state"])
    unified.save(output / "schedule.json", modeled)
    unified.save(output / "numerical.json", checks)
    def metric(value, unit):
        return {"value": value, "unit": unit, "scope": "declared_request_set_including_host_transfers",
                "aggregation": "makespan" if unit == "ns" else "sum", "origin": "declared_device_and_fabric_models"}
    metrics = {"latency": metric(modeled["modeled_latency_ns"], "ns")}
    if modeled["modeled_energy_pj"] is not None:
        metrics["energy"] = metric(modeled["modeled_energy_pj"], "pJ")
    return {"execution_mode": "shared_tensor_graph", "metrics": metrics,
            "quality": {"numerical": {"status": "passed" if all(c["passed"] for c in checks.values()) else "failed", "checks": checks},
                        "task": {"status": "not_evaluated", "reason": "generic numerical reference is not a task dataset evaluator"}},
            "platform_identity": result["platform_execution"]["platform_identity"],
            "cost_contract": {"schema": "tensor-resource-model/v1", "scope": "host transfers, compute, packing, live memory, device idle energy",
                              "scheduler": modeled["scheduler"], "memory_policy": modeled["memory_policy"],
                              "state": scenario["initial_state"], "cost_model": "supplied models; non-matmul tensor-element proxy; uncalibrated"},
            "execution_identity": {"placement": diagnostics["placement"], "numpy_version": np.__version__,
                                   "runtime_bindings": result["platform_execution"]["runtime_bindings"]},
            "diagnostics": diagnostics, "hardware_execution": False}

#!/usr/bin/env python3
"""Run one actual-tensor graph under four explicit, illustrative placements.

Requires the installed pim-foundry[tensor] extra. No downloads or native
executables are needed. Device latency is predicted from supplied coefficients;
functional tensors execute on the host. Existing output directories are refused.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from pim_foundry import application, tensor_platform


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rounded(value, code):
    """IEEE binary16/binary32 round-to-nearest via Python's scalar format."""
    return struct.unpack("!" + code, struct.pack("!" + code, value))[0]


def fixture(directory):
    directory.mkdir()
    w1 = [[1, 0, -1, 0], [0, 1, 0, -1], [1, 0, 1, 0], [0, 1, 0, 1]]
    w2 = [[0.25, 0], [0, 0.25], [0.25, 0.25], [-0.25, 0.25]]
    graph = {
        "schema": "pim-foundry.tensor-graph.v2", "inputs": ["input"],
        "outputs": ["probabilities"], "initializers": {"w1": w1, "w2": w2},
        "values": {}, "nodes": [],
    }
    for name, shape, dtype in (
        ("input", [2, 3, 4], "int8"), ("w1", [4, 4], "int8"),
        ("hidden_i32", [2, 3, 4], "int32"), ("relu_i32", [2, 3, 4], "int32"),
        ("hidden_f16", [2, 3, 4], "float16"), ("w2", [4, 2], "float16"),
        ("logits_f16", [2, 3, 2], "float16"), ("logits_f32", [2, 3, 2], "float32"),
        ("probabilities", [2, 3, 2], "float32"),
    ):
        graph["values"][name] = {"shape": shape, "dtype": dtype}
    for name, op, inputs, output, attrs in (
        ("integer_projection", "matmul", ["input", "w1"], "hidden_i32", {}),
        ("activation", "relu", ["hidden_i32"], "relu_i32", {}),
        ("to_fp16", "cast", ["relu_i32"], "hidden_f16", {"dtype": "float16"}),
        ("half_projection", "matmul", ["hidden_f16", "w2"], "logits_f16", {}),
        ("to_fp32", "cast", ["logits_f16"], "logits_f32", {"dtype": "float32"}),
        ("probabilities", "softmax", ["logits_f32"], "probabilities", {"axis": -1}),
    ):
        graph["nodes"].append({"name": name, "op": op, "inputs": inputs, "output": output, "attrs": attrs})
    data = (np.arange(24) % 5 - 2).astype(np.int8).reshape(2, 3, 4)
    # No runtime matmul/activation/softmax implementation is reused by the oracle.
    expected = []
    for batch in data.tolist():
        rows = []
        for row in batch:
            hidden = [rounded(max(0, sum(row[k] * w1[k][j] for k in range(4))), "e") for j in range(4)]
            logits = []
            for j in range(2):
                total = 0.0
                for k in range(4):
                    total = rounded(total + rounded(hidden[k] * w2[k][j], "f"), "f")
                logits.append(rounded(total, "e"))
            exponentials = [math.exp(x - max(logits)) for x in logits]
            rows.append([rounded(x / sum(exponentials), "f") for x in exponentials])
        expected.append(rows)
    save(directory / "graph.json", graph)
    np.savez(directory / "inputs.npz", input=data)
    np.savez(directory / "reference.npz", probabilities=np.asarray(expected, dtype=np.float32))
    assets = {}
    for key, role, suffix, format_name in (
        ("graph", "model", ".json", "pim-foundry.tensor-graph.v2"),
        ("inputs", "inputs", ".npz", "npz"), ("reference", "quality", ".npz", "npz"),
    ):
        assets[key] = application.asset(directory / (key + suffix), role, format_name)
        assets[key]["path"] = key + suffix  # The generated fixture is relocatable.
    manifest = {
        "schema": application.APP_SCHEMA, "id": "architecture-gallery-mixed-precision",
        "adapter": "tensor-graph/v2", "assets": assets, "parameters": {},
        "scenario": {"kind": "tensor_requests", "arrivals_ns": [0, 0, 0], "max_inflight": 3, "initial_state": "warm"},
        "quality": {"kind": "numerical_equivalence", "reference": "python_scalar_loops_struct_rounding_math_exp/v1", "atol": 1e-7, "rtol": 1e-6},
    }
    save(directory / "application.json", manifest)
    return graph


def platform_and_placements(graph):
    devices = [copy.deepcopy(tensor_platform.default_platform(kind)["devices"][0]) for kind in ("sram", "hbm", "gpu", "npu")]
    for device in devices:
        device["capacity_bytes"] = 4096
    endpoints = ["host", *[device["name"] for device in devices]]
    platform = {
        "schema_version": 1, "devices": devices,
        "links": [{"source": a, "destination": b, "bytes_per_ns": 16, "latency_ns": 5, "energy_pj_per_byte": 0}
                  for a in endpoints for b in endpoints if a != b],
    }
    gpu = {node["name"]: "gpu" for node in graph["nodes"]}
    return platform, {
        "gpu": gpu,
        "hbm_gpu": {**gpu, "half_projection": "hbm"},
        "hbm_npu": {node: "hbm" if node == "half_projection" else "npu" for node in gpu},
        "sram_hbm_gpu": {**gpu, "integer_projection": "sram", "half_projection": "hbm"},
    }


def rejected_schedule(profile, platform, expected_message):
    try:
        tensor_platform.schedule([profile] * 3, platform, arrivals=[0, 0, 0], max_inflight=3, state="warm")
    except ValueError as error:
        if expected_message not in str(error):
            raise
        return {"rejected": True, "reason": str(error)}
    raise AssertionError("invalid platform was accepted: " + expected_message)


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    graph = fixture(output / "fixture")
    platform, placements = platform_and_placements(graph)
    save(output / "platform.json", platform)
    scenarios = {}
    for name, placement in placements.items():
        directory = output / name
        directory.mkdir()
        experiment = {"schema": application.EXPERIMENT_SCHEMA, "platform": platform,
                      "mapping": {"strategy": "explicit", "placement": placement}, "variant": {}}
        save(directory / "experiment.json", experiment)
        result = application.run(output / "fixture/application.json", directory / "experiment.json", directory / "bundle")
        verified = application.verify(directory / "bundle")
        replay = application.replay(directory / "bundle", directory / "replay")
        schedule = json.loads((directory / "bundle/execution/schedule.json").read_text())
        profile = json.loads((directory / "bundle/execution/functional/platform-execution.json").read_text())["profile"]
        serial = tensor_platform.schedule([profile] * 3, platform, arrivals=[0, 0, 0], max_inflight=1, state="warm")
        transfers = [event for event in schedule["events"] if event["kind"] == "transfer"]
        by_route = {}
        for event in transfers:
            key = event["source"] + "->" + event["destination"]
            by_route[key] = by_route.get(key, 0) + event["bytes"]
        comparison = application.compare(directory / "bundle", output / "gpu/bundle")
        scenarios[name] = {
            "id": name, "label": {"gpu": "GPU", "hbm_gpu": "HBM-PIM + GPU", "hbm_npu": "HBM-PIM + NPU", "sram_hbm_gpu": "SRAM-CIM + HBM-PIM + GPU"}[name],
            "placement": placement, "status": result["status"], "numerical": result["quality"]["numerical"],
            "bundle_verified": verified, "replay": {"passed": replay["passed"], "checks": replay["checks"]},
            "bundle": name + "/bundle", "experiment": name + "/experiment.json",
            "task_sha256": result["task_sha256"], "platform_sha256": result["platform_sha256"],
            "comparison_to_all_gpu": comparison, "modeled_latency_ns": schedule["modeled_latency_ns"],
            "serial_reference": {"max_inflight": 1, "modeled_latency_ns": serial["modeled_latency_ns"],
                                 "scope": "same placement and arrivals, changed admission limit; modeled counterfactual"},
            "requests": schedule["requests"], "peak_device_bytes": schedule["peak_device_bytes"],
            "transfer_bytes": sum(event["bytes"] for event in transfers),
            "transfer_breakdown": {"total_link_bytes": sum(event["bytes"] for event in transfers),
                               "initializer_link_bytes": sum(event["bytes"] for event in transfers if event["tensor"] in graph["initializers"]),
                               "request_input_link_bytes": sum(event["bytes"] for event in transfers if event["tensor"] in graph["inputs"]),
                               "request_output_link_bytes": sum(event["bytes"] for event in transfers if event["tensor"] in graph["outputs"]),
                               "host_link_bytes": sum(event["bytes"] for event in transfers if "host" in (event["source"], event["destination"])),
                               "device_link_bytes": sum(event["bytes"] for event in transfers if "host" not in (event["source"], event["destination"])),
                               "by_route": by_route},
            "events": schedule["events"],
        }
    profile = json.loads((output / "sram_hbm_gpu/bundle/execution/functional/platform-execution.json").read_text())["profile"]
    disconnected = copy.deepcopy(platform)
    disconnected["links"] = []
    small = copy.deepcopy(platform)
    for device in small["devices"]:
        device["capacity_bytes"] = 1
    guards = {
        "missing_route": rejected_schedule(profile, disconnected, "route"),
        "capacity_exceeded": rejected_schedule(profile, small, "capacity"),
    }
    passed = all(row["status"] == "completed" and row["numerical"]["status"] == "passed"
                 and row["bundle_verified"]["passed"] and row["replay"]["passed"]
                 and row["comparison_to_all_gpu"]["same_model_comparable"] for row in scenarios.values())
    summary = {
        "schema": "pim-foundry.architecture-gallery.v1", "passed": passed,
        "workload": {"graph": "fixture/graph.json", "application": "fixture/application.json",
                     "input_shape": [2, 3, 4], "output_shape": [2, 3, 2],
                     "precision_flow": "INT8 matmul -> INT32 ReLU -> FP16 cast and matmul -> FP32 cast and softmax",
                     "arrivals_ns": [0, 0, 0], "max_inflight": 3, "state": "warm",
                     "weight_residency": "empty initially; first transfer populates a cache shared by subsequent requests",
                     "functional_evaluations_per_placement": 1,
                     "scheduled_identical_requests_per_placement": 3,
                     "reference": "Python scalar loops, explicit IEEE rounding and math.exp; no shared tensor operators"},
        "platform": platform, "scenarios": list(scenarios.values()), "rejection_checks": guards,
        "evidence": {"functional_execution": "actual tensors on the host via NumPy",
                     "timing": "deterministic resource schedule using equal illustrative analytical coefficients",
                     "shared_platform": "All four experiments declare the identical SRAM/HBM/GPU/NPU devices and directed links; only placement changes.",
                     "task_quality_evaluated": False, "hardware_execution": False, "hardware_calibrated": False,
                     "native_backends_executed": False, "energy_claim": False,
                     "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "numpy_version": np.__version__},
    }
    save(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for portable fixtures, experiments, sealed runs and summary")
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({"passed": result["passed"], "summary": str(args.out / "summary.json"),
                      "scenarios": {row["id"]: {key: row[key] for key in ("modeled_latency_ns", "numerical", "transfer_bytes", "peak_device_bytes")}
                                    for row in result["scenarios"]}}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

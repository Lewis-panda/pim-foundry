#!/usr/bin/env python3
"""Offline actual-tensor example with two explicitly illustrative device models.

Install pim-foundry[tensor] or pim-foundry[onnx] first. This script does not add
the checkout to sys.path, so it also tests an installed wheel outside the repo.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from pim_foundry import application, tensor_platform


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def make_fixture(output):
    output.mkdir(parents=True)
    graph = {"schema": "pim-foundry.tensor-graph.v2", "values": {},
             "inputs": ["input"], "outputs": ["probabilities"], "nodes": [],
             "initializers": {"w1": [[1, 0, -1, 0], [0, 1, 0, -1], [1, 0, 1, 0], [0, 1, 0, 1]],
                              "w2": [[1, 0], [0, 1], [1, 1], [-1, 1]]}}
    for name, shape in (("input", [2, 3, 4]), ("w1", [4, 4]), ("hidden", [2, 3, 4]),
                        ("activation", [2, 3, 4]), ("w2", [4, 2]), ("logits", [2, 3, 2]), ("probabilities", [2, 3, 2])):
        graph["values"][name] = {"shape": shape, "dtype": "float32"}
    for name, op, args, out, attrs in (("projection", "matmul", ["input", "w1"], "hidden", {}),
                                      ("activation", "relu", ["hidden"], "activation", {}),
                                      ("head", "matmul", ["activation", "w2"], "logits", {}),
                                      ("softmax", "softmax", ["logits"], "probabilities", {"axis": -1})):
        graph["nodes"].append({"name": name, "op": op, "inputs": args, "output": out, "attrs": attrs})
    inputs = np.arange(-12, 12, dtype=np.float32).reshape(2, 3, 4)
    expected = []
    for batch in inputs.tolist():
        rows = []
        for row in batch:
            hidden = [max(0, sum(row[k] * graph["initializers"]["w1"][k][j] for k in range(4))) for j in range(4)]
            logits = [sum(hidden[k] * graph["initializers"]["w2"][k][j] for k in range(4)) for j in range(2)]
            exps = [math.exp(x - max(logits)) for x in logits]
            rows.append([x / sum(exps) for x in exps])
        expected.append(rows)
    save(output / "graph.json", graph)
    np.savez(output / "inputs.npz", input=inputs)
    np.savez(output / "reference.npz", probabilities=np.asarray(expected, dtype=np.float32))
    manifest = {"schema": application.APP_SCHEMA, "id": "offline-batched-probabilities", "adapter": "tensor-graph/v2",
                "assets": {"graph": application.asset(output / "graph.json", "model", "pim-foundry.tensor-graph.v2"),
                           "inputs": application.asset(output / "inputs.npz", "inputs", "npz"),
                           "reference": application.asset(output / "reference.npz", "quality", "npz")},
                "parameters": {},
                "scenario": {"kind": "tensor_requests", "arrivals_ns": [0, 0, 50], "max_inflight": 2, "initial_state": "warm"},
                "quality": {"kind": "numerical_equivalence", "reference": "independent_python_loops_and_scalar_exp/v1", "atol": 1e-7, "rtol": 1e-6}}
    save(output / "application.json", manifest)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for all files and reports")
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    graph = make_fixture(output / "fixture")
    devices = []
    for kind in ("hbm", "gpu"):
        devices.append(copy.deepcopy(tensor_platform.default_platform(kind)["devices"][0]))
    # Identical coefficients deliberately avoid suggesting real device rankings.
    platform = {"schema_version": 1, "devices": devices,
                "links": [{"source": a, "destination": b, "bytes_per_ns": 16, "latency_ns": 5, "energy_pj_per_byte": 0}
                          for a in ("host", "hbm", "gpu") for b in ("host", "hbm", "gpu") if a != b]}
    placement = {node["name"]: "gpu" for node in graph["nodes"]}
    placement["projection"] = "hbm"
    experiment = {"schema": application.EXPERIMENT_SCHEMA, "platform": platform,
                  "mapping": {"strategy": "explicit", "placement": placement}, "variant": {}}
    save(output / "experiment.json", experiment)
    result = application.run(output / "fixture/application.json", output / "experiment.json", output / "application")
    verified = application.verify(output / "application")
    replay = application.replay(output / "application", output / "replay")
    baseline = copy.deepcopy(experiment)
    baseline["mapping"]["placement"] = {node["name"]: "gpu" for node in graph["nodes"]}
    save(output / "baseline-experiment.json", baseline)
    baseline_result = application.run(output / "fixture/application.json", output / "baseline-experiment.json", output / "baseline")
    comparison = application.compare(output / "application", output / "baseline")
    passed = result["status"] == baseline_result["status"] == "completed" and verified["passed"] and replay["passed"] and comparison["same_model_comparable"]
    summary = {"passed": passed, "actual_tensor_quality": result["quality"], "verified": verified, "replay": replay,
               "comparison": comparison, "platform": "HBM/GPU labels with identical illustrative analytical coefficients",
               "hardware_performance_claim": False, "report": str(output / "application/report.html")}
    save(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

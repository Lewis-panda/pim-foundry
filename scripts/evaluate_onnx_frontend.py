#!/usr/bin/env python3
"""Create real ONNX application files and evaluate their imported semantics.

Run with the optional ONNX environment. These are deterministic interoperability
fixtures, not trained accuracy benchmarks or paper performance reproductions.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from pim_foundry.onnx_frontend import load_model
from pim_foundry.typed import execute, graph_to_dict


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def make_model(dtype=T.FLOAT16, *, cast_input=False, flatten=False):
    """A 16 -> 24 -> 16 network independent of backend/device identifiers."""
    storage = np.float16 if dtype == T.FLOAT16 else np.float32
    rng = np.random.default_rng(713)
    w1 = rng.uniform(-0.5, 0.5, size=(16, 24)).astype(storage)
    w2 = rng.uniform(-0.5, 0.5, size=(24, 16)).astype(storage)
    bias = rng.uniform(-0.125, 0.125, size=24).astype(storage)
    nodes, current = [], "features"
    shape = ["batch", 4, 4] if flatten else ["batch", 16]
    if flatten:
        nodes.append(h.make_node("Flatten", [current], ["flat"], name="flatten", axis=1))
        current = "flat"
    if cast_input:
        nodes.append(h.make_node("Cast", [current], ["half_input"], name="input_cast", to=T.FLOAT16))
        current = "half_input"
    nodes.extend([h.make_node("Gemm", [current, "encoder_weight", "encoder_bias"], ["hidden"], name="encoder"),
                  h.make_node("Relu", ["hidden"], ["activated"], name="activation"),
                  h.make_node("MatMul", ["activated", "decoder_weight"], ["prediction"], name="decoder")])
    graph = h.make_graph(nodes, "portable_feedforward_16_24_16",
                         [h.make_tensor_value_info("features", T.INT8 if cast_input else dtype, shape)],
                         [h.make_tensor_value_info("prediction", dtype, ["batch", 16])],
                         [nh.from_array(w1, "encoder_weight"), nh.from_array(bias, "encoder_bias"),
                          nh.from_array(w2, "decoder_weight")])
    model = h.make_model(graph, producer_name="pim-foundry-interoperability-fixture", ir_version=10,
                         opset_imports=[h.make_opsetid("", 17)])
    onnx.checker.check_model(model, full_check=True)
    return model


def evaluate(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output directory already exists; use a new destination to preserve evidence")
    output.mkdir(parents=True)
    requirements = {name: importlib.metadata.version(name) for name in
                    ("onnx", "numpy", "protobuf", "typing_extensions", "ml_dtypes")}
    (output / "requirements.txt").write_text("".join(f"{name}=={version}\n" for name, version in requirements.items()))
    cases = []
    configurations = (("fp32-mlp", T.FLOAT, False, False), ("fp16-mlp", T.FLOAT16, False, False),
                      ("int8-input-mlp", T.FLOAT16, True, False), ("flatten-mlp", T.FLOAT16, False, True))
    for name, dtype, cast_input, flatten in configurations:
        directory = output / name
        directory.mkdir()
        model_path = directory / "model.onnx"
        onnx.save(make_model(dtype, cast_input=cast_input, flatten=flatten), model_path)
        checks = []
        for batch in (1, 2, 4):
            shape = [batch, 4, 4] if flatten else [batch, 16]
            imported = load_model(model_path, input_shapes={"features": shape})
            atol, rtol = ((2e-6, 2e-5) if dtype == T.FLOAT else (0.015625, 0.002))
            for seed in range(8):
                rng = np.random.default_rng(seed)
                values = (rng.integers(-4, 5, size=shape).tolist() if cast_input else
                          rng.uniform(-1, 1, size=shape).tolist())
                inputs = {"features": values}
                actual = execute(imported.graph, imported.prepare_inputs(inputs))
                check = imported.compare_reference(inputs, actual, atol=atol, rtol=rtol)
                checks.append({"batch": batch, "seed": seed, **check})
                if batch == 2 and seed == 0:
                    save(directory / "inputs.json", inputs)
                    save(directory / "input-shapes.json", {"features": shape})
                    save(directory / "graph.json", graph_to_dict(imported.graph))
                    save(directory / "import.json", imported.report)
                    save(directory / "outputs.json", actual)
                    save(directory / "reference.json", imported.reference_outputs(inputs))
                    save(directory / "quality.json", check)
            if not all(check["passed"] for check in checks):
                raise ValueError(f"independent ONNX reference mismatch in {name}")
        result = {"name": name, "model": str(model_path.relative_to(output)),
                  "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  "tested_input_sets": len(checks), "passed": all(check["passed"] for check in checks),
                  "compared_elements": sum(check["compared_elements"] for check in checks),
                  "max_absolute_error": max(check["max_absolute_error"] for check in checks), "checks": checks}
        save(directory / "evaluation.json", result)
        cases.append({key: value for key, value in result.items() if key != "checks"})
    summary = {"schema": "pim-foundry.onnx-frontend-evaluation.v1", "dependencies": requirements,
               "cases": cases, "passed": all(case["passed"] for case in cases),
               "total_input_sets": sum(case["tested_input_sets"] for case in cases),
               "total_compared_elements": sum(case["compared_elements"] for case in cases),
               "scope": "synthetic ONNX application interoperability and numerical checks, not a trained model accuracy or hardware performance claim",
               "independent_reference": "onnx.reference.ReferenceEvaluator", "hardware_measured": False}
    save(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    evaluate(parser.parse_args().output)

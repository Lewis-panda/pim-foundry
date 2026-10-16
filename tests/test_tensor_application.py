"""Sealed ONNX application bundles retain the complete classifier path."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from pim_foundry import application, tensor_platform, unified

AVAILABLE = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("numpy") is not None


def cnn_fixture(directory):
    """Create a deterministic interoperability fixture; no trained accuracy claim."""
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
    from onnx.reference import ReferenceEvaluator
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261015)
    weights = {"conv_w": rng.normal(0, 0.2, (2, 1, 3, 3)).astype(np.float32),
               "conv_b": np.array([0.1, -0.2], np.float32),
               "dense_w": rng.normal(0, 0.2, (8, 4)).astype(np.float32),
               "dense_b": np.array([0.1, 0, -0.1, 0.2], np.float32)}
    nodes = [helper.make_node("Conv", ["image", "conv_w", "conv_b"], ["convolution"]),
             helper.make_node("Relu", ["convolution"], ["activation"]),
             helper.make_node("AveragePool", ["activation"], ["pool"], kernel_shape=[2, 2], strides=[2, 2]),
             helper.make_node("Flatten", ["pool"], ["features"]),
             helper.make_node("Gemm", ["features", "dense_w", "dense_b"], ["logits"]),
             helper.make_node("Softmax", ["logits"], ["probabilities"], axis=-1)]
    model = helper.make_model(helper.make_graph(nodes, "complete_classifier_interop",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["B", 1, 6, 6])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["B", 4])],
        initializer=[onnx.numpy_helper.from_array(a, n) for n, a in weights.items()]),
        opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    image = rng.normal(size=(2, 1, 6, 6)).astype(np.float32)
    expected = ReferenceEvaluator(model).run(None, {"image": image})[0]
    np.savez(directory / "inputs.npz", image=image)
    np.savez(directory / "reference.npz", probabilities=expected)
    manifest = {"schema": application.APP_SCHEMA, "id": "complete-cnn-onnx-interoperability", "adapter": "onnx-tensor/v2",
                "assets": {"model": application.asset(directory / "model.onnx", "model", "onnx"),
                           "inputs": application.asset(directory / "inputs.npz", "inputs", "numpy-npz"),
                           "reference": application.asset(directory / "reference.npz", "quality", "numpy-npz")},
                "parameters": {}, "scenario": {"kind": "tensor_requests", "arrivals_ns": [0, 0, 30], "max_inflight": 2, "initial_state": "cold"},
                "quality": {"kind": "numerical_equivalence", "reference": "original_onnx.reference.ReferenceEvaluator", "atol": 1e-6, "rtol": 1e-5}}
    experiment = {"schema": application.EXPERIMENT_SCHEMA, "platform": tensor_platform.default_platform("gpu"),
                  "mapping": {"strategy": "default"}, "variant": {}}
    unified.save(directory / "application.json", manifest)
    unified.save(directory / "experiment.json", experiment)
    return directory / "application.json", directory / "experiment.json"


@unittest.skipUnless(AVAILABLE, "ONNX/NumPy optional dependencies unavailable")
class TensorApplicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.app, self.experiment = cnn_fixture(self.root / "fixture")

    def test_complete_onnx_bundle_verifies_and_replays(self):
        result = application.run(self.app, self.experiment, self.root / "run")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["quality"]["numerical"]["status"], "passed")
        self.assertEqual(result["quality"]["task"]["status"], "not_evaluated")
        self.assertFalse(result["hardware_performance_comparison_eligible"])
        self.assertTrue(application.verify(self.root / "run")["passed"])
        self.assertTrue(application.replay(self.root / "run", self.root / "replay")["passed"])
        self.assertTrue(application.verify(self.root / "replay")["passed"])
        profile = json.loads((self.root / "run" / "execution" / "functional" / "platform-execution.json").read_text())
        self.assertGreaterEqual(len(profile["nodes"]), 6)

    def test_wrong_independent_reference_is_quality_failure(self):
        import numpy as np
        np.savez(self.root / "fixture" / "reference.npz", probabilities=np.zeros((2, 4), np.float32))
        manifest = json.loads(self.app.read_text())
        manifest["assets"]["reference"] = application.asset(self.root / "fixture" / "reference.npz", "quality", "numpy-npz")
        unified.save(self.app, manifest)
        result = application.run(self.app, self.experiment, self.root / "run")
        self.assertEqual(result["status"], "quality_failed")
        self.assertEqual(result["quality"]["numerical"]["status"], "failed")
        self.assertFalse(application.compare_results(result, result)["same_model_comparable"])

    def test_source_model_tampering_is_detected_before_run(self):
        with (self.root / "fixture" / "model.onnx").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "modified application asset"):
            application.run(self.app, self.experiment, self.root / "run")


if __name__ == "__main__":
    unittest.main()

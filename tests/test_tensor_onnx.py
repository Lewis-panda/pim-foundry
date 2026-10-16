"""Independent original-ONNX versus imported common-IR application checks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from pim_foundry import tensor_onnx

AVAILABLE = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(AVAILABLE, "ONNX/NumPy optional dependencies unavailable")
class TensorOnnxTests(unittest.TestCase):
    def setUp(self):
        import onnx
        import numpy as np
        self.onnx, self.np = onnx, np
        self.helper, self.T = onnx.helper, onnx.TensorProto
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.onnx"

    def value(self, name, shape, dtype=None):
        return self.helper.make_tensor_value_info(name, dtype or self.T.FLOAT, shape)

    def save(self, nodes, inputs, outputs, constants=None, opset=18):
        initializers = [self.onnx.numpy_helper.from_array(array, name) for name, array in (constants or {}).items()]
        model = self.helper.make_model(self.helper.make_graph(nodes, "test", inputs, outputs, initializer=initializers), opset_imports=[self.helper.make_opsetid("", opset)], ir_version=10)
        self.onnx.checker.check_model(model)
        self.onnx.save(model, self.path)
        return model

    def test_complete_cnn_with_gemm_matches_original_onnx(self):
        np, h = self.np, self.helper
        rng = np.random.default_rng(30)
        nodes = [h.make_node("Conv", ["image", "conv_w", "conv_b"], ["conv"]),
                 h.make_node("Relu", ["conv"], ["relu"]),
                 h.make_node("AveragePool", ["relu"], ["pool"], kernel_shape=[2, 2], strides=[2, 2]),
                 h.make_node("Flatten", ["pool"], ["flat"], axis=1),
                 h.make_node("Gemm", ["flat", "dense_w", "dense_b"], ["logits"], alpha=0.75, beta=1.25),
                 h.make_node("Softmax", ["logits"], ["probability"], axis=-1)]
        constants = {"conv_w": rng.normal(size=(2, 1, 3, 3)).astype(np.float32), "conv_b": rng.normal(size=(2,)).astype(np.float32),
                     "dense_w": rng.normal(size=(8, 3)).astype(np.float32), "dense_b": rng.normal(size=(3,)).astype(np.float32)}
        self.save(nodes, [self.value("image", ["B", 1, 6, 6])], [self.value("probability", ["B", 3])], constants)
        result = tensor_onnx.compare_reference(self.path, {"image": rng.normal(size=(2, 1, 6, 6)).astype(np.float32)})
        self.assertTrue(result["passed"], result)
        self.assertLess(result["max_abs_error"], 2e-6)
        self.assertFalse(result["hardware_performance_validated"])

    def test_attention_residual_normalization_matches_original(self):
        np, h = self.np, self.helper
        rng = np.random.default_rng(31)
        nodes = [h.make_node("Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2]),
                 h.make_node("MatMul", ["q", "kt"], ["score"]),
                 h.make_node("Mul", ["score", "scale"], ["scaled"]),
                 h.make_node("Add", ["scaled", "mask"], ["masked"]),
                 h.make_node("Softmax", ["masked"], ["attention"], axis=-1),
                 h.make_node("MatMul", ["attention", "v"], ["context"]),
                 h.make_node("Add", ["context", "q"], ["residual"]),
                 h.make_node("LayerNormalization", ["residual", "gamma", "beta"], ["y"], axis=-1, epsilon=1e-5)]
        constants = {"scale": np.array(0.5, np.float32), "mask": np.triu(np.full((4, 4), -10000, np.float32), 1), "gamma": np.ones((4,), np.float32), "beta": np.zeros((4,), np.float32)}
        self.save(nodes, [self.value(n, ["B", 2, 4, 4]) for n in ("q", "k", "v")], [self.value("y", ["B", 2, 4, 4])], constants)
        feeds = {n: rng.normal(size=(2, 2, 4, 4)).astype(np.float32) for n in ("q", "k", "v")}
        self.assertTrue(tensor_onnx.compare_reference(self.path, feeds)["passed"])

    def test_dynamic_int64_token_gather_checked_and_mapped(self):
        np, h = self.np, self.helper
        weights = np.arange(30, dtype=np.float32).reshape(10, 3)
        self.save([h.make_node("Gather", ["table", "input ids"], ["embedding output"], axis=0)],
                  [self.value("input ids", ["B", "T"], self.T.INT64)], [self.value("embedding output", ["B", "T", 3])], {"table": weights})
        graph = tensor_onnx.load_model(self.path)
        self.assertIn("input ids", graph["metadata"]["onnx"]["narrowed_int64_values"])
        self.assertTrue(tensor_onnx.compare_reference(self.path, {"input ids": np.array([[2, 3, 1]], np.int64)})["passed"])
        with self.assertRaisesRegex(ValueError, "range"):
            tensor_onnx.prepare_inputs(graph, {"input ids": np.array([[2 ** 40]], np.int64)})

    def test_runtime_shape_binding_and_constant_shape_program(self):
        np, h = self.np, self.helper
        nodes = [h.make_node("Shape", ["x"], ["shape"]), h.make_node("Gather", ["shape", "index"], ["batch"], axis=0),
                 h.make_node("Concat", ["batch", "tail"], ["newshape"], axis=0), h.make_node("Reshape", ["x", "newshape"], ["y"])]
        self.save(nodes, [self.value("x", ["B", 2, 3])], [self.value("y", ["B", 6])], {"index": np.array([0], np.int64), "tail": np.array([6], np.int64)})
        with self.assertRaisesRegex(ValueError, "concrete input_shapes"):
            tensor_onnx.load_model(self.path)
        feeds = {"x": np.arange(24, dtype=np.float32).reshape(4, 2, 3)}
        self.assertTrue(tensor_onnx.compare_reference(self.path, feeds, input_shapes={"x": [4, 2, 3]})["passed"])
        with self.assertRaisesRegex(ValueError, "static dimension"):
            tensor_onnx.load_model(self.path, {"x": [4, 3, 2]})

    def test_reshape_zero_copy_and_transpose(self):
        np, h = self.np, self.helper
        self.save([h.make_node("Reshape", ["x", "shape"], ["r"]), h.make_node("Transpose", ["r"], ["y"], perm=[1, 0])],
                  [self.value("x", ["B", 2, 3])], [self.value("y", [6, "B"])], {"shape": np.array([0, -1], np.int64)})
        self.assertTrue(tensor_onnx.compare_reference(self.path, {"x": np.arange(12, dtype=np.float32).reshape(2, 2, 3)})["passed"])

    def test_argmax_exact_output_and_bias_broadcast_gemm(self):
        np, h = self.np, self.helper
        self.save([h.make_node("Gemm", ["x", "w", "b"], ["logits"], transB=1), h.make_node("ArgMax", ["logits"], ["token"], axis=-1, keepdims=0)],
                  [self.value("x", [2, 3])], [self.value("token", [2], self.T.INT64)],
                  {"w": np.eye(3, dtype=np.float32), "b": np.array([[0, 2, 0]], np.float32)})
        report = tensor_onnx.compare_reference(self.path, {"x": np.array([[1, 0, 2], [10, 1, 1]], np.float32)})
        self.assertTrue(report["passed"])
        self.assertEqual(report["outputs"]["token"]["comparison"], "exact_integer")

    def test_actual_backend_wrong_output_is_detected(self):
        np, h = self.np, self.helper
        self.save([h.make_node("Relu", ["x"], ["y"])], [self.value("x", [2])], [self.value("y", [2])])
        result = tensor_onnx.compare_reference(self.path, {"x": np.array([-1, 2], np.float32)}, actual_outputs={"y": np.zeros((2,), np.float32)})
        self.assertFalse(result["passed"])
        self.assertEqual(result["max_abs_error"], 2)

    def test_reject_float_integer_cast_semantic_mismatch(self):
        np, h = self.np, self.helper
        self.save([h.make_node("Cast", ["x"], ["y"], to=self.T.INT32)], [self.value("x", [2])], [self.value("y", [2], self.T.INT32)])
        with self.assertRaisesRegex(ValueError, "truncation"):
            tensor_onnx.load_model(self.path)

    def test_reject_unsupported_operator_and_semantic_attributes(self):
        np, h = self.np, self.helper
        self.save([h.make_node("Sin", ["x"], ["y"])], [self.value("x", [2])], [self.value("y", [2])])
        with self.assertRaisesRegex(ValueError, "unsupported ONNX operator Sin"):
            tensor_onnx.load_model(self.path)
        self.save([h.make_node("ArgMax", ["x"], ["y"], select_last_index=1)], [self.value("x", [2])], [self.value("y", [1], self.T.INT64)])
        with self.assertRaisesRegex(ValueError, "tie selection"):
            tensor_onnx.load_model(self.path)

    def test_reject_int64_arithmetic_and_nonfinite_weights(self):
        np, h = self.np, self.helper
        self.save([h.make_node("Add", ["x", "w"], ["y"])], [self.value("x", [2], self.T.INT64)], [self.value("y", [2], self.T.INT64)], {"w": np.array([1, 2], np.int64)})
        with self.assertRaisesRegex(ValueError, "only as Gather indices"):
            tensor_onnx.load_model(self.path)
        self.save([h.make_node("Add", ["x", "w"], ["y"])], [self.value("x", [2])], [self.value("y", [2])], {"w": np.array([np.nan, 0], np.float32)})
        with self.assertRaisesRegex(ValueError, "nonfinite initializer"):
            tensor_onnx.load_model(self.path)

    def test_reject_external_initializers(self):
        np, h = self.np, self.helper
        model = self.save([h.make_node("MatMul", ["x", "w"], ["y"])], [self.value("x", [2, 2])], [self.value("y", [2, 2])], {"w": np.eye(2, dtype=np.float32)})
        self.onnx.save_model(model, self.path, save_as_external_data=True, all_tensors_to_one_file=True, location="weights.bin", size_threshold=0)
        with self.assertRaisesRegex(ValueError, "external initializer"):
            tensor_onnx.load_model(self.path)

    def test_global_pool_and_gelu_match_original(self):
        np, h = self.np, self.helper
        self.save([h.make_node("GlobalAveragePool", ["x"], ["p"]), h.make_node("Gelu", ["p"], ["y"], approximate="tanh")],
                  [self.value("x", [2, 3, 4, 4])], [self.value("y", [2, 3, 1, 1])], opset=20)
        x = np.random.default_rng(4).normal(size=(2, 3, 4, 4)).astype(np.float32)
        self.assertTrue(tensor_onnx.compare_reference(self.path, {"x": x})["passed"])


if __name__ == "__main__":
    unittest.main()

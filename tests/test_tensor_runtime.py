"""Tensor v2 semantics: independent equations, ONNX oracles, and rejection cases."""
import copy
import importlib.util
import math
import unittest

from pim_foundry.tensor_ir import SCHEMA, validate_graph
from pim_foundry.tensor_executor import execute

try:
    import numpy as np
except ImportError:
    np = None


def operation(op, arrays, *, attrs=None, dtype=None, output_shape=None):
    names = [f"x{i}" for i in range(len(arrays))]
    dtype = dtype or arrays[0].dtype.name
    values = {n: {"shape": list(a.shape), "dtype": a.dtype.name} for n, a in zip(names, arrays)}
    values["y"] = {"shape": output_shape, "dtype": dtype}
    return {"schema": SCHEMA, "values": values, "inputs": names, "outputs": ["y"],
            "nodes": [{"name": "op", "op": op, "inputs": names, "output": "y", "attrs": attrs or {}}]}


def run_op(op, arrays, shape, attrs=None, dtype=None):
    graph = operation(op, arrays, attrs=attrs, dtype=dtype, output_shape=shape)
    return execute(graph, {f"x{i}": a for i, a in enumerate(arrays)})["outputs"]["y"]


def scalar_attention(q, k, v, offset):
    """Independent double-precision scalar oracle; includes GQA and cache offset."""
    out = np.zeros(q.shape[:3] + (v.shape[3],), dtype=np.float64)
    repeat = q.shape[1] // k.shape[1]
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for t in range(q.shape[2]):
                key_indices = range(offset + t + 1)
                scores = [sum(float(q[b, h, t, d]) * float(k[b, h // repeat, s, d]) for d in range(q.shape[3])) / math.sqrt(q.shape[3]) for s in key_indices]
                maximum = max(scores)
                probabilities = [math.exp(s - maximum) for s in scores]
                denom = sum(probabilities)
                for d in range(v.shape[3]):
                    out[b, h, t, d] = sum(p / denom * float(v[b, h // repeat, s, d]) for s, p in zip(key_indices, probabilities))
    return out.astype(np.float32)


@unittest.skipIf(np is None, "NumPy optional dependency unavailable")
class TensorRuntimeTests(unittest.TestCase):
    def test_symbolic_shapes_and_explicit_state(self):
        graph = {"schema": SCHEMA, "values": {
            "cache": {"shape": ["B", "P", 2], "dtype": "float32"},
            "new": {"shape": ["B", "T", 2], "dtype": "float32"},
            "next_cache": {"shape": ["B", "S", 2], "dtype": "float32"}},
            "inputs": ["cache", "new"], "outputs": ["next_cache"],
            "shape_contract": {"B": {"min": 1, "max": 4}, "S": {"max": 8}},
            "nodes": [{"name": "append", "op": "concat", "inputs": ["cache", "new"], "output": "next_cache", "attrs": {"axis": 1}}]}
        result = execute(graph, {"cache": np.empty((1, 0, 2), np.float32), "new": np.array([[[2, 3]]], np.float32)})
        np.testing.assert_array_equal(result["outputs"]["next_cache"], [[[2, 3]]])
        self.assertEqual(result["symbols"], {"B": 1, "P": 0, "T": 1, "S": 1})
        self.assertFalse(result["hardware_performance_validated"])
        self.assertEqual(result["execution_mode"], "numpy_functional_reference")

    def test_batched_broadcast_matmul_against_scalar_loops(self):
        rng = np.random.default_rng(10)
        a = rng.uniform(-2, 2, (2, 1, 3, 4)).astype(np.float32)
        b = rng.uniform(-2, 2, (1, 5, 4, 2)).astype(np.float32)
        got = run_op("matmul", [a, b], [2, 5, 3, 2])
        expected = np.empty(got.shape, np.float32)
        for i in range(2):
            for j in range(5):
                for r in range(3):
                    for c in range(2):
                        expected[i, j, r, c] = sum(float(a[i, 0, r, k]) * float(b[0, j, k, c]) for k in range(4))
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-6)

    def test_vector_and_scalar_matmul_outputs(self):
        a = np.array([1, 2, 3], np.float32)
        self.assertEqual(run_op("matmul", [a, a], []).item(), 14)
        np.testing.assert_array_equal(run_op("matmul", [a, np.array([[1, 2], [3, 4], [5, 6]], np.float32)], [2]), [22, 28])

    def test_integer_matmul_exact_and_overflow_rejected(self):
        a = np.array([[127, -128]], np.int8)
        b = np.array([[127], [127]], np.int8)
        self.assertEqual(run_op("matmul", [a, b], [1, 1], dtype="int32").item(), -127)
        a = np.array([[2 ** 30, 2 ** 30]], np.int32)
        with self.assertRaisesRegex(ValueError, "representable"):
            run_op("matmul", [a, np.ones((2, 1), np.int32)], [1, 1], dtype="int32")
        with self.assertRaisesRegex(ValueError, "int64 accumulation"):
            run_op("matmul", [np.full((1, 3), -2 ** 31, np.int32), np.full((3, 1), -2 ** 31, np.int32)], [1, 1], dtype="int32")

    def test_half_storage_accumulates_in_float32(self):
        a = np.array([[2048, 1, -2048]], np.float16)
        b = np.ones((3, 1), np.float16)
        self.assertEqual(run_op("matmul", [a, b], [1, 1]).item(), 1)

    def test_softmax_stability_and_shift_invariance(self):
        a = np.array([[10000, 10001, 9999]], np.float32)
        p = run_op("softmax", [a], [1, 3])
        denominator = math.exp(-1) + 1 + math.exp(-2)
        np.testing.assert_allclose(p, [[math.exp(-1) / denominator, 1 / denominator, math.exp(-2) / denominator]], rtol=1e-6)
        np.testing.assert_array_equal(p, run_op("softmax", [a - 10000], [1, 3]))

    def test_causal_attention_matches_independent_scalar_oracle(self):
        rng = np.random.default_rng(4)
        q = rng.normal(size=(2, 4, 3, 6)).astype(np.float32)
        k = rng.normal(size=(2, 2, 5, 6)).astype(np.float32)
        v = rng.normal(size=(2, 2, 5, 7)).astype(np.float32)
        got = run_op("causal_attention", [q, k, v], [2, 4, 3, 7], {"query_offset": 2})
        np.testing.assert_allclose(got, scalar_attention(q, k, v, 2), rtol=2e-6, atol=3e-7)

    def test_full_attention_equals_incremental_cache_execution(self):
        rng = np.random.default_rng(5)
        q = rng.normal(size=(1, 2, 5, 4)).astype(np.float32)
        k = rng.normal(size=q.shape).astype(np.float32)
        v = rng.normal(size=q.shape).astype(np.float32)
        full = run_op("causal_attention", [q, k, v], list(q.shape))
        for position in range(5):
            incremental = run_op("causal_attention", [q[:, :, position:position + 1], k[:, :, :position + 1], v[:, :, :position + 1]], [1, 2, 1, 4], {"query_offset": position})
            np.testing.assert_allclose(incremental, full[:, :, position:position + 1], atol=2e-7, rtol=1e-6)

    def test_attention_future_tokens_cannot_affect_past_output(self):
        q = np.ones((1, 1, 3, 2), np.float32)
        k = q.copy()
        v = np.array([[[[1, 2], [3, 4], [1000, 2000]]]], np.float32)
        first = run_op("causal_attention", [q, k, v], list(q.shape))
        v[:, :, -1] = -10000
        second = run_op("causal_attention", [q, k, v], list(q.shape))
        np.testing.assert_array_equal(first[:, :, :2], second[:, :, :2])

    def test_layer_norm_and_rms_norm_equations(self):
        x = np.array([[1, 2, 4], [3, -5, 6]], np.float32)
        gamma = np.array([2, 3, 4], np.float32)
        beta = np.array([0.5, -1, 2], np.float32)
        for op in ("layer_norm", "rms_norm"):
            args = [x, gamma, beta] if op == "layer_norm" else [x, gamma]
            got = run_op(op, args, [2, 3], {"epsilon": 1e-4})
            expected = []
            for row in x.tolist():
                mean = sum(row) / len(row) if op == "layer_norm" else 0
                centered = [v - mean for v in row]
                scale = math.sqrt(sum(v * v for v in centered) / len(row) + 1e-4)
                expected.append([value / scale * float(gamma[i]) + (float(beta[i]) if op == "layer_norm" else 0) for i, value in enumerate(centered)])
            np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_embedding_gather_and_negative_index_policy(self):
        weight = np.array([[1, 2], [3, 4], [5, 6]], np.float32)
        np.testing.assert_array_equal(run_op("embedding", [weight, np.array([[2, 0]], np.int32)], [1, 2, 2]), [[[5, 6], [1, 2]]])
        np.testing.assert_array_equal(run_op("gather", [weight, np.array([-1], np.int32)], [1, 2]), [[5, 6]])
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            run_op("embedding", [weight, np.array([-1], np.int32)], [1, 2])

    def test_transpose_reshape_slice_concat(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        y = run_op("transpose", [x], [2, 4, 3], {"perm": [0, 2, 1]})
        np.testing.assert_array_equal(y[0], [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]])
        self.assertEqual(run_op("reshape", [y], [4, 6], {"shape": [-1, 6]}).shape, (4, 6))
        z = run_op("slice", [x], [2, 2, 4], {"starts": [0], "ends": [3], "steps": [2], "axes": [1]})
        np.testing.assert_array_equal(z[0], [[0, 1, 2, 3], [8, 9, 10, 11]])

    def test_rotary_both_conventions_and_unchanged_tail(self):
        x = np.array([[[[1, 2, 3, 4, 5, 6]]]], np.float32)
        cos, sin = np.zeros((2,), np.float32), np.ones((2,), np.float32)
        np.testing.assert_array_equal(run_op("rotary", [x, cos, sin], list(x.shape), {"rotary_dim": 4}), [[[[-3, -4, 1, 2, 5, 6]]]])
        np.testing.assert_array_equal(run_op("rotary", [x, cos, sin], list(x.shape), {"rotary_dim": 4, "interleaved": True}), [[[[-2, 1, -4, 3, 5, 6]]]])

    def test_activations_and_round_even_cast(self):
        x = np.array([-3, -1, 0, 1, 3], np.float32)
        got = run_op("gelu", [x], [5])
        expected = [0.5 * float(v) * (1 + math.erf(float(v) / math.sqrt(2))) for v in x]
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(run_op("silu", [x], [5]), [float(v) / (1 + math.exp(-float(v))) for v in x], atol=1e-7)
        np.testing.assert_array_equal(run_op("cast", [np.array([-2.5, -1.5, 1.5, 2.5], np.float32)], [4], dtype="int8"), [-2, -2, 2, 2])

    def test_grouped_convolution_known_result(self):
        x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], np.float32)
        w = np.array([[[[2]]], [[[3]]]], np.float32)
        got = run_op("conv2d", [x, w], [1, 2, 2, 2], {"groups": 2})
        np.testing.assert_array_equal(got, [[[[2, 4], [6, 8]], [[15, 18], [21, 24]]]])

    def test_argmax_greedy_first_tie_and_scalar_gelu(self):
        x = np.array([[2, 5, 5], [4, 3, 2]], np.float32)
        np.testing.assert_array_equal(run_op("argmax", [x], [2], dtype="int32"), [1, 0])
        np.testing.assert_array_equal(run_op("argmax", [x], [2, 1], {"keepdims": True}, dtype="int32"), [[1], [0]])
        self.assertAlmostEqual(run_op("gelu", [np.array(0.5, np.float32)], []).item(), 0.34573123, places=7)

    def test_pooling_padding_denominator(self):
        x = np.array([[[[2, 4], [6, 8]]]], np.float32)
        args = {"kernel": [2, 2], "pads": [1, 1, 0, 0]}
        np.testing.assert_array_equal(run_op("avg_pool2d", [x], [1, 1, 2, 2], args), [[[[2, 3], [4, 5]]]])
        np.testing.assert_array_equal(run_op("avg_pool2d", [x], [1, 1, 2, 2], dict(args, count_include_pad=True)), [[[[0.5, 1.5], [2, 5]]]])
        np.testing.assert_array_equal(run_op("max_pool2d", [x], [1, 1, 2, 2], args), [[[[2, 4], [6, 8]]]])

    def test_callback_actual_output_reaches_downstream_node(self):
        x = np.array([[-1, -2]], np.float32)
        graph = operation("matmul", [x, np.eye(2, dtype=np.float32)], output_shape=[1, 2])
        graph["values"]["z"] = {"shape": [1, 2], "dtype": "float32"}
        graph["nodes"].append({"name": "relu", "op": "relu", "inputs": ["y"], "output": "z"})
        graph["outputs"] = ["z"]
        def callback(node, arrays, spec, context):
            self.assertFalse(arrays[0].flags.writeable)
            if node["op"] == "matmul":
                self.assertEqual(context["output_shape"], (1, 2))
                return {"value": np.array([[-10, 7]], np.float32), "metadata": {"source": "test_backend"}}
        result = execute(graph, {"x0": x, "x1": np.eye(2, dtype=np.float32)}, node_callback=callback)
        np.testing.assert_array_equal(result["outputs"]["z"], [[0, 7]])
        self.assertEqual(result["trace"][0]["execution"], "backend_callback")
        self.assertEqual(result["trace"][1]["execution"], "numpy_reference")

    def test_callback_wrong_shape_dtype_or_nonfinite_rejected(self):
        x = np.array([1, 2], np.float32)
        graph = operation("relu", [x], output_shape=[2])
        for bad in (np.ones((3,), np.float32), np.ones((2,), np.float64), np.array([1, np.nan], np.float32)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    execute(graph, {"x0": x}, node_callback=lambda *args: bad)

    def test_reject_nonfinite_input_and_overflow(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            run_op("relu", [np.array([np.nan], np.float32)], [1])
        with self.assertRaisesRegex(ValueError, "overflow"):
            run_op("mul", [np.array([3e38], np.float32)] * 2, [1])
        with self.assertRaisesRegex(ValueError, "division by zero"):
            run_op("div", [np.array([1], np.float32), np.array([0], np.float32)], [1])
        with self.assertRaisesRegex(ValueError, "float16 overflow"):
            run_op("cast", [np.array([1e10], np.float32)], [1], dtype="float16")

    def test_reject_mixed_dtype_implicit_input_cast(self):
        with self.assertRaisesRegex(ValueError, "mixed input dtypes"):
            run_op("add", [np.array([1], np.float16), np.array([1], np.float32)], [1])
        graph = operation("relu", [np.array([1], np.float32)], output_shape=[1])
        with self.assertRaisesRegex(ValueError, "ndarray dtype"):
            execute(graph, {"x0": np.array([1], np.float64)})

    def test_reject_ssa_and_unbound_input(self):
        x = np.array([1], np.float32)
        graph = operation("relu", [x], output_shape=[1])
        graph["nodes"].append(dict(graph["nodes"][0], name="again"))
        with self.assertRaisesRegex(ValueError, "assigned once"):
            validate_graph(graph)
        graph["nodes"] = [dict(graph["nodes"][0], inputs=["missing"])]
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_graph(graph)

    def test_reject_symbol_conflicts_bounds_and_unbound_reshape(self):
        x = np.ones((2, 3), np.float32)
        graph = operation("relu", [x], output_shape=["B", 3])
        graph["values"]["x0"]["shape"] = ["B", 3]
        graph["shape_contract"] = {"B": {"max": 1}}
        with self.assertRaisesRegex(ValueError, "shape_contract"):
            execute(graph, {"x0": x})
        graph = operation("reshape", [x], output_shape=[6], attrs={"shape": ["unknown"]})
        with self.assertRaisesRegex(ValueError, "unbound shape symbol"):
            execute(graph, {"x0": x})
        graph = operation("relu", [x], output_shape=["B", "B"])
        with self.assertRaisesRegex(ValueError, "dimension symbol"):
            execute(graph, {"x0": x})

    def test_allocation_and_work_limits_precede_callback(self):
        x = np.ones((4, 4), np.float32)
        graph = operation("matmul", [x, x], output_shape=[4, 4])
        graph["limits"] = {"max_operations": 10}
        seen = []
        with self.assertRaisesRegex(ValueError, "operation budget"):
            execute(graph, {"x0": x, "x1": x}, node_callback=lambda *args: seen.append(True))
        self.assertEqual(seen, [])
        graph["limits"] = {"max_tensor_elements": 10}
        with self.assertRaisesRegex(ValueError, "allocation"):
            execute(graph, {"x0": x, "x1": x})

    def test_invalid_operator_attributes_and_indices_rejected(self):
        x = np.ones((2, 3), np.float32)
        for op, attrs in (("transpose", {"perm": [0, 0]}), ("reshape", {"shape": [-1, -1]}), ("softmax", {"axis": 2}), ("slice", {"starts": [0], "ends": [1], "steps": [0]})):
            with self.subTest(op=op):
                with self.assertRaises(ValueError):
                    run_op(op, [x], [2, 3], attrs)
        with self.assertRaisesRegex(ValueError, "unknown operation attribute"):
            run_op("relu", [x], [2, 3], {"typo": True})


@unittest.skipIf(np is None or importlib.util.find_spec("onnx") is None, "ONNX independent reference unavailable")
class TensorOnnxOracleTests(unittest.TestCase):
    def reference(self, op, arrays, shape, **attrs):
        import onnx
        from onnx import helper, TensorProto
        from onnx.reference import ReferenceEvaluator
        names = [f"x{i}" for i in range(len(arrays))]
        inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, list(a.shape)) for n, a in zip(names, arrays)]
        outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)]
        model = helper.make_model(helper.make_graph([helper.make_node(op, names, ["y"], **attrs)], "oracle", inputs, outputs), opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
        onnx.checker.check_model(model)
        return ReferenceEvaluator(model).run(None, dict(zip(names, arrays)))[0]

    def test_matmul_matches_onnx_reference(self):
        rng = np.random.default_rng(8)
        a, b = rng.normal(size=(2, 3, 4)).astype(np.float32), rng.normal(size=(4, 6)).astype(np.float32)
        np.testing.assert_allclose(run_op("matmul", [a, b], [2, 3, 6]), self.reference("MatMul", [a, b], [2, 3, 6]), atol=1e-6, rtol=1e-6)

    def test_layer_norm_matches_onnx_reference(self):
        rng = np.random.default_rng(9)
        x, gamma, beta = rng.normal(size=(2, 3, 4)).astype(np.float32), rng.normal(size=(4,)).astype(np.float32), rng.normal(size=(4,)).astype(np.float32)
        got = run_op("layer_norm", [x, gamma, beta], [2, 3, 4], {"epsilon": 1e-5})
        expected = self.reference("LayerNormalization", [x, gamma, beta], [2, 3, 4], epsilon=1e-5, axis=-1)
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=2e-6)

    def test_grouped_dilated_convolution_matches_onnx_reference(self):
        rng = np.random.default_rng(10)
        x, w, b = rng.normal(size=(2, 4, 7, 8)).astype(np.float32), rng.normal(size=(6, 2, 2, 3)).astype(np.float32), rng.normal(size=(6,)).astype(np.float32)
        attrs = {"groups": 2, "strides": [2, 1], "pads": [1, 0, 1, 2], "dilations": [2, 1]}
        got = run_op("conv2d", [x, w, b], [2, 6, 4, 8], attrs)
        expected = self.reference("Conv", [x, w, b], [2, 6, 4, 8], group=2, strides=[2, 1], pads=[1, 0, 1, 2], dilations=[2, 1])
        np.testing.assert_allclose(got, expected, atol=2e-6, rtol=2e-6)

    def test_pooling_matches_onnx_reference(self):
        x = np.arange(30, dtype=np.float32).reshape(1, 1, 5, 6) - 10
        attrs = {"kernel": [3, 2], "strides": [2, 2], "pads": [1, 0, 1, 0]}
        for op, onnx_op in (("avg_pool2d", "AveragePool"), ("max_pool2d", "MaxPool")):
            with self.subTest(op=op):
                got = run_op(op, [x], [1, 1, 3, 3], attrs)
                expected = self.reference(onnx_op, [x], [1, 1, 3, 3], kernel_shape=[3, 2], strides=[2, 2], pads=[1, 0, 1, 0])
                np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

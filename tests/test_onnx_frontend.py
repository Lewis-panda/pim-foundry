"""Optional ONNX frontend checks against ONNX's independent evaluator."""

from pathlib import Path
import tempfile
import unittest

from pim_foundry.onnx_frontend import OnnxImportError, load_model
from pim_foundry.typed import execute, graph_from_dict, graph_to_dict

try:
    import numpy as np
    import onnx
    from onnx import TensorProto as T, helper as h, numpy_helper as nh
except ImportError:
    onnx = None


@unittest.skipIf(onnx is None, "install optional ONNX dependencies to test model interoperability")
class OnnxFrontendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "network.onnx"

    def save(self, nodes, inputs, outputs, initializers=(), *, opset=17):
        model = h.make_model(h.make_graph(nodes, "network", inputs, outputs, list(initializers)),
                             opset_imports=[h.make_opsetid("", opset)], ir_version=10)
        onnx.save(model, self.path)
        return model

    def mlp(self, dtype=None, dynamic=False, gemm=True):
        dtype = T.FLOAT if dtype is None else dtype
        dt = np.float16 if dtype == T.FLOAT16 else np.float32
        w1 = (np.arange(20).reshape(4, 5) % 7 - 3).astype(dt) / 4
        w2 = (np.arange(15).reshape(5, 3) % 5 - 2).astype(dt) / 4
        b = np.array([0.25, -0.5, 0.0, 0.75, -0.25], dtype=dt)
        nodes = [h.make_node("Gemm", ["x", "w1", "b"], ["hidden"], name="dense")]
        if not gemm:
            nodes = [h.make_node("MatMul", ["x", "w1"], ["product"], name="dense"),
                     h.make_node("Add", ["product", "b"], ["hidden"], name="bias")]
        nodes += [h.make_node("Relu", ["hidden"], ["activated"], name="relu"),
                  h.make_node("MatMul", ["activated", "w2"], ["y"], name="project")]
        self.save(nodes, [h.make_tensor_value_info("x", dtype, ["batch" if dynamic else 2, 4])],
                  [h.make_tensor_value_info("y", dtype, ["batch" if dynamic else 2, 3])],
                  [nh.from_array(w1, "w1"), nh.from_array(w2, "w2"), nh.from_array(b, "b")])

    def assert_reference(self, imported, values, *, atol=0.0, rtol=0.0):
        actual = execute(imported.graph, imported.prepare_inputs(values))
        result = imported.compare_reference(values, actual, atol=atol, rtol=rtol)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["reference"], "onnx.reference.ReferenceEvaluator")
        self.assertFalse(imported.report["evidence"]["independent_numerical_check"])
        return actual

    def test_raw_initializer_bytes_fp32_mlp(self):
        self.mlp()
        imported = load_model(self.path)
        self.assert_reference(imported, {"x": [[-2.0, 0.5, 1.0, 2.0], [1.0, -1.0, 2.0, 0.25]]})
        self.assertEqual(imported.report["source_nodes"], 3)
        self.assertEqual(imported.report["ir_nodes"], 4)
        self.assertEqual(graph_from_dict(graph_to_dict(imported.graph)), imported.graph)
        self.assertEqual(imported.report["ir_constant_bytes"], 4 * (20 + 15 + 10))

    def test_fp16_mlp_storage_is_preserved(self):
        self.mlp(T.FLOAT16)
        imported = load_model(self.path)
        self.assertTrue(all(t.dtype == "float16" for t in imported.graph.tensors))
        self.assert_reference(imported, {"x": [1.0, 2.0, 3.0, 4.0, -1.0, 0.25, 1.0, -0.5]})
        self.assertFalse(imported.report["numeric_policy"]["implicit_precision_conversion"])

    def test_explicit_add_initializer_broadcast(self):
        self.mlp(gemm=False)
        imported = load_model(self.path)
        self.assert_reference(imported, {"x": [0.5] * 8})
        self.assertIn("materialize_broadcast_constant", imported.report["lowerings"][1]["transformations"])

    def test_dynamic_dimensions_require_binding(self):
        self.mlp(dynamic=True)
        with self.assertRaisesRegex(OnnxImportError, "bind a positive"):
            load_model(self.path)
        imported = load_model(self.path, input_shapes={"x": [3, 4]})
        self.assert_reference(imported, {"x": [0.5] * 12})
        self.assertEqual(imported.graph.tensor_map["y"].shape, (3, 3))

    def test_cannot_change_static_dimensions(self):
        self.mlp()
        with self.assertRaisesRegex(OnnxImportError, "changes a static dimension"):
            load_model(self.path, input_shapes={"x": [3, 4]})
        with self.assertRaisesRegex(OnnxImportError, "unknown input names"):
            load_model(self.path, input_shapes={"typo": [2, 4]})

    def test_input_shape_and_integer_validation(self):
        self.mlp()
        imported = load_model(self.path)
        with self.assertRaisesRegex(OnnxImportError, "expected source shape"):
            imported.prepare_inputs({"x": [[1.0] * 8]})
        with self.assertRaisesRegex(OnnxImportError, "exactly"):
            imported.prepare_inputs({})

    def test_boundary_flatten_changes_view_not_storage(self):
        w = np.arange(12, dtype=np.float32).reshape(4, 3) / 4
        self.save([h.make_node("Flatten", ["x"], ["flat"], name="flatten", axis=1),
                   h.make_node("MatMul", ["flat", "w"], ["y"], name="dense")],
                  [h.make_tensor_value_info("x", T.FLOAT, [2, 2, 2])],
                  [h.make_tensor_value_info("y", T.FLOAT, [2, 3])], [nh.from_array(w, "w")])
        imported = load_model(self.path)
        self.assertEqual(imported.input_schema["x"]["shape"], [2, 2, 2])
        self.assertEqual(imported.input_schema["x"]["ir_shape"], [2, 4])
        self.assert_reference(imported, {"x": np.arange(8, dtype=np.float32).reshape(2, 2, 2)})

    def test_flatten_shared_input_is_rejected(self):
        self.save([h.make_node("Flatten", ["x"], ["flat"], name="reshape", axis=0),
                   h.make_node("Relu", ["x"], ["y"], name="relu")],
                  [h.make_tensor_value_info("x", T.FLOAT, [2, 2])],
                  [h.make_tensor_value_info("flat", T.FLOAT, [1, 4]), h.make_tensor_value_info("y", T.FLOAT, [2, 2])])
        with self.assertRaisesRegex(OnnxImportError, "reshape.*Flatten.*unshared"):
            load_model(self.path)

    def test_identity_output_aliases_preserve_both_outputs(self):
        self.save([h.make_node("Relu", ["x"], ["a"]), h.make_node("Identity", ["a"], ["b"])],
                  [h.make_tensor_value_info("x", T.FLOAT, [2])],
                  [h.make_tensor_value_info("a", T.FLOAT, [2]), h.make_tensor_value_info("b", T.FLOAT, [2])])
        imported = load_model(self.path)
        self.assertEqual(imported.graph.outputs, ("a",))
        self.assert_reference(imported, {"x": [-1.0, 2.0]})

    def test_transpose_constant_gemm_weight(self):
        self.save([h.make_node("Gemm", ["x", "w"], ["y"], name="linear", transB=1)],
                  [h.make_tensor_value_info("x", T.FLOAT16, [2, 3])],
                  [h.make_tensor_value_info("y", T.FLOAT16, [2, 4])],
                  [nh.from_array(np.arange(12, dtype=np.float16).reshape(4, 3), "w")])
        imported = load_model(self.path)
        self.assert_reference(imported, {"x": [1.0, -1.0, 0.5, 2.0, 3.0, 1.0]})
        self.assertIn("transpose_constant_weight", imported.report["lowerings"][0]["transformations"])

    def test_gemm_scaling_cannot_be_silently_dropped(self):
        self.mlp()
        model = onnx.load(self.path)
        model.graph.node[0].attribute.extend([h.make_attribute("alpha", 0.5)])
        onnx.save(model, self.path)
        with self.assertRaisesRegex(OnnxImportError, "dense.*Gemm.*alpha=beta"):
            load_model(self.path)

    def test_explicit_int8_to_half_cast(self):
        self.save([h.make_node("Cast", ["x"], ["half"], name="declared_cast", to=T.FLOAT16),
                   h.make_node("MatMul", ["half", "w"], ["y"], name="linear")],
                  [h.make_tensor_value_info("x", T.INT8, [2, 2])],
                  [h.make_tensor_value_info("y", T.FLOAT16, [2, 2])],
                  [nh.from_array(np.array([[1, 2], [-1, 0]], dtype=np.float16), "w")])
        imported = load_model(self.path)
        self.assert_reference(imported, {"x": [-2, 3, 127, -128]})
        self.assertEqual(len(imported.report["precision_conversions"]), 1)
        with self.assertRaisesRegex(ValueError, "int8 value"):
            imported.prepare_inputs({"x": [128, 0, 0, 0]})

    def test_float_to_integer_cast_rejects_rounding_mismatch(self):
        self.save([h.make_node("Cast", ["x"], ["y"], name="integer_cast", to=T.INT32)],
                  [h.make_tensor_value_info("x", T.FLOAT, [2])],
                  [h.make_tensor_value_info("y", T.INT32, [2])])
        with self.assertRaisesRegex(OnnxImportError, "integer_cast.*truncation/wrapping"):
            load_model(self.path)

    def test_int8_add_rejects_wrong_overflow_semantics(self):
        self.save([h.make_node("Add", ["x", "x"], ["y"], name="overflow")],
                  [h.make_tensor_value_info("x", T.INT8, [2])],
                  [h.make_tensor_value_info("y", T.INT8, [2])])
        with self.assertRaisesRegex(OnnxImportError, "overflow.*wrapping"):
            load_model(self.path)

    def test_int32_add_wrap_matches_onnx(self):
        self.save([h.make_node("Add", ["x", "b"], ["y"], name="sum")],
                  [h.make_tensor_value_info("x", T.INT32, [2])],
                  [h.make_tensor_value_info("y", T.INT32, [2])],
                  [nh.from_array(np.array([1, -1], dtype=np.int32), "b")])
        imported = load_model(self.path)
        self.assert_reference(imported, {"x": [2**31 - 1, -(2**31)]})

    def test_unsupported_operator_reports_source_node(self):
        self.save([h.make_node("Sigmoid", ["x"], ["y"], name="activation")],
                  [h.make_tensor_value_info("x", T.FLOAT, [2])],
                  [h.make_tensor_value_info("y", T.FLOAT, [2])])
        with self.assertRaisesRegex(OnnxImportError, "node 0 'activation' \\(Sigmoid\\).*unsupported"):
            load_model(self.path)

    def test_external_initializers_rejected_before_file_access(self):
        self.mlp()
        model = onnx.load(self.path)
        value = model.graph.initializer[0]
        value.ClearField("raw_data")
        value.data_location = T.EXTERNAL
        item = value.external_data.add()
        item.key, item.value = "location", "../../outside.bin"
        self.path.write_bytes(model.SerializeToString())
        with self.assertRaisesRegex(OnnxImportError, "external data"):
            load_model(self.path)

    def test_runtime_weight_rejected_by_supported_application_subset(self):
        self.save([h.make_node("MatMul", ["x", "w"], ["y"], name="dynamic_weight")],
                  [h.make_tensor_value_info("x", T.FLOAT, [2, 3]), h.make_tensor_value_info("w", T.FLOAT, [3, 2])],
                  [h.make_tensor_value_info("y", T.FLOAT, [2, 2])])
        with self.assertRaisesRegex(OnnxImportError, "frozen initializers"):
            load_model(self.path)

    def test_runtime_broadcast_is_not_silently_removed(self):
        self.save([h.make_node("Add", ["x", "b"], ["y"], name="broadcast")],
                  [h.make_tensor_value_info("x", T.FLOAT, [2, 3]), h.make_tensor_value_info("b", T.FLOAT, [3])],
                  [h.make_tensor_value_info("y", T.FLOAT, [2, 3])])
        with self.assertRaisesRegex(OnnxImportError, "broadcast.*runtime tensor"):
            load_model(self.path)

    def test_nonfinite_initializer_rejected(self):
        self.save([h.make_node("Add", ["x", "b"], ["y"])],
                  [h.make_tensor_value_info("x", T.FLOAT, [1])],
                  [h.make_tensor_value_info("y", T.FLOAT, [1])],
                  [nh.from_array(np.array([np.inf], dtype=np.float32), "b")])
        with self.assertRaisesRegex(OnnxImportError, "nonfinite"):
            load_model(self.path)

    def test_corrupted_actual_output_fails_independent_quality(self):
        self.mlp()
        imported = load_model(self.path)
        values = {"x": [1.0] * 8}
        actual = execute(imported.graph, imported.prepare_inputs(values))
        bad = {"y": tuple(value + 1.0 for value in actual["y"])}
        self.assertFalse(imported.compare_reference(values, bad, atol=0.0, rtol=0.0)["passed"])

    def test_source_bytes_are_bound_even_if_file_changes(self):
        self.mlp()
        imported = load_model(self.path)
        self.path.write_bytes(b"changed")
        self.assert_reference(imported, {"x": [0.25] * 8})


if __name__ == "__main__":
    unittest.main()

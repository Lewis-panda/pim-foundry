"""Independent arithmetic examples and rejection gates for the typed contract."""

import copy
import json
import math
import unittest

from pim_foundry.typed import (
    Arithmetic, BackendCapabilities, Graph, Node, Requantization, Tensor,
    compare_outputs, evaluate, execute, execute_node, graph_digest,
    graph_from_dict, graph_to_dict, round_float,
)


def linear(dtype="int8", output="int32", arithmetic=None, shape=(2,), weights=(1, 2, 3, 4)):
    arithmetic = arithmetic or Arithmetic()
    k = shape[-1]
    n = len(weights) // k
    result_shape = (*shape[:-1], n)
    return Graph(
        (Tensor("x", shape, dtype), Tensor("w", (k, n), dtype, tuple(weights)), Tensor("y", result_shape, output)),
        (Node("linear", "matvec" if len(shape) == 1 else "matmul", ("x", "w"), "y", arithmetic),),
        ("x",), ("y",),
    )


def unary(source, destination, op="cast", arithmetic=None):
    return Graph((Tensor("x", (3,), source), Tensor("y", (3,), destination)),
                 (Node("convert", op, ("x",), "y", arithmetic or Arithmetic()),), ("x",), ("y",))


class TypedNumericsTests(unittest.TestCase):
    def test_signed_int8_matvec_known_values(self):
        graph = linear(weights=(-2, 5, 3, -4))
        self.assertEqual(execute(graph, {"x": (-7, 2)}), {"y": (20, -43)})
        self.assertEqual(graph.tensor_map["w"].nbytes, 4)
        self.assertEqual(graph.tensor_map["y"].nbytes, 8)

    def test_row_major_matmul_known_values(self):
        graph = linear(shape=(2, 2), weights=(-2, 5, 3, -4))
        self.assertEqual(execute(graph, {"x": (-7, 2, 3, -1)}), {"y": (20, -43, -9, 19)})

    def test_runtime_weights_are_supported(self):
        graph = Graph((Tensor("x", (2,), "int8"), Tensor("w", (2, 1), "int8"), Tensor("y", (1,), "int32")),
                      (Node("dot", "matvec", ("x", "w"), "y"),), ("x", "w"), ("y",))
        self.assertEqual(execute(graph, {"x": (3, -2), "w": (7, 4)}), {"y": (13,)})

    def test_fanout_duplicate_add_and_relu(self):
        graph = Graph((Tensor("x", (2,), "int8"), Tensor("w", (2, 2), "int8", (-2, 5, 3, -4)),
                       Tensor("a", (2,), "int32"), Tensor("b", (2,), "int32"), Tensor("c", (2,), "int32")),
                      (Node("linear", "matvec", ("x", "w"), "a"), Node("double", "add", ("a", "a"), "b"),
                       Node("activation", "relu", ("a",), "c")), ("x",), ("b", "c"))
        values = evaluate(graph, {"x": (-7, 2)})
        self.assertEqual(values["a"], (20, -43))
        self.assertEqual(values["b"], (40, -86))
        self.assertEqual(values["c"], (20, 0))

    def test_int32_add_overflow_policies(self):
        for policy, expected in (("error", None), ("wrap", -(2**31)), ("saturate", 2**31 - 1)):
            graph = Graph((Tensor("x", (1,), "int32"), Tensor("b", (1,), "int32", (1,)), Tensor("y", (1,), "int32")),
                          (Node("sum", "add", ("x", "b"), "y", Arithmetic(overflow=policy)),), ("x",), ("y",))
            if expected is None:
                with self.assertRaisesRegex(ValueError, "overflow"):
                    execute(graph, {"x": (2**31 - 1,)})
            else:
                self.assertEqual(execute(graph, {"x": (2**31 - 1,)}), {"y": (expected,)})

    def test_int32_mac_overflow_at_intermediate_accumulation(self):
        # 131072 products of (-128)*(-128) reach 2**31; a last negative
        # product brings the exact final sum back into range. Error policy must
        # reject the intermediate overflow, rather than only check the output.
        k = 131073
        weights = (-128,) * (k - 1) + (127,)
        x = (-128,) * k
        with self.assertRaisesRegex(ValueError, "overflow"):
            execute(linear(shape=(k,), weights=weights), {"x": x})
        saturated = linear(shape=(k,), weights=weights, arithmetic=Arithmetic(overflow="saturate"))
        wrapped = linear(shape=(k,), weights=weights, arithmetic=Arithmetic(overflow="wrap"))
        self.assertEqual(execute(saturated, {"x": x})["y"], (2147467391,))
        self.assertEqual(execute(wrapped, {"x": x})["y"], (2147467392,))

    def test_requantization_negative_ties_and_clipping(self):
        quantization = Requantization(1, 1, 0)
        self.assertEqual([quantization.apply(x) for x in (-5, -3, -1, 1, 3, 5)], [-2, -2, 0, 0, 2, 2])
        self.assertEqual(Requantization(3, 0, 7).apply(1000), 127)
        self.assertEqual(Requantization(3, 0, 7).apply(-1000), -128)
        graph = linear(output="int8", arithmetic=Arithmetic(requantize=Requantization(1, 2)), weights=(1, 0, 0, 1))
        self.assertEqual(execute(graph, {"x": (6, -10)}), {"y": (2, -2)})

    def test_requantization_float_cast_exact_ties(self):
        graph = unary("float16", "int8", arithmetic=Arithmetic(requantize=Requantization(1, 1, 1)))
        self.assertEqual(execute(graph, {"x": (3.0, -3.0, 1000.0)}), {"y": (3, -1, 127)})

    def test_explicit_integer_float_cast_chain(self):
        graph = Graph((Tensor("x", (3,), "int32"), Tensor("f", (3,), "float16"), Tensor("q", (3,), "int8")),
                      (Node("to_half", "cast", ("x",), "f", Arithmetic("float32")),
                       Node("to_byte", "cast", ("f",), "q", Arithmetic(requantize=Requantization(1, 1)))),
                      ("x",), ("f", "q"))
        self.assertEqual(execute(graph, {"x": (2049, -3, 5)}), {"f": (2048.0, -3.0, 5.0), "q": (127, -2, 2)})

    def test_float_to_int32_rounding_overflow(self):
        graph = unary("float32", "int32")
        self.assertEqual(execute(graph, {"x": (1.5, 2.5, -1.5)}), {"y": (2, 2, -2)})
        with self.assertRaisesRegex(ValueError, "overflow"):
            execute(graph, {"x": (2**31, 0.0, 0.0)})

    def test_half_input_rounding_and_output_rounding(self):
        graph = linear("float16", "float16", Arithmetic("float32"), weights=(1.0, 0.0, 0.0, 1.0))
        self.assertEqual(execute(graph, {"x": (1 + 2**-11, 1 + 3 * 2**-11)}), {"y": (1.0, 1 + 2**-9)})
        self.assertEqual(round_float(2**-25, "float16"), 0.0)
        self.assertEqual(round_float(3 * 2**-25, "float16"), 2**-23)
        self.assertEqual(math.copysign(1, round_float(-2**-26, "float16")), -1)

    def test_fp32_accumulation_preserves_half_product(self):
        graph = linear("float16", "float32", Arithmetic("float32"), shape=(1,), weights=(1 + 2**-10,))
        self.assertEqual(execute(graph, {"x": (1 + 2**-10,)}), {"y": (1 + 2**-9 + 2**-20,)})

    def test_nonfused_fp32_and_fixed_reduction_order(self):
        graph = linear("float32", "float32", Arithmetic("float32"), weights=(1.0, 1 - 2**-23))
        # A fused second MAC would produce -2**-46; separate rounded product is 1.
        self.assertEqual(execute(graph, {"x": (-1.0, 1 + 2**-23)}), {"y": (0.0,)})
        ordered = linear("float32", "float32", Arithmetic("float32"), shape=(3,), weights=(1.0, 1.0, 1.0))
        self.assertEqual(execute(ordered, {"x": (2**24, 1.0, -(2**24))}), {"y": (0.0,)})

    def test_nonfinite_and_fp16_overflow_rejected(self):
        graph = linear("float16", "float16", Arithmetic("float32"), shape=(1,), weights=(2.0,))
        for value in (float("inf"), float("nan"), 65520.0):
            with self.assertRaises(ValueError):
                execute(graph, {"x": (value,)})
        with self.assertRaises(ValueError):
            execute(graph, {"x": (65504.0,)})
        with self.assertRaises(ValueError):
            round_float(1e1000, "float32")

    def test_relu_positive_zero(self):
        graph = unary("float16", "float16", "relu", Arithmetic("float32"))
        result = execute(graph, {"x": (-0.0, -1.0, 2.0)})["y"]
        self.assertEqual(result, (0.0, 0.0, 2.0))
        self.assertEqual(math.copysign(1, result[0]), 1.0)

    def test_exact_integer_comparison_ignores_tolerance(self):
        graph = linear()
        result = compare_outputs(graph, {"y": (1, 3)}, {"y": (1, 2)}, atol=100)
        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatches"][0]["index"], 1)

    def test_float_comparison_storage_gate_and_tolerance(self):
        graph = linear("float16", "float16", Arithmetic("float32"))
        self.assertTrue(compare_outputs(graph, {"y": (1 + 2**-10, 2.0)}, {"y": (1.0, 2.0)}, atol=0.001)["passed"])
        self.assertFalse(compare_outputs(graph, {"y": (1 + 2**-12, 2.0)}, {"y": (1.0, 2.0)}, atol=0.001)["passed"])
        self.assertFalse(compare_outputs(graph, {"y": (float("nan"), 2.0)}, {"y": (1.0, 2.0)})["passed"])

    def test_json_roundtrip_and_digest_bind_arithmetic(self):
        graph = linear(output="int8", arithmetic=Arithmetic(requantize=Requantization(3, 4, -1)))
        restored = graph_from_dict(json.loads(json.dumps(graph_to_dict(graph))))
        self.assertEqual(restored, graph)
        self.assertEqual(graph_digest(restored), graph_digest(graph))
        changed = copy.deepcopy(graph_to_dict(graph))
        changed["nodes"][0]["arithmetic"]["requantize"]["shift"] = 5
        self.assertNotEqual(graph_digest(graph_from_dict(changed)), graph_digest(graph))

    def test_backend_capability_rejection(self):
        capability = BackendCapabilities("int8-sram", ("matvec",), ("int8",), ("int32",), ("int32",))
        graph = linear()
        self.assertEqual(capability.check(graph.nodes[0], graph.tensor_map), ())
        fp = linear("float16", "float16", Arithmetic("float32"))
        self.assertEqual(len(capability.check(fp.nodes[0], fp.tensor_map)), 3)

    def test_rejects_implicit_casts(self):
        for graph in (linear(output="int8"), linear("float16", "float16"), unary("int32", "float16"), unary("float16", "int8")):
            with self.assertRaises(ValueError):
                graph.validate()
        graph = Graph((Tensor("x", (2,), "int8"), Tensor("w", (2, 1), "float16", (1.0, 2.0)), Tensor("y", (1,), "int32")),
                      (Node("linear", "matvec", ("x", "w"), "y"),), ("x",), ("y",))
        with self.assertRaisesRegex(ValueError, "mixed"):
            graph.validate()

    def test_rejects_invalid_scalar_storage(self):
        for data in ((128,), (-129,), (True,), (1.0,)):
            with self.assertRaises(ValueError):
                Tensor("bad", (1,), "int8", data)
        with self.assertRaises(ValueError):
            Tensor("bad", (1,), "float16", (float("nan"),))
        with self.assertRaises(ValueError):
            Tensor("bad", (0,), "int8")
        with self.assertRaises(ValueError):
            Tensor("bad", (True,), "int8")

    def test_rejects_bad_shapes_ssa_and_forward_references(self):
        original = graph_to_dict(linear())
        variants = []
        variant = copy.deepcopy(original)
        variant["tensors"][2]["shape"] = [3]
        variants.append(variant)
        variant = copy.deepcopy(original)
        variant["nodes"][0]["output"] = "x"
        variants.append(variant)
        variant = copy.deepcopy(original)
        variant["nodes"][0]["inputs"] = ["y", "w"]
        variants.append(variant)
        variant = copy.deepcopy(original)
        variant["inputs"] = ["x", "x"]
        variants.append(variant)
        for variant in variants:
            with self.assertRaises(ValueError):
                graph_from_dict(variant)

    def test_rejects_unknown_json_fields_and_wrong_arrays(self):
        for mutate in (lambda v: v.update(extra=True), lambda v: v["nodes"][0]["arithmetic"].update(fma=True),
                       lambda v: v.update(inputs="x"), lambda v: v["tensors"][0].update(shape="2")):
            data = graph_to_dict(linear())
            mutate(data)
            with self.assertRaises(ValueError):
                graph_from_dict(data)

    def test_input_mapping_shape_and_bounded_execution(self):
        graph = linear()
        for inputs in ({}, {"x": (1,)}, {"x": (1, 2), "w": (1, 2, 3, 4)}, {"x": (True, 1)}):
            with self.assertRaises(ValueError):
                execute(graph, inputs)
        with self.assertRaisesRegex(ValueError, "operation limit"):
            execute(graph, {"x": (1, 2)}, max_operations=3)
        with self.assertRaises(ValueError):
            execute_node(graph.nodes[0], graph.tensor_map, {"x": (1, 2), "w": (1, 2)})

    def test_policy_rejection(self):
        for kwargs in ({"nonfinite": "propagate"}, {"rounding": "toward_zero"}, {"mac": "fused"},
                       {"accumulator": "float16"}, {"accumulator": "float32", "overflow": "wrap"}):
            with self.assertRaises(ValueError):
                Arithmetic(**kwargs)
        for kwargs in ({"multiplier": 0}, {"shift": -1}, {"zero_point": 128}, {"shift": True}):
            with self.assertRaises(ValueError):
                Requantization(**kwargs)


if __name__ == "__main__":
    unittest.main()

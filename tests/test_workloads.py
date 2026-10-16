"""Synthetic suite reproducibility and strict external-workload boundaries."""

from copy import deepcopy
from dataclasses import replace
import json
from math import prod
import random
import unittest

from pim_foundry.benchmark import content_hash
from pim_foundry.core import Graph, Node, Platform, ReferenceDevice, Tensor
from pim_foundry.research import execute, scalar_oracle
from pim_foundry.workloads import (
    WORKLOAD_NAMES, WORKLOAD_SUITE_VERSION, make_workload, workload_from_dict,
    workload_identity, workload_to_dict,
)


def custom_graph():
    """Two runtime inputs, a matrix branch, addition and matrix ReLU."""
    graph = Graph(
        (Tensor("x", (2, 3)), Tensor("offset", (2, 2)),
         Tensor("w", (3, 2), (1, -1, 2, 3, -2, 4)),
         Tensor("product", (2, 2)), Tensor("sum", (2, 2)), Tensor("result", (2, 2))),
        (Node("multiply", "matmul", ("x", "w"), "product"),
         Node("combine", "add", ("product", "offset"), "sum"),
         Node("activate", "relu", ("sum",), "result")),
        ("x", "offset"), ("product", "result"),
    )
    inputs = {"x": (1, -2, 3, -1, 0, 2), "offset": (10, -20, 6, 1)}
    return graph, inputs


class SuiteTests(unittest.TestCase):
    def test_suite_is_versioned_and_reproducible(self):
        self.assertEqual(WORKLOAD_SUITE_VERSION, 1)
        self.assertEqual(WORKLOAD_NAMES, ("gemv", "gemm", "mlp", "residual"))
        for name in WORKLOAD_NAMES:
            with self.subTest(name=name):
                first = make_workload(name, seed=31)
                self.assertEqual(first, make_workload(name, seed=31))
                self.assertNotEqual(workload_identity(*first), workload_identity(*make_workload(name, seed=32)))
                self.assertEqual(workload_identity(*first), content_hash(workload_to_dict(*first)))

    def test_local_rng_does_not_change_global_state(self):
        before = random.getstate()
        make_workload("residual", seed=9)
        self.assertEqual(random.getstate(), before)

    def test_all_scales_have_expected_shapes_and_safe_arithmetic(self):
        for name in WORKLOAD_NAMES:
            for scale in range(1, 5):
                with self.subTest(name=name, scale=scale):
                    graph, inputs = make_workload(name, scale=scale, seed=2**32 - 1)
                    graph.validate()
                    expected_input = (64 * scale,) if name == "gemv" else (8 * scale, 32 * scale)
                    expected_output = ((48 * scale,) if name == "gemv" else
                                       (8 * scale, (24 if name == "mlp" else 32 if name == "residual" else 48) * scale))
                    self.assertEqual(graph.tensor_map["x"].shape, expected_input)
                    self.assertEqual(graph.tensor_map["y"].shape, expected_output)
                    for tensor in graph.tensors:
                        if tensor.data is not None:
                            self.assertTrue(all(type(value) is int and -2 <= value <= 2 for value in tensor.data))
                    self.assertTrue(all(type(value) is int and -2 <= value <= 2 for value in inputs["x"]))
                    result = scalar_oracle(graph, inputs)
                    self.assertEqual(len(result["y"]), prod(expected_output))
                    self.assertTrue(all(type(value) is int and -(2**63) <= value < 2**63 for value in result["y"]))

    def test_suite_executes_on_tiled_reference(self):
        platform = Platform((ReferenceDevice("test", "npu", 2**20, 7, 5, 3),),
                            (("host", "test"), ("test", "host")))
        for name in WORKLOAD_NAMES:
            with self.subTest(name=name):
                graph, inputs = make_workload(name, seed=7)
                result = execute(graph, inputs, platform, {node.name: "test" for node in graph.nodes})
                self.assertEqual(result["outputs"], {key: list(value) for key, value in scalar_oracle(graph, inputs).items()})

    def test_residual_has_shared_input_and_branch_merge(self):
        graph, _ = make_workload("residual")
        self.assertEqual(graph.nodes[0].inputs[0], "x")
        self.assertEqual(graph.nodes[1].inputs[0], "x")
        self.assertEqual(graph.nodes[2].op, "add")
        self.assertEqual(graph.nodes[2].inputs, (graph.nodes[0].output, graph.nodes[1].output))

    def test_invalid_generator_arguments_are_rejected(self):
        for name in ("", "unknown", True, 1, [], None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                make_workload(name)
        for scale in (0, -1, 5, True, 1.0, "1", None):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                make_workload("gemv", scale=scale)
        for seed in (-1, 2**32, True, 0.0, "0", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                make_workload("gemv", seed=seed)


class InterchangeTests(unittest.TestCase):
    def test_suite_roundtrips_through_real_json(self):
        for name in WORKLOAD_NAMES:
            with self.subTest(name=name):
                original = make_workload(name, seed=99)
                encoded = json.loads(json.dumps(workload_to_dict(*original), allow_nan=False))
                restored = workload_from_dict(encoded)
                self.assertEqual(restored, original)
                self.assertEqual(workload_identity(*restored), workload_identity(*original))

    def test_custom_multiple_inputs_outputs_roundtrip_and_known_arithmetic(self):
        graph, inputs = custom_graph()
        payload = workload_to_dict(graph, inputs)
        before = deepcopy(payload)
        restored = workload_from_dict(payload)
        self.assertEqual(payload, before)
        self.assertEqual(restored, (graph, inputs))
        self.assertEqual(scalar_oracle(*restored), {"product": (-9, 5, -5, 9), "result": (1, 0, 1, 10)})

    def test_identity_tracks_values_not_dictionary_order(self):
        graph, inputs = custom_graph()
        identity = workload_identity(graph, inputs)
        self.assertEqual(identity, workload_identity(graph, dict(reversed(list(inputs.items())))))
        changed = dict(inputs, offset=(11, -20, 6, 1))
        self.assertNotEqual(identity, workload_identity(graph, changed))
        tensors = list(graph.tensors)
        tensors[2] = replace(tensors[2], data=(2, -1, 2, 3, -2, 4))
        self.assertNotEqual(identity, workload_identity(replace(graph, tensors=tuple(tensors)), inputs))

    def test_object_fields_are_exact_at_every_level(self):
        base = workload_to_dict(*custom_graph())
        paths = ((), ("graph",), ("graph", "tensors", 0), ("graph", "nodes", 0), ("inputs",))
        for path in paths:
            for modification in ("extra", "missing"):
                with self.subTest(path=path, modification=modification):
                    payload = deepcopy(base)
                    target = payload
                    for item in path:
                        target = target[item]
                    if modification == "extra":
                        target["unknown"] = 1
                    else:
                        del target[next(iter(target))]
                    with self.assertRaises(ValueError):
                        workload_from_dict(payload)

    def test_json_containers_text_and_version_are_strict(self):
        base = workload_to_dict(*custom_graph())
        changes = (
            (("schema_version",), True), (("schema_version",), 1.0), (("schema_version",), 2),
            (("graph",), []), (("graph", "tensors"), {}), (("graph", "nodes"), ()),
            (("graph", "inputs"), "x"), (("graph", "outputs"), [True]),
            (("graph", "tensors", 0, "shape"), (2, 3)),
            (("graph", "tensors", 0, "name"), True),
            (("graph", "nodes", 0, "inputs"), "x"),
            (("graph", "nodes", 0, "inputs"), ["x", True]),
            (("graph", "nodes", 0, "name"), ""),
            (("graph", "nodes", 0, "op"), []),
            (("graph", "nodes", 0, "output"), 1),
            (("inputs", "x"), tuple(base["inputs"]["x"])),
        )
        for path, value in changes:
            with self.subTest(path=path, value=value):
                payload = deepcopy(base)
                target = payload
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    workload_from_dict(payload)
        for root in (None, [], True, "{}"):
            with self.subTest(root=root), self.assertRaises(ValueError):
                workload_from_dict(root)

    def test_dimensions_and_integer_data_are_strict(self):
        base = workload_to_dict(*custom_graph())
        for shape in ([], [1, 1, 6], [True, 3], [2.0, 3], [0, 3], [-2, 3], [1_000_001]):
            with self.subTest(shape=shape):
                payload = deepcopy(base)
                payload["graph"]["tensors"][0]["shape"] = shape
                with self.assertRaises(ValueError):
                    workload_from_dict(payload)
        for bad in (True, 1.0, "1", None, [], 2**63, -(2**63) - 1):
            for destination in ("input", "weight"):
                with self.subTest(bad=bad, destination=destination):
                    payload = deepcopy(base)
                    data = payload["inputs"]["x"] if destination == "input" else payload["graph"]["tensors"][2]["data"]
                    data[0] = bad
                    with self.assertRaises(ValueError):
                        workload_from_dict(payload)
        for destination in ("input", "weight"):
            payload = deepcopy(base)
            if destination == "input":
                payload["inputs"]["x"] = [1]
            else:
                payload["graph"]["tensors"][2]["data"] = [1]
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                workload_from_dict(payload)

    def test_graph_semantics_and_export_inputs_are_validated(self):
        base = workload_to_dict(*custom_graph())
        for bad_op in ("unknown", "matvec"):
            payload = deepcopy(base)
            payload["graph"]["nodes"][0]["op"] = bad_op
            with self.subTest(op=bad_op), self.assertRaises(ValueError):
                workload_from_dict(payload)
        payload = deepcopy(base)
        payload["graph"]["nodes"].reverse()
        with self.assertRaises(ValueError):
            workload_from_dict(payload)
        graph, inputs = custom_graph()
        invalid_inputs = (None, [], {}, {**inputs, "extra": (1,)},
                          {**inputs, "x": list(inputs["x"])}, {**inputs, "x": (True,) * 6})
        for data in invalid_inputs:
            with self.subTest(data=data), self.assertRaises(ValueError):
                workload_to_dict(graph, data)
        with self.assertRaises(ValueError):
            workload_to_dict(None, {})


if __name__ == "__main__":
    unittest.main()

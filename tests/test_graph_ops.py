"""Exact matrix and multi-input graph semantics, with honest streamed traffic."""

from collections import Counter
from dataclasses import replace
import random
import unittest
from unittest.mock import patch

from pim_foundry.core import Graph, Node, Platform, ReferenceDevice, Tensor
from pim_foundry.reference import MAX_KERNEL_EVENTS, evaluate_add, evaluate_matmul, evaluate_matvec
from pim_foundry.research import execute, plan, scalar_oracle


def platform(capacity_a=4096, capacity_b=4096):
    devices = (ReferenceDevice("a", "sram", capacity_a, 3, 2, 2),
               ReferenceDevice("b", "gpu", capacity_b, 3, 2, 4))
    names = ("host", "a", "b")
    return Platform(devices, tuple((a, b) for a in names for b in names if a != b))


def matrix_graph(rows=3, k_size=7, n_size=5):
    return Graph(
        (Tensor("x", (rows, k_size)),
         Tensor("w", (k_size, n_size), tuple(i % 13 - 6 for i in range(k_size * n_size))),
         Tensor("y", (rows, n_size))),
        (Node("linear", "matmul", ("x", "w"), "y"),), ("x",), ("y",),
    )


def untiled_product(x, weight, rows):
    k_size, n_size = weight.shape
    return tuple(sum(x[m * k_size + k] * weight.data[k * n_size + n] for k in range(k_size))
                 for m in range(rows) for n in range(n_size))


class MatrixKernelTests(unittest.TestCase):
    def test_random_rectangular_matrices_with_tail_tiles_on_every_kind(self):
        rng = random.Random(929)
        for kind in ("sram", "hbm", "gpu", "npu"):
            for tile_k, tile_n in ((1, 1), (3, 2), (16, 16)):
                target = ReferenceDevice("target", kind, 4096, tile_k, tile_n, 3)
                for rows, k_size, n_size in ((1, 1, 1), (3, 7, 5), (2, 4, 9)):
                    x = tuple(rng.randint(-11, 11) for _ in range(rows * k_size))
                    weight = Tensor("w", (k_size, n_size),
                                    tuple(rng.randint(-17, 17) for _ in range(k_size * n_size)))
                    with self.subTest(kind=kind, tiles=(tile_k, tile_n), shape=(rows, k_size, n_size)):
                        result = evaluate_matmul(x, weight, target)
                        self.assertEqual(result.values, untiled_product(x, weight, rows))
                        self.assertEqual(result, evaluate_matmul(x, weight, target))

    def test_each_row_reloads_and_uses_each_weight_once_in_tiled_execution(self):
        graph = matrix_graph()
        target = platform().devices[1]
        result = evaluate_matmul(tuple(range(-10, 11)), graph.tensors[1], target)
        coverage = Counter()
        order = []
        for load, compute in zip(result.events[::2], result.events[1::2]):
            self.assertEqual(load["kind"], "weight_load")
            self.assertEqual(compute["kind"], "gpu_tiled_matmul")
            for field in ("m", "k0", "k1", "n0", "n1", "unit"):
                self.assertEqual(load[field], compute[field])
            m, k0, k1, n0, n1 = (load[key] for key in ("m", "k0", "k1", "n0", "n1"))
            coverage.update((m, k, n) for k in range(k0, k1) for n in range(n0, n1))
            order.append((m, n0, k0))
            self.assertEqual(load["logical_bytes"], 8 * compute["macs"])
            self.assertEqual(compute["macs"], (k1 - k0) * (n1 - n0))
            self.assertEqual(load["unit"], (n0 // target.tile_n) % target.units)
        self.assertEqual(coverage, Counter((m, k, n) for m in range(3) for k in range(7) for n in range(5)))
        self.assertEqual(order, [(m, n, k) for m in range(3) for n in (0, 2, 4) for k in (0, 3, 6)])
        self.assertEqual(sum(event.get("logical_bytes", 0) for event in result.events), 3 * 7 * 5 * 8)

    def test_exact_reductions_can_exceed_int64_before_cancellation(self):
        maximum = 2**63 - 1
        weight = Tensor("w", (3, 1), (maximum, -maximum, 1))
        x = (maximum, maximum, -3, maximum, maximum, 4)
        for kind in ("sram", "hbm", "gpu", "npu"):
            target = ReferenceDevice("target", kind, 1024, 1, 1, 1)
            self.assertEqual(evaluate_matmul(x, weight, target).values, (-3, 4))
            for value in (maximum, -(2**63)):
                with self.assertRaisesRegex(ValueError, "signed int64"):
                    evaluate_matmul((1, value), Tensor("w", (1, 1), (2,)), target)

    def test_capacity_counts_complete_matrix_input_output_and_one_weight_tile(self):
        graph = matrix_graph(2, 3, 2)
        target = ReferenceDevice("target", "hbm", 96, 2, 1, 1)
        x = tuple(range(6))
        self.assertEqual(evaluate_matmul(x, graph.tensors[1], target).values,
                         untiled_product(x, graph.tensors[1], 2))
        with self.assertRaisesRegex(ValueError, "96 logical bytes"):
            evaluate_matmul(x, graph.tensors[1], replace(target, capacity_bytes=95))
        mapped = Platform((target,), (("host", "target"), ("target", "host")))
        result = plan(graph, mapped, {"linear": "target"})
        self.assertEqual(result["peak_live_bytes"], {"target": 96})
        self.assertEqual(result["nodes"][0]["input_staging_bytes"], 48)

    def test_flat_input_validation_and_bounded_work(self):
        target = platform().devices[1]
        weight = Tensor("w", (3, 2), (1,) * 6)
        for x in ((), [1, 2, 3], (1, 2), (1, True, 3), (1, 2.0, 3), (1, 2**63, 3)):
            with self.subTest(x=x), self.assertRaises(ValueError):
                evaluate_matmul(x, weight, target)
        with self.assertRaisesRegex(ValueError, "work limit"):
            evaluate_matmul((1,) * 3000, Tensor("w", (3, 1000), (1,) * 3000), target)


class GraphOperationValidationTests(unittest.TestCase):
    def test_matmul_rejects_wrong_ranks_dimensions_output_arity_and_dynamic_weights(self):
        base = matrix_graph()
        invalid = (
            replace(base, tensors=(Tensor("x", (21,)), base.tensors[1], base.tensors[2])),
            replace(base, tensors=(Tensor("x", (3, 8)), base.tensors[1], base.tensors[2])),
            replace(base, tensors=(*base.tensors[:2], Tensor("y", (15,)))),
            replace(base, tensors=(*base.tensors[:2], Tensor("y", (5, 3)))),
            replace(base, nodes=(Node("linear", "matmul", ("x",), "y"),)),
            replace(base, tensors=(base.tensors[0], Tensor("w", (7, 5)), base.tensors[2]), inputs=("x", "w")),
        )
        for graph in invalid:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                graph.validate()

    def test_add_requires_exact_shapes_without_implicit_broadcast(self):
        base = Graph((Tensor("x", (2, 3)), Tensor("z", (2, 3)), Tensor("y", (2, 3))),
                     (Node("sum", "add", ("x", "z"), "y"),), ("x", "z"), ("y",))
        base.validate()
        for shape in ((6,), (1, 3), (3, 2)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "identical shapes"):
                replace(base, tensors=(base.tensors[0], Tensor("z", shape), base.tensors[2])).validate()
        for inputs in (("x",), ("x", "z", "x")):
            with self.assertRaisesRegex(ValueError, "two inputs"):
                replace(base, nodes=(Node("sum", "add", inputs, "y"),)).validate()

    def test_rank_two_relu_is_valid_and_preserves_shape(self):
        graph = Graph((Tensor("x", (2, 3)), Tensor("y", (2, 3))),
                      (Node("relu", "relu", ("x",), "y"),), ("x",), ("y",))
        result = execute(graph, {"x": (-1, 2, -3, 4, -5, 6)}, platform(), {"relu": "b"})
        self.assertEqual(result["outputs"], {"y": [0, 2, 0, 4, 0, 6]})
        with self.assertRaisesRegex(ValueError, "matching output shape"):
            replace(graph, tensors=(graph.tensors[0], Tensor("y", (3, 2)))).validate()

    def test_graph_work_bound_counts_every_matrix_row(self):
        with self.assertRaisesRegex(ValueError, "work limit"):
            matrix_graph(201, 100, 100).validate()


class KernelEventBudgetTests(unittest.TestCase):
    def test_full_budget_accepts_exact_boundary_without_materializing_events(self):
        self.assertEqual(MAX_KERNEL_EVENTS, 100_000)
        graph = matrix_graph(2, 1, 25000)
        target = ReferenceDevice("target", "gpu", 1_000_000, 1, 1, 1)
        mapped = Platform((target,), (("host", "target"), ("target", "host")))
        result = plan(graph, mapped, {"linear": "target"})
        self.assertEqual(result["kernel_event_count"], MAX_KERNEL_EVENTS)
        self.assertEqual(result["nodes"][0]["kernel_event_count"], MAX_KERNEL_EVENTS)
        over = matrix_graph(2, 1, 25001)
        with self.assertRaisesRegex(ValueError, "100000-kernel-event limit"):
            plan(over, mapped, {"linear": "target"})

    def test_standalone_preflight_rejects_real_oversized_trace_before_capacity_allocation(self):
        weight = Tensor("w", (1, 50001), (1,) * 50001)
        target = ReferenceDevice("target", "hbm", 1_000_000, 1, 1, 1)
        with patch("pim_foundry.reference._check_capacity") as capacity:
            for evaluate in (evaluate_matvec, evaluate_matmul):
                with self.assertRaisesRegex(ValueError, "100000-kernel-event limit"):
                    evaluate((1,), weight, target)
            capacity.assert_not_called()

    def test_standalone_boundary_with_small_trace_preserves_inclusive_limit(self):
        target = ReferenceDevice("target", "sram", 1024, 1, 1, 1)
        with patch("pim_foundry.reference.MAX_KERNEL_EVENTS", 4):
            for evaluate in (evaluate_matvec, evaluate_matmul):
                self.assertEqual(len(evaluate((2,), Tensor("w", (1, 2), (3, 4)), target).events), 4)
                with self.assertRaisesRegex(ValueError, "4-kernel-event limit"):
                    evaluate((2,), Tensor("w", (1, 3), (3, 4, 5)), target)

    def test_total_graph_budget_is_checked_before_first_kernel_runs(self):
        graph = Graph(
            (Tensor("x", (1,)), Tensor("w1", (1, 30000), (1,) * 30000), Tensor("middle", (30000,)),
             Tensor("w2", (30000, 1), (1,) * 30000), Tensor("y", (1,))),
            (Node("first", "matvec", ("x", "w1"), "middle"),
             Node("second", "matvec", ("middle", "w2"), "y")), ("x",), ("y",),
        )
        target = ReferenceDevice("target", "npu", 1_000_000, 1, 1, 1)
        mapped = Platform((target,), (("host", "target"), ("target", "host")))
        with patch("pim_foundry.reference.evaluate_matvec") as evaluate:
            with self.assertRaisesRegex(ValueError, "100000-kernel-event limit"):
                execute(graph, {"x": (1,)}, mapped, {"first": "target", "second": "target"})
            evaluate.assert_not_called()

    def test_scalar_operations_count_once_and_tails_match_actual_events(self):
        graph = Graph(
            (Tensor("x", (2, 3)), Tensor("w", (3, 5), (1,) * 15), Tensor("middle", (2, 5)),
             Tensor("positive", (2, 5)), Tensor("y", (2, 5))),
            (Node("matrix", "matmul", ("x", "w"), "middle"),
             Node("relu", "relu", ("middle",), "positive"),
             Node("add", "add", ("positive", "positive"), "y")), ("x",), ("y",),
        )
        mapping = {node.name: "a" for node in graph.nodes}
        result = execute(graph, {"x": (-1, 2, 3, -4, 5, 6)}, platform(), mapping)
        self.assertEqual(result["plan"]["kernel_event_count"], len(result["kernel_events"]))
        self.assertEqual([node["kernel_event_count"] for node in result["plan"]["nodes"]], [12, 1, 1])


class AddAndTransferTests(unittest.TestCase):
    def test_duplicate_input_reuses_storage_and_transfers_once(self):
        graph = Graph((Tensor("x", (2,)), Tensor("y", (2,))),
                      (Node("double", "add", ("x", "x"), "y"),), ("x",), ("y",))
        result = execute(graph, {"x": (-3, 4)}, platform(capacity_a=32), {"double": "a"})
        self.assertEqual(result["outputs"], {"y": [-6, 8]})
        transfers = [item for item in result["plan"]["transfers"] if item["tensor"] == "x"]
        self.assertEqual(len(transfers), 1)
        self.assertEqual(result["plan"]["peak_live_bytes"]["a"], 32)
        self.assertEqual(result["plan"]["nodes"][0]["input_staging_bytes"], 16)
        with self.assertRaisesRegex(ValueError, "capacity"):
            plan(graph, platform(capacity_a=31), {"double": "a"})

    def test_two_distinct_inputs_are_both_transferred_and_retained_for_add(self):
        graph = Graph((Tensor("x", (2,)), Tensor("z", (2,)), Tensor("y", (2,))),
                      (Node("sum", "add", ("x", "z"), "y"),), ("x", "z"), ("y",))
        result = execute(graph, {"x": (-3, 4), "z": (6, -8)}, platform(capacity_b=48), {"sum": "b"})
        self.assertEqual(result["outputs"], {"y": [3, -4]})
        self.assertEqual([item["tensor"] for item in result["plan"]["transfers"]], ["x", "z", "y"])
        self.assertEqual(result["plan"]["nodes"][0]["input_staging_bytes"], 32)
        self.assertEqual(result["plan"]["peak_live_bytes"]["b"], 48)
        with self.assertRaisesRegex(ValueError, "capacity"):
            plan(graph, platform(capacity_b=47), {"sum": "b"})

    def test_add_fanout_keeps_source_for_later_consumers_and_graph_output(self):
        graph = Graph(
            tuple(Tensor(name, (2, 2)) for name in ("x", "z", "a0", "b0", "d", "s", "y")),
            (Node("left", "relu", ("x",), "a0"), Node("right", "relu", ("z",), "b0"),
             Node("double", "add", ("a0", "a0"), "d"),
             Node("merge", "add", ("a0", "b0"), "s"),
             Node("finish", "add", ("d", "s"), "y")),
            ("x", "z"), ("a0", "y"),
        )
        mapping = {"left": "a", "right": "b", "double": "b", "merge": "b", "finish": "a"}
        inputs = {"x": (-1, 2, 3, -4), "z": (5, -6, 7, 8)}
        result = execute(graph, inputs, platform(128, 128), mapping)
        self.assertEqual(result["outputs"], {"a0": [0, 2, 3, 0], "y": [5, 6, 16, 8]})
        self.assertEqual(result["plan"]["peak_live_bytes"], {"a": 128, "b": 128})
        copies = [item for item in result["plan"]["transfers"]
                  if item["tensor"] == "a0" and item["destination"] == "b"]
        self.assertEqual([item["consumer"] for item in copies], ["double", "merge"])
        self.assertEqual(result["outputs"], {name: list(value) for name, value in scalar_oracle(graph, inputs).items()})

    def test_matmul_weight_traffic_matches_actual_loads_including_relay_hops(self):
        graph = matrix_graph()
        routed = replace(platform(), links=(("host", "a"), ("a", "b"), ("b", "host")))
        result = execute(graph, {"x": tuple(range(-10, 11))}, routed, {"linear": "b"})
        weights = next(item for item in result["plan"]["transfers"] if item["streamed"])
        loads = [event for event in result["kernel_events"] if event["kind"] == "weight_load"]
        self.assertEqual(weights["payload_bytes"], 7 * 5 * 8)
        self.assertEqual(weights["repetitions"], 3)
        self.assertEqual(weights["max_chunk_bytes"], 3 * 2 * 8)
        self.assertEqual(weights["path"], ["host", "a", "b"])
        self.assertEqual(sum(event["logical_bytes"] for event in loads),
                         weights["payload_bytes"] * weights["repetitions"])
        self.assertEqual(weights["logical_link_bytes"], 3 * 7 * 5 * 8 * 2)
        self.assertEqual(result["plan"]["logical_bytes_by_link"], {
            "host->a": (3 * 7 + 3 * 7 * 5) * 8,
            "a->b": (3 * 7 + 3 * 7 * 5) * 8,
            "b->host": 3 * 5 * 8,
        })

    def test_same_constant_can_be_activation_and_weight_with_distinct_transfer_modes(self):
        graph = Graph((Tensor("w", (2, 2), (1, 2, 3, 4)), Tensor("y", (2, 2))),
                      (Node("square", "matmul", ("w", "w"), "y"),), (), ("y",))
        result = execute(graph, {}, platform(), {"square": "b"})
        self.assertEqual(result["outputs"], {"y": [7, 10, 15, 22]})
        transfers = [item for item in result["plan"]["transfers"] if item["tensor"] == "w"]
        self.assertEqual([(item["streamed"], item["repetitions"]) for item in transfers], [(False, 1), (True, 2)])

    def test_add_overflow_checked_before_following_relu(self):
        graph = Graph(tuple(Tensor(name, (1,)) for name in ("x", "z", "sum", "y")),
                      (Node("add", "add", ("x", "z"), "sum"), Node("relu", "relu", ("sum",), "y")),
                      ("x", "z"), ("y",))
        for inputs in ({"x": (2**63 - 1,), "z": (1,)}, {"x": (-(2**63),), "z": (-1,)}):
            with self.assertRaisesRegex(ValueError, "signed int64"):
                execute(graph, inputs, platform(), {"add": "a", "relu": "b"})
            with self.assertRaisesRegex(ValueError, "signed int64"):
                scalar_oracle(graph, inputs)

    def test_add_kernel_validation_and_alias_capacity(self):
        x = tuple(range(2))
        y = tuple(range(2))
        target = ReferenceDevice("target", "npu", 32, 1, 1, 1)
        self.assertEqual(evaluate_add(x, x, target).values, (0, 2))
        with self.assertRaisesRegex(ValueError, "capacity"):
            evaluate_add(x, y, target)
        with self.assertRaisesRegex(ValueError, "matching flat lengths"):
            evaluate_add(x, (1,), target)
        with self.assertRaisesRegex(ValueError, "signed int64"):
            evaluate_add(x, (False, 1), target)


if __name__ == "__main__":
    unittest.main()

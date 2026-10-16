"""Contracts for portable execution, liveness, routing and placement search."""

from collections import Counter
from dataclasses import replace
import json
import unittest

from pim_foundry.core import Graph, Node, Platform, ReferenceDevice, Tensor
from pim_foundry.research import demo, execute, plan, rank_placements, scalar_oracle


def platform(capacity_a=4096, capacity_b=4096, direct=True):
    links = (("host", "a"), ("a", "host"), ("host", "b"), ("b", "host"))
    if direct:
        links += (("a", "b"), ("b", "a"))
    return Platform(
        (ReferenceDevice("a", "sram", capacity_a, 2, 1, 2),
         ReferenceDevice("b", "hbm", capacity_b, 3, 2, 2)),
        links,
    )


def chain(length=2, width=2):
    return Graph(
        tuple(Tensor(f"v{i}", (width,)) for i in range(length + 1)),
        tuple(Node(f"n{i}", "relu", (f"v{i}",), f"v{i + 1}") for i in range(length)),
        ("v0",), (f"v{length}",),
    )


def matvec():
    return Graph(
        (Tensor("x", (3,)), Tensor("w", (3, 2), (1, -2, 3, 4, -5, 6)), Tensor("y", (2,))),
        (Node("linear", "matvec", ("x", "w"), "y"),),
        ("x",), ("y",),
    )


class GraphValidationTests(unittest.TestCase):
    def test_valid_graphs(self):
        chain().validate()
        matvec().validate()

    def test_matvec_dimensions_and_constant_weights_are_checked(self):
        base = matvec()
        replacements = (
            (Tensor("x", (4,)), base.tensors[1], base.tensors[2]),
            (base.tensors[0], base.tensors[1], Tensor("y", (3,))),
            (Tensor("x", (1, 3)), base.tensors[1], base.tensors[2]),
            (base.tensors[0], Tensor("w", (6,), tuple(range(6))), base.tensors[2]),
        )
        for tensors in replacements:
            with self.subTest(tensors=tensors), self.assertRaises(ValueError):
                replace(base, tensors=tensors).validate()
        dynamic = replace(base, tensors=(base.tensors[0], Tensor("w", (3, 2)), base.tensors[2]),
                          inputs=("x", "w"))
        with self.assertRaisesRegex(ValueError, "constant weights"):
            dynamic.validate()

    def test_relu_shape_and_arity_are_checked(self):
        base = chain(1)
        invalid = (
            replace(base, tensors=(Tensor("v0", (2,)), Tensor("v1", (3,)))),
            replace(base, tensors=(Tensor("v0", (1, 2)), Tensor("v1", (2, 1)))),
            replace(base, nodes=(Node("n0", "relu", (), "v1"),)),
            replace(base, nodes=(Node("n0", "relu", ("v0", "v0"), "v1"),)),
        )
        for graph in invalid:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                graph.validate()

    def test_use_before_definition_and_undeclared_tensor_are_rejected(self):
        base = chain()
        with self.assertRaisesRegex(ValueError, "defined before use"):
            replace(base, nodes=tuple(reversed(base.nodes))).validate()
        with self.assertRaisesRegex(ValueError, "defined before use"):
            replace(base, nodes=(Node("n0", "relu", ("missing",), "v1"), *base.nodes[1:])).validate()
        with self.assertRaisesRegex(ValueError, "single producer"):
            replace(base, nodes=(*base.nodes[:-1], Node("n1", "relu", ("v1",), "missing"))).validate()

    def test_output_single_producer_and_unique_names(self):
        base = chain()
        invalid = (
            replace(base, nodes=(*base.nodes, Node("n2", "relu", ("v0",), "v1"))),
            replace(base, nodes=(Node("n0", "relu", ("v0",), "v0"),)),
            replace(base, nodes=(base.nodes[0], replace(base.nodes[1], name="n0"))),
            replace(base, tensors=(*base.tensors, Tensor("v0", (2,)))),
            replace(base, inputs=("v0", "v0")),
            replace(base, outputs=("v2", "v2")),
        )
        for graph in invalid:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                graph.validate()

    def test_unknown_operations_and_unproduced_tensors_are_rejected(self):
        base = chain(1)
        with self.assertRaisesRegex(ValueError, "unsupported operation"):
            replace(base, nodes=(replace(base.nodes[0], op="softmax"),)).validate()
        with self.assertRaisesRegex(ValueError, "unproduced"):
            replace(base, tensors=(*base.tensors, Tensor("unused", (2,)))).validate()
        with self.assertRaisesRegex(ValueError, "outputs must be defined"):
            replace(base, outputs=("missing",)).validate()

    def test_tensor_dimension_initializer_and_int64_contract(self):
        invalid = (
            ("x", (True,), None), ("x", (0,), None), ("x", (2, 2, 2), None),
            ("x", [2], None), ("x", (2,), (1,)), ("x", (1,), (True,)),
            ("x", (1,), (1.0,)), ("x", (1,), (2**63,)),
            ("x", (1,), (-(2**63) - 1,)), ("x", (1_000_001,), None),
        )
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                Tensor(*args)


class PlacementMemoryTests(unittest.TestCase):
    def test_capacity_boundary_counts_input_output_and_one_weight_tile(self):
        result = plan(matvec(), platform(capacity_a=56), {"linear": "a"})
        self.assertEqual(result["peak_live_bytes"]["a"], 56)
        self.assertEqual(result["nodes"][0]["input_staging_bytes"], 24)
        self.assertEqual(result["nodes"][0]["weight_tile_bytes"], 16)
        with self.assertRaisesRegex(ValueError, "capacity"):
            plan(matvec(), platform(capacity_a=55), {"linear": "a"})

    def test_chain_releases_last_used_inputs_without_accumulating_all_outputs(self):
        graph = chain(length=20)
        result = plan(graph, platform(capacity_a=32), {node.name: "a" for node in graph.nodes})
        self.assertEqual(result["peak_live_bytes"]["a"], 32)
        self.assertEqual([item["working_peak_bytes"] for item in result["nodes"]], [32] * 20)
        self.assertEqual(len(result["transfers"]), 2)

    def test_fanout_keeps_original_until_last_consumer_and_keeps_graph_outputs(self):
        graph = Graph(
            tuple(Tensor(name, (2,)) for name in ("x", "a0", "b0", "a1", "b1")),
            (Node("start", "relu", ("x",), "a0"),
             Node("remote", "relu", ("a0",), "b0"),
             Node("local", "relu", ("a0",), "a1"),
             Node("finish", "relu", ("a1",), "b1")),
            ("x",), ("b0", "b1"),
        )
        mapping = {"start": "a", "remote": "b", "local": "a", "finish": "b"}
        result = plan(graph, platform(capacity_a=32, capacity_b=48), mapping)
        self.assertEqual(result["peak_live_bytes"], {"a": 32, "b": 48})
        self.assertEqual([n["working_peak_bytes"] for n in result["nodes"]], [32, 32, 32, 48])
        self.assertEqual(execute(graph, {"x": (-3, 4)}, platform(), mapping)["outputs"],
                         {"b0": [0, 4], "b1": [0, 4]})
        with self.assertRaisesRegex(ValueError, "capacity"):
            plan(graph, platform(capacity_b=47), mapping)

    def test_intermediate_device_staging_counts_existing_live_storage(self):
        graph = Graph(
            (Tensor("x", (2,)), Tensor("held", (2,)), Tensor("v", (4,)), Tensor("y", (4,))),
            (Node("keep", "relu", ("x",), "held"), Node("consume", "relu", ("v",), "y")),
            ("x", "v"), ("held", "y"),
        )
        base = platform(capacity_a=48)
        routed = replace(base, links=(("host", "a"), ("a", "host"), ("a", "b"), ("b", "a")))
        mapping = {"keep": "a", "consume": "b"}
        result = plan(graph, routed, mapping)
        self.assertEqual(result["peak_live_bytes"]["a"], 48)
        self.assertIn(["host", "a", "b"], [item["path"] for item in result["transfers"]])
        undersized = replace(routed, devices=(replace(routed.devices[0], capacity_bytes=47), routed.devices[1]))
        with self.assertRaisesRegex(ValueError, "capacity"):
            plan(graph, undersized, mapping)

    def test_weight_staging_through_intermediate_device_uses_tile_not_whole_matrix(self):
        graph = Graph(
            (Tensor("x", (3,)), Tensor("w", (3, 6), tuple(range(18))), Tensor("y", (6,))),
            (Node("linear", "matvec", ("x", "w"), "y"),), ("x",), ("y",),
        )
        base = platform(capacity_a=48, capacity_b=128)
        base = replace(base, devices=(base.devices[0], replace(base.devices[1], tile_k=2, tile_n=3)))
        routed = replace(base, links=(("host", "a"), ("a", "b"), ("b", "host")))
        result = plan(graph, routed, {"linear": "b"})
        self.assertEqual(result["peak_live_bytes"]["a"], 48)
        weights = next(item for item in result["transfers"] if item["tensor"] == "w")
        self.assertTrue(weights["streamed"])
        self.assertEqual(weights["max_chunk_bytes"], 48)
        self.assertEqual(weights["payload_bytes"], 144)
        undersized = replace(routed, devices=(replace(routed.devices[0], capacity_bytes=47), routed.devices[1]))
        with self.assertRaisesRegex(ValueError, "capacity"):
            plan(graph, undersized, {"linear": "b"})

    def test_incomplete_and_unknown_placements_are_rejected(self):
        for mapping in ({"n0": "a"}, {"n0": "a", "n1": "missing"},
                        {"n0": "a", "n1": "b", "unused": "a"}):
            with self.subTest(mapping=mapping), self.assertRaisesRegex(ValueError, "placement"):
                plan(chain(), platform(), mapping)


class RoutingAndSearchTests(unittest.TestCase):
    def test_missing_route_is_rejected_and_links_are_directed(self):
        base = platform()
        one_way = replace(base, links=(("host", "a"),))
        self.assertEqual(one_way.path("host", "a"), ("host", "a"))
        with self.assertRaisesRegex(ValueError, "no transfer route"):
            plan(chain(1), one_way, {"n0": "a"})

    def test_host_staging_adds_one_hop_of_logical_payload(self):
        graph, mapping = chain(), {"n0": "a", "n1": "b"}
        direct = plan(graph, platform(), mapping)
        staged = plan(graph, platform(direct=False), mapping)
        self.assertEqual(direct["logical_transfer_bytes"], 48)
        self.assertEqual(staged["logical_transfer_bytes"], 64)
        activation = next(t for t in staged["transfers"] if t["tensor"] == "v1")
        self.assertEqual(activation["path"], ["a", "host", "b"])
        self.assertEqual(activation["payload_bytes"], 16)
        self.assertEqual(activation["logical_link_bytes"], 32)

    def test_routes_are_shortest_hop_and_independent_of_link_declaration_order(self):
        base = platform()
        self.assertEqual(base.path("a", "b"), ("a", "b"))
        self.assertEqual(base.path("a", "a"), ("a",))
        self.assertEqual(base.path("host", "b"), replace(base, links=tuple(reversed(base.links))).path("host", "b"))

    def test_planning_and_search_are_deterministic(self):
        graph, target = chain(), platform()
        mapping = {"n0": "a", "n1": "b"}
        self.assertEqual(plan(graph, target, mapping), plan(graph, target, mapping))
        self.assertEqual(json.dumps(plan(graph, target, mapping)),
                         json.dumps(plan(graph, target, {"n1": "b", "n0": "a"})))
        self.assertEqual(rank_placements(graph, target), rank_placements(graph, target))

    def test_search_enumerates_each_assignment_and_reports_capacity_rejections(self):
        result = rank_placements(chain(), platform(capacity_a=31))
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(len(result["rejected"]), 3)
        self.assertEqual(result["candidates"][0]["placement"], {"n0": "b", "n1": "b"})
        self.assertTrue(all("capacity" in item["reason"] for item in result["rejected"]))
        self.assertEqual(result["objective"], "logical_transfer_bytes_only")
        self.assertFalse(result["performance_optimum_claimed"])

    def test_search_orders_by_payload_then_lexical_assignment(self):
        result = rank_placements(chain(), platform())
        self.assertEqual([tuple(c["placement"].values()) for c in result["candidates"]],
                         [("a", "a"), ("b", "b"), ("a", "b"), ("b", "a")])
        self.assertEqual([c["logical_transfer_bytes"] for c in result["candidates"]], [32, 32, 48, 48])

    def test_search_allows_4096_candidates_and_rejects_larger_problem(self):
        result = rank_placements(chain(12, width=1), platform())
        self.assertEqual(len(result["candidates"]) + len(result["rejected"]), 4096)
        with self.assertRaisesRegex(ValueError, "4096"):
            rank_placements(chain(13, width=1), platform())


class PortableExecutionTests(unittest.TestCase):
    def test_demo_placements_match_independent_scalar_calculation(self):
        experiment = demo()
        graph = experiment["graph"]
        values = {tensor["name"]: tuple(tensor["data"]) for tensor in graph["tensors"] if tensor["data"] is not None}
        values.update({name: tuple(data) for name, data in experiment["inputs"].items()})
        shapes = {tensor["name"]: tensor["shape"] for tensor in graph["tensors"]}
        for node in graph["nodes"]:
            x = values[node["inputs"][0]]
            if node["op"] == "relu":
                result = tuple(value if value > 0 else 0 for value in x)
            else:
                weight_name = node["inputs"][1]
                rows, columns = shapes[weight_name]
                result = tuple(sum(x[i] * values[weight_name][i * columns + j] for i in range(rows))
                               for j in range(columns))
            values[node["output"]] = result
        expected = {name: list(values[name]) for name in graph["outputs"]}
        self.assertEqual(set(experiment["runs"]), {"sram_only", "hbm_only", "heterogeneous"})
        for name, run in experiment["runs"].items():
            with self.subTest(placement=name):
                self.assertEqual(run["outputs"], expected)
                self.assertTrue(run["validation"]["matches_scalar_oracle"])
                self.assertFalse(run["validation"]["vendor_backend_numerics"])
                self.assertFalse(run["validation"]["hardware_execution"])

    def test_heterogeneous_plan_has_cross_device_transfer_and_per_link_accounting(self):
        run = demo()["runs"]["heterogeneous"]
        transfers = run["plan"]["transfers"]
        self.assertTrue(any(item["source"] == "hbm" and item["destination"] == "sram" for item in transfers))
        expected = Counter()
        for item in transfers:
            for source, destination in zip(item["path"], item["path"][1:]):
                expected[f"{source}->{destination}"] += item["payload_bytes"]
        self.assertEqual(run["plan"]["logical_bytes_by_link"], dict(expected))
        self.assertEqual(run["plan"]["logical_transfer_bytes"], sum(expected.values()))
        kinds = {event["kind"] for event in run["kernel_events"]}
        self.assertTrue({"bank_mac", "array_mvm", "vector_relu"}.issubset(kinds))

    def test_demo_can_be_serialized_and_is_reproducible(self):
        self.assertEqual(json.dumps(demo(), sort_keys=True), json.dumps(demo(), sort_keys=True))

    def test_input_names_shapes_booleans_and_out_of_range_values_are_rejected(self):
        graph, target, mapping = chain(1, width=1), platform(), {"n0": "a"}
        invalid = ({}, {"wrong": (1,)}, {"v0": (1,), "extra": (1,)},
                   {"v0": ()}, {"v0": [1]}, {"v0": (True,)}, {"v0": (1.0,)},
                   {"v0": (2**63,)}, {"v0": (-(2**63) - 1,)})
        for inputs in invalid:
            with self.subTest(inputs=inputs):
                with self.assertRaises(ValueError):
                    execute(graph, inputs, target, mapping)
                with self.assertRaises(ValueError):
                    scalar_oracle(graph, inputs)

    def test_node_output_overflow_is_rejected_before_later_relu(self):
        graph = Graph(
            (Tensor("x", (1,)), Tensor("w", (1, 1), (2,)), Tensor("hidden", (1,)), Tensor("y", (1,))),
            (Node("linear", "matvec", ("x", "w"), "hidden"), Node("relu", "relu", ("hidden",), "y")),
            ("x",), ("y",),
        )
        for value in (2**62, -(2**62) - 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "signed int64"):
                    execute(graph, {"x": (value,)}, platform(), {"linear": "a", "relu": "b"})
                with self.assertRaisesRegex(ValueError, "signed int64"):
                    scalar_oracle(graph, {"x": (value,)})


if __name__ == "__main__":
    unittest.main()

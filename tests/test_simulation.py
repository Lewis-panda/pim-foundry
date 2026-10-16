"""Hand-derived scheduling checks; these assert model behavior, not hardware."""

from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

from pim_foundry.core import Graph, Node, Platform, ReferenceDevice, Tensor
from pim_foundry.simulation import example_model, simulate, validate_model


def device(name="d", capacity=4096, tile_k=1, tile_n=1):
    return ReferenceDevice(name, "sram", capacity, tile_k, tile_n, 1)


def direct_platform(capacity=4096):
    return Platform((device(capacity=capacity),), (("host", "d"), ("d", "host")))


def linear_graph(matmul=False):
    shape = (2, 2) if matmul else (2,)
    return Graph(
        (Tensor("x", shape), Tensor("w", (2, 2), (1, 2, 3, 4)), Tensor("y", shape)),
        (Node("linear", "matmul" if matmul else "matvec", ("x", "w"), "y"),),
        ("x",), ("y",),
    )


def cost_model(platform):
    result = example_model(platform)
    for costs in result["devices"].values():
        costs.update(macs_per_ns=1, elements_per_ns=1, local_bytes_per_ns=8,
                     launch_ns=3, mac_energy_pj=2, element_energy_pj=2,
                     local_energy_pj_per_byte=0.5, idle_power_w=0.001)
    for costs in result["links"]:
        costs.update(bytes_per_ns=8, latency_ns=1, energy_pj_per_byte=1)
    return result


class ModelValidationTests(unittest.TestCase):
    def setUp(self):
        self.platform = direct_platform()
        self.model = cost_model(self.platform)

    def test_example_is_explicitly_synthetic_and_complete(self):
        model = example_model(self.platform)
        validate_model(self.platform, model)
        self.assertEqual(model["provenance"]["kind"], "synthetic")
        self.assertIn("Uncalibrated", model["provenance"]["description"])
        self.assertEqual(json.loads(json.dumps(model)), model)

    def test_exact_schema_and_topology(self):
        mutations = [
            lambda m: m.update(extra=1),
            lambda m: m.pop("links"),
            lambda m: m.update(schema_version=True),
            lambda m: m["provenance"].update(extra="x"),
            lambda m: m["provenance"].update(kind="guess"),
            lambda m: m["provenance"].update(description=" "),
            lambda m: m["devices"].clear(),
            lambda m: m["devices"].update(unknown=m["devices"]["d"]),
            lambda m: m["devices"]["d"].pop("launch_ns"),
            lambda m: m["devices"]["d"].update(extra=0),
            lambda m: m["links"].pop(),
            lambda m: m["links"].append(deepcopy(m["links"][0])),
            lambda m: m["links"][0].update(source="missing"),
            lambda m: m["links"][0].update(resource=""),
            lambda m: m["links"][0].update(resource=True),
            lambda m: m["links"][0].update(extra=1),
        ]
        for mutate in mutations:
            model = deepcopy(self.model)
            mutate(model)
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    validate_model(self.platform, model)

    def test_all_numeric_fields_reject_bool_nan_infinity_and_negative(self):
        for section, fields in (
            ("device", tuple(self.model["devices"]["d"])),
            ("link", ("bytes_per_ns", "latency_ns", "energy_pj_per_byte")),
        ):
            for field in fields:
                for value in (True, False, float("nan"), float("inf"), -1, "1", None):
                    model = deepcopy(self.model)
                    target = model["devices"]["d"] if section == "device" else model["links"][0]
                    target[field] = value
                    with self.subTest(section=section, field=field, value=value):
                        with self.assertRaises(ValueError):
                            validate_model(self.platform, model)

    def test_rates_must_be_positive_but_costs_can_be_zero(self):
        for field in ("macs_per_ns", "elements_per_ns", "local_bytes_per_ns"):
            model = deepcopy(self.model)
            model["devices"]["d"][field] = 0
            with self.assertRaises(ValueError):
                validate_model(self.platform, model)
        model = deepcopy(self.model)
        model["links"][0]["bytes_per_ns"] = 0
        with self.assertRaises(ValueError):
            validate_model(self.platform, model)
        for costs in self.model["devices"].values():
            for field in costs:
                if not field.endswith("_per_ns"):
                    costs[field] = 0
        for costs in self.model["links"]:
            costs.update(latency_ns=0, energy_pj_per_byte=0)
        validate_model(self.platform, self.model)


class AnalyticalTimingTests(unittest.TestCase):
    def setUp(self):
        self.graph = linear_graph()
        self.platform = direct_platform()
        self.model = cost_model(self.platform)
        self.placement = {"linear": "d"}

    def run_sim(self, **kwargs):
        return simulate(self.graph, self.platform, self.placement, self.model, **kwargs)

    def test_tiled_matvec_matches_hand_calculated_timing_and_energy(self):
        result = self.run_sim()
        # Input/output: each 1+16/8 = 3 ns. Four weight tiles: 2 ns each.
        # Each compute tile reads/writes 32 local bytes: max(1,32/8)=4 ns.
        # Launch occurs once: 3+8+(4*4+3)+3 = 33 ns.
        self.assertEqual(result["completion_ns"], [33])
        self.assertEqual(result["latency_ns"], [33])
        self.assertEqual(result["makespan_ns"], 33)
        self.assertEqual(result["logical_transfer_bytes"], 64)
        compute = [task for task in result["timeline"] if task["kind"] == "compute"]
        weights = [task for task in result["timeline"] if task.get("tensor") == "w"]
        self.assertEqual([task["bytes"] for task in weights], [8, 8, 8, 8])
        self.assertEqual([task["duration_ns"] for task in compute], [7, 4, 4, 4])
        self.assertEqual(sum(task["local_bytes"] for task in compute), 128)
        self.assertEqual(sum(task["launch_charged"] for task in compute), 1)
        self.assertEqual(result["resource_busy_ns"]["compute:d"], 19)
        self.assertEqual(result["resource_busy_ns"]["dma:host"], 14)
        # 4 MACs*2 + 128 local B*0.5 + 64 link B*1 = 136 pJ dynamic.
        # 0.001 W * 33 ns * 1000 pJ/(W ns) = 33 pJ baseline.
        self.assertEqual(result["energy"]["dynamic_pj"], 136)
        self.assertAlmostEqual(result["energy"]["idle_pj"], 33)
        self.assertAlmostEqual(result["energy"]["total_pj"], 169)
        self.assertEqual(result["evidence"], "analytical")
        self.assertIs(result["calibrated"], False)
        self.assertEqual(result["peak_reserved_bytes"], {"d": 40})

    def test_cold_trace_alternates_actual_weight_tiles_and_compute(self):
        result = self.run_sim()
        middle = result["timeline"][1:-1]
        self.assertEqual([task["kind"] for task in middle], ["transfer", "compute"] * 4)
        for load, compute in zip(middle[::2], middle[1::2]):
            self.assertEqual(load["tile"], compute["tile"])
            self.assertEqual(compute["dependencies"], [load["id"]])
            self.assertEqual(compute["start_ns"], load["end_ns"])

    def test_matmul_reloads_weight_tiles_per_row(self):
        self.graph = linear_graph(matmul=True)
        result = self.run_sim()
        weights = [task for task in result["timeline"] if task.get("tensor") == "w"]
        self.assertEqual(len(weights), 8)
        self.assertEqual([task["tile"]["m"] for task in weights], [0] * 4 + [1] * 4)
        self.assertEqual(sum(task["bytes"] for task in weights), 64)
        self.assertEqual(result["logical_transfer_bytes"], 128)
        # Input/output 5 ns each, weights 8*2 ns, compute 8*4+3 ns.
        self.assertEqual(result["makespan_ns"], 61)
        self.assertEqual(sum(task.get("macs", 0) for task in result["timeline"]), 8)
        self.assertEqual(sum(task.get("launch_charged", False) for task in result["timeline"]), 1)

    def test_partial_tiles_transfer_only_declared_elements(self):
        self.graph = Graph((Tensor("x", (3,)), Tensor("w", (3, 3), tuple(range(9))), Tensor("y", (3,))),
                           (Node("linear", "matvec", ("x", "w"), "y"),), ("x",), ("y",))
        self.platform = Platform((device(tile_k=2, tile_n=2),), self.platform.links)
        result = self.run_sim()
        loads = [task for task in result["timeline"] if task.get("tensor") == "w"]
        self.assertEqual([task["bytes"] for task in loads], [32, 16, 16, 8])
        self.assertEqual(sum(task.get("macs", 0) for task in result["timeline"]), 9)
        self.assertEqual(result["logical_transfer_bytes"], 120)

    def test_warm_preload_is_excluded_and_persistent_capacity_is_charged(self):
        result = self.run_sim(initial_state="warm")
        self.assertEqual(result["makespan_ns"], 25)
        self.assertEqual(result["logical_transfer_bytes"], 32)
        self.assertEqual(result["excluded_setup_logical_bytes"], 32)
        self.assertEqual(result["persistent_weight_bytes"], {"d": 32})
        self.assertEqual(result["peak_reserved_bytes"], {"d": 72})
        self.assertFalse(any(task.get("tensor") == "w" for task in result["timeline"]))
        self.assertEqual(result["energy"]["dynamic_pj"], 104)
        self.platform = direct_platform(capacity=71)
        with self.assertRaisesRegex(ValueError, "reserves 72"):
            self.run_sim(initial_state="warm")

    def test_warm_shared_weights_preload_once_for_same_device(self):
        self.graph = Graph(
            (Tensor("x", (2,)), Tensor("w", (2, 2), (1, 0, 0, 1)),
             Tensor("h", (2,)), Tensor("y", (2,))),
            (Node("first", "matvec", ("x", "w"), "h"),
             Node("second", "matvec", ("h", "w"), "y")), ("x",), ("y",),
        )
        self.placement = {"first": "d", "second": "d"}
        result = self.run_sim(initial_state="warm", arrivals_ns=(0, 0), max_inflight=2)
        self.assertEqual(result["persistent_weight_bytes"], {"d": 32})
        self.assertEqual(result["excluded_setup_logical_bytes"], 32)
        self.assertEqual(result["logical_transfer_bytes"], 64)

    def test_relu_and_add_local_traffic_and_second_input_transfer(self):
        for op, inputs, expected_ns, local_bytes in (
            ("relu", ("x",), 21, 64),
            ("add", ("x", "other"), 30, 96),
            ("add", ("x", "x"), 25, 96),
        ):
            tensors = [Tensor("x", (2, 2)), Tensor("y", (2, 2))]
            external = ("x",)
            if "other" in inputs:
                tensors.append(Tensor("other", (2, 2)))
                external += ("other",)
            self.graph = Graph(tuple(tensors), (Node("element", op, inputs, "y"),), external, ("y",))
            self.placement = {"element": "d"}
            with self.subTest(op=op, inputs=inputs):
                result = self.run_sim()
                self.assertEqual(result["makespan_ns"], expected_ns)
                compute = next(task for task in result["timeline"] if task["kind"] == "compute")
                self.assertEqual(compute["local_bytes"], local_bytes)
                self.assertEqual(compute["elements"], 4)

    def test_fractional_costs_are_rounded_without_decimal_ratio_error(self):
        self.graph = Graph((Tensor("x", (3,)), Tensor("y", (3,))),
                           (Node("relu", "relu", ("x",), "y"),), ("x",), ("y",))
        self.placement = {"relu": "d"}
        self.model["devices"]["d"].update(elements_per_ns=0.3, local_bytes_per_ns=1000, launch_ns=0.1)
        for costs in self.model["links"]:
            costs.update(bytes_per_ns=1000, latency_ns=0.1)
        result = self.run_sim()
        self.assertEqual([task["duration_ns"] for task in result["timeline"]], [2, 11, 2])

    def test_inflight_reserves_full_plan_peak_per_slot(self):
        self.platform = direct_platform(capacity=79)
        with self.assertRaisesRegex(ValueError, "reserves 80"):
            self.run_sim(arrivals_ns=(0, 0), max_inflight=2)
        self.platform = direct_platform(capacity=80)
        result = self.run_sim(arrivals_ns=(0, 0), max_inflight=2)
        self.assertEqual(result["peak_reserved_bytes"], {"d": 80})

    def test_arrival_gaps_and_all_device_baseline_power_are_counted(self):
        self.platform = Platform((device(), device("unused")), self.platform.links)
        self.model = cost_model(self.platform)
        result = self.run_sim(arrivals_ns=(100, 200), max_inflight=1)
        self.assertEqual(result["completion_ns"], [133, 233])
        self.assertEqual(result["latency_ns"], [33, 33])
        self.assertEqual(result["makespan_ns"], 133)
        self.assertEqual(result["resource_busy_ns"]["compute:unused"], 0)
        self.assertAlmostEqual(result["energy"]["idle_pj_by_device"]["unused"], 133)
        self.assertAlmostEqual(result["throughput_requests_per_second"], 2e9 / 133)

    def test_determinism_and_json_serialization(self):
        kwargs = {"arrivals_ns": (0, 0, 5), "max_inflight": 2}
        first = self.run_sim(**kwargs)
        second = self.run_sim(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first, allow_nan=False), json.dumps(second, allow_nan=False))
        self.assertEqual([task["start_ns"] for task in first["timeline"]],
                         sorted(task["start_ns"] for task in first["timeline"]))
        self.model["devices"]["d"]["macs_per_ns"] = 999
        self.assertEqual(first["model"]["devices"]["d"]["macs_per_ns"], 1)

    def test_graph_output_node_name_does_not_duplicate_readback(self):
        self.graph = Graph((Tensor("x", (2,)), Tensor("y", (2,))),
                           (Node("graph_output", "relu", ("x",), "y"),), ("x",), ("y",))
        self.placement = {"graph_output": "d"}
        result = self.run_sim()
        self.assertEqual([task["kind"] for task in result["timeline"]], ["transfer", "compute", "transfer"])
        self.assertEqual(result["logical_transfer_bytes"], 32)

    def test_invalid_arrivals_inflight_and_state(self):
        for arrivals in ((), (1, 0), (-1,), (True,), (1.0,), (math.inf,), "0", (0,) * 1025):
            with self.subTest(arrivals=repr(arrivals)[:80]):
                with self.assertRaises(ValueError):
                    self.run_sim(arrivals_ns=arrivals)
        for inflight in (0, -1, True, 1.0, 2):
            with self.subTest(inflight=inflight):
                with self.assertRaises(ValueError):
                    self.run_sim(max_inflight=inflight)
        with self.assertRaisesRegex(ValueError, "initial_state"):
            self.run_sim(initial_state="cached")

    def test_bounded_event_and_derived_time_failures(self):
        with patch("pim_foundry.simulation.MAX_EVENTS", 5):
            with self.assertRaisesRegex(ValueError, "event limit"):
                self.run_sim()
        self.model["devices"]["d"]["macs_per_ns"] = 1e-300
        with self.assertRaisesRegex(ValueError, "nanosecond limit"):
            self.run_sim()


class ContentionAndRoutingTests(unittest.TestCase):
    def chain(self, names):
        tensors = (Tensor("x", (1,)), *(Tensor(f"y{i}", (1,)) for i in range(len(names))))
        nodes = tuple(Node(f"node{i}", "relu", ("x" if i == 0 else f"y{i-1}",), f"y{i}")
                      for i in range(len(names)))
        graph = Graph(tensors, nodes, ("x",), (f"y{len(names)-1}",))
        endpoints = ("host", *names, "host")
        platform = Platform(tuple(device(name) for name in names), tuple(zip(endpoints, endpoints[1:])))
        model = cost_model(platform)
        for costs in model["devices"].values():
            costs.update(elements_per_ns=0.1, local_bytes_per_ns=1000, launch_ns=0)
        for costs in model["links"]:
            costs.update(bytes_per_ns=8, latency_ns=0)
        return graph, platform, {node.name: name for node, name in zip(nodes, names)}, model

    def assert_resources_and_dependencies(self, result):
        tasks = {task["id"]: task for task in result["timeline"]}
        resources = {}
        for task in result["timeline"]:
            for dependency in task["dependencies"]:
                self.assertLessEqual(tasks[dependency]["end_ns"], task["start_ns"])
            for resource in task["resources"]:
                resources.setdefault(resource, []).append(task)
        for resource, events in resources.items():
            events.sort(key=lambda event: (event["start_ns"], event["end_ns"]))
            for first, second in zip(events, events[1:]):
                self.assertLessEqual(first["end_ns"], second["start_ns"], resource)

    def test_independent_requests_pipeline_and_fixed_slot_dependency(self):
        args = self.chain(("a", "b"))
        serial = simulate(*args, arrivals_ns=(0, 0), max_inflight=1)
        parallel = simulate(*args, arrivals_ns=(0, 0), max_inflight=2)
        self.assertEqual(serial["completion_ns"], [23, 46])
        self.assertEqual(parallel["completion_ns"], [23, 33])
        self.assert_resources_and_dependencies(parallel)
        first_request_one = next(task for task in serial["timeline"] if task["request"] == 1)
        last_request_zero = [task for task in serial["timeline"] if task["request"] == 0][-1]
        self.assertEqual(first_request_one["dependencies"], [last_request_zero["id"]])
        self.assertEqual(first_request_one["start_ns"], 23)

    def test_shared_bus_serializes_disjoint_endpoint_transfers(self):
        graph, platform, placement, model = self.chain(("a", "b", "c", "d"))
        for costs in model["devices"].values():
            costs["elements_per_ns"] = 1
        for costs in model["links"]:
            costs["bytes_per_ns"] = 1
        separate = simulate(graph, platform, placement, model, arrivals_ns=(0, 0, 0), max_inflight=3)
        shared = deepcopy(model)
        for costs in shared["links"]:
            costs["resource"] = "shared-bus"
        serialized = simulate(graph, platform, placement, shared, arrivals_ns=(0, 0, 0), max_inflight=3)
        self.assertGreater(serialized["makespan_ns"], separate["makespan_ns"])
        transfers = [task for task in serialized["timeline"] if task["kind"] == "transfer"]
        self.assertEqual(serialized["resource_busy_ns"]["link:shared-bus"], 120)
        for first, second in zip(transfers, transfers[1:]):
            self.assertLessEqual(first["end_ns"], second["start_ns"])
        self.assert_resources_and_dependencies(separate)
        self.assert_resources_and_dependencies(serialized)

    def test_multihop_routes_use_each_link_and_relay_dma(self):
        graph = Graph((Tensor("x", (2,)), Tensor("y", (2,))),
                      (Node("relu", "relu", ("x",), "y"),), ("x",), ("y",))
        platform = Platform((device("relay"), device("d")),
                            (("host", "relay"), ("relay", "d"),
                             ("d", "relay"), ("relay", "host")))
        model = cost_model(platform)
        for costs in model["devices"].values():
            costs.update(elements_per_ns=1000, local_bytes_per_ns=1000, launch_ns=0)
        for costs in model["links"]:
            costs["latency_ns"] = 0
        result = simulate(graph, platform, {"relu": "d"}, model)
        self.assertEqual(result["makespan_ns"], 9)
        self.assertEqual(result["logical_transfer_bytes"], 64)
        self.assertEqual(result["resource_busy_ns"]["dma:relay"], 8)
        self.assertEqual(result["resource_busy_ns"]["compute:relay"], 0)
        transfers = [task for task in result["timeline"] if task["kind"] == "transfer"]
        self.assertEqual([(task["source"], task["destination"]) for task in transfers],
                         [("host", "relay"), ("relay", "d"), ("d", "relay"), ("relay", "host")])
        self.assert_resources_and_dependencies(result)


if __name__ == "__main__":
    unittest.main()

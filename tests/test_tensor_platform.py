import copy
import os
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from pim_foundry import tensor_platform as platform


def graph(dtype="float32", output_dtype=None):
    return {"schema": "pim-foundry.tensor-graph.v2", "values": {
        "a": {"shape": [2, 2, 3], "dtype": dtype}, "b": {"shape": [3, 2], "dtype": dtype},
        "c": {"shape": [2, 2, 2], "dtype": output_dtype or dtype}},
        "inputs": ["a"], "outputs": ["c"], "initializers": {"b": [[1, 0], [0, 1], [1, 1]]},
        "nodes": [{"name": "linear", "op": "matmul", "inputs": ["a", "b"], "output": "c"}]}


@unittest.skipIf(np is None, "optional NumPy required")
class TensorPlatformTests(unittest.TestCase):
    def run_graph(self):
        return platform.run(graph(), {"a": np.arange(12, dtype=np.float32).reshape(2, 2, 3)}, platform.default_platform())

    def test_batched_actual_output_and_transfer_accounting(self):
        result = self.run_graph()
        np.testing.assert_array_equal(result["outputs"]["c"], np.arange(12, dtype=np.float32).reshape(2,2,3) @ np.array([[1,0],[0,1],[1,1]], dtype=np.float32))
        model = result["platform_execution"]
        self.assertFalse(model["hardware_execution"])
        self.assertEqual(sum(e["bytes"] for e in model["events"] if e["kind"] == "transfer"), 48 + 24 + 32)
        self.assertGreater(model["modeled_latency_ns"], model["nodes"][0]["duration_ns"])

    def test_warm_weights_shared_and_input_transferred_each_request(self):
        profile = self.run_graph()["platform_execution"]["profile"]
        cold = platform.schedule([profile, profile], platform.default_platform(), state="cold")
        warm = platform.schedule([profile, profile], platform.default_platform(), state="warm")
        def transferred(s): return sum(e["bytes"] for e in s["events"] if e["kind"] == "transfer")
        self.assertEqual(transferred(cold) - transferred(warm), 24)
        self.assertLess(warm["modeled_latency_ns"], cold["modeled_latency_ns"])

    def test_concurrency_keeps_all_resources_exclusive(self):
        profile = self.run_graph()["platform_execution"]["profile"]
        result = platform.schedule([profile] * 4, platform.default_platform(), max_inflight=3)
        resources = {}
        for event in result["events"]:
            for resource in event["resources"]:
                resources.setdefault(resource, []).append((event["start_ns"], event["end_ns"]))
        for ranges in resources.values():
            ranges.sort()
            self.assertTrue(all(a[1] <= b[0] for a,b in zip(ranges,ranges[1:])))
        self.assertGreaterEqual(result["requests"][3]["admitted_ns"], result["requests"][0]["finish_ns"])

    def test_capacity_uses_live_inputs_and_output(self):
        profile = self.run_graph()["platform_execution"]["profile"]
        config = platform.default_platform()
        config["devices"][0]["capacity_bytes"] = 90
        with self.assertRaisesRegex(ValueError, "capacity"):
            platform.schedule([profile], config)

    def test_missing_route_rejected(self):
        config = platform.default_platform()
        config["links"] = []
        with self.assertRaisesRegex(ValueError, "route"):
            platform.run(graph(), {"a": np.ones((2,2,3), dtype=np.float32)}, config)

    def test_invalid_profile_timing_and_ssa_rejected(self):
        profile = self.run_graph()["platform_execution"]["profile"]
        for field, value in (("duration_ns", -1), ("energy_pj", float("nan")), ("workspace_bytes", -1), ("inputs", ["missing"])):
            broken = copy.deepcopy(profile); broken["nodes"][0][field] = value
            with self.assertRaises(ValueError): platform.schedule([broken], platform.default_platform())

    def test_placement_and_implicit_precision_rejected(self):
        with self.assertRaisesRegex(ValueError, "every node"):
            platform.run(graph(), {"a": np.ones((2,2,3), dtype=np.float32)}, platform.default_platform(), placement={})
        from pim_foundry.unified import HBMBackend
        backend = HBMBackend({"runtime_descriptor": "missing", "packing_bytes_per_ns": 16, "packing_energy_pj_per_byte": 0})
        with self.assertRaisesRegex(ValueError, "float16"):
            platform._native(graph()["nodes"][0], [np.ones((2,3),dtype=np.float32),np.ones((3,2),dtype=np.float32)],
                             {"dtype": "float32"}, backend, Path("unused"), 1024)

    def test_executed_unused_node_keeps_request_alive(self):
        profile = self.run_graph()["platform_execution"]["profile"]
        profile["outputs"] = ["a"]
        profile["nodes"][0]["duration_ns"] = 1000
        result = platform.schedule([profile], platform.default_platform())
        self.assertGreaterEqual(result["modeled_latency_ns"], max(e["end_ns"] for e in result["events"]))

    def test_attention_cost_accounts_for_quadratic_scores(self):
        device = platform.default_platform()["devices"][0]
        node = {"op": "causal_attention", "attrs": {}}
        def cost(t):
            a = np.ones((1,2,t,4),dtype=np.float32)
            return platform._cost(node,[a,a,a],a,device)
        small, large = cost(4), cost(8)
        self.assertEqual(large["mac_count"],4*small["mac_count"])
        self.assertEqual(large["workspace_bytes"],4*small["workspace_bytes"])

    def test_timestamp_translation_does_not_change_latency_or_energy(self):
        profile = self.run_graph()["platform_execution"]["profile"]
        config = platform.default_platform(); config["devices"][0]["idle_power_w"] = 3
        early = platform.schedule([profile],config,arrivals=[0])
        late = platform.schedule([profile],config,arrivals=[1000])
        self.assertEqual(early["modeled_latency_ns"],late["modeled_latency_ns"])
        self.assertEqual(early["modeled_energy_pj"],late["modeled_energy_pj"])
        self.assertEqual(late["observation_start_ns"],1000)

    def test_sram_split_reduction_requires_explicit_graph(self):
        from pim_foundry.unified import SRAMBackend
        backend = SRAMBackend({"executable":"unused", "packing_bytes_per_ns":16, "packing_energy_pj_per_byte":0})
        with self.assertRaisesRegex(ValueError,"cross-tile reduction"):
            platform._native({"name":"linear","op":"matmul"},
                             [np.ones((1,129),dtype=np.int8),np.ones((129,1),dtype=np.int8)],
                             {"dtype":"int32"},backend,Path("unused"),1024**2)

    @unittest.skipUnless(os.environ.get("PIM_FOUNDRY_TEST_NATIVE_HBM") == "1", "native HBM opt-in")
    def test_native_hbm_batched_actual_tensors(self):
        from pim_foundry.unified_examples import platform as native_platform
        root = Path(__file__).resolve().parents[1]
        config = native_platform(("hbm",), root=root)
        with tempfile.TemporaryDirectory() as directory:
            result = platform.run(graph("float16"), {"a": np.arange(12, dtype=np.float16).reshape((2,2,3))},
                                  config, {"linear": "hbm"}, Path(directory) / "execution")
            np.testing.assert_array_equal(result["outputs"]["c"], [[[2,3],[8,9]],[[14,15],[20,21]]])
            self.assertEqual(result["platform_execution"]["nodes"][0]["numerical_check"], "passed")

    @unittest.skipUnless(os.environ.get("PIM_FOUNDRY_CIMFLOW_SIMULATOR"), "native SRAM opt-in")
    def test_native_sram_batched_actual_tensors(self):
        from pim_foundry.unified_examples import platform as native_platform
        root = Path(__file__).resolve().parents[1]
        config = native_platform(("sram",), root=root)
        config["devices"][0]["options"]["executable"] = os.environ["PIM_FOUNDRY_CIMFLOW_SIMULATOR"]
        with tempfile.TemporaryDirectory() as directory:
            result = platform.run(graph("int8", "int32"), {"a": np.arange(12, dtype=np.int8).reshape((2,2,3))},
                                  config, {"linear": "sram"}, Path(directory) / "execution")
            np.testing.assert_array_equal(result["outputs"]["c"], [[[2,3],[8,9]],[[14,15],[20,21]]])


if __name__ == "__main__": unittest.main()

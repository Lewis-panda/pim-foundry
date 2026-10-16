"""Logical platform combinations; these tests do not run GPU or NPU hardware."""

from itertools import combinations
import json
import random
import unittest

from pim_foundry.core import ReferenceDevice, Tensor
from pim_foundry.reference import evaluate_matvec, evaluate_relu
from pim_foundry.research import demo, platform_demo


KINDS = ("sram", "hbm", "gpu", "npu")
EVENTS = {
    "sram": "array_mvm", "hbm": "bank_mac",
    "gpu": "gpu_tiled_matvec", "npu": "npu_tiled_matvec",
}


def expected_demo_output():
    """Compute the fixed workload without the graph executor or tiled engine."""
    x = tuple(range(-4, 4))
    w1 = tuple((i * 7) % 11 - 5 for i in range(48))
    hidden = [sum(x[k] * w1[k * 6 + n] for k in range(8)) for n in range(6)]
    activated = [max(value, 0) for value in hidden]
    w2 = tuple((i * 3) % 7 - 3 for i in range(24))
    return {"y": [sum(activated[k] * w2[k * 4 + n] for k in range(6)) for n in range(4)]}


class PlatformDemoTests(unittest.TestCase):
    def test_every_subset_has_all_single_device_baselines_and_matches_scalar_output(self):
        expected = expected_demo_output()
        for count in range(1, 5):
            for kinds in combinations(KINDS, count):
                with self.subTest(kinds=kinds):
                    experiment = platform_demo(kinds)
                    expected_names = {f"{kind}_only" for kind in kinds}
                    if count > 1:
                        expected_names.add("heterogeneous")
                    self.assertEqual(set(experiment["runs"]), expected_names)
                    for name, run in experiment["runs"].items():
                        self.assertEqual(run["outputs"], expected)
                        self.assertTrue(run["validation"]["matches_scalar_oracle"])
                        self.assertFalse(run["validation"]["hardware_execution"])
                        self.assertFalse(run["validation"]["timing_simulation"])
                        self.assertFalse(run["validation"]["vendor_backend_numerics"])
                        if name.endswith("_only"):
                            kind = name.removesuffix("_only")
                            self.assertEqual(set(run["plan"]["placement"].values()), {kind})
                            compute = {event["kind"] for event in run["kernel_events"]
                                       if event["kind"] not in ("weight_load", "vector_relu")}
                            self.assertEqual(compute, {EVENTS[kind]})
                        else:
                            self.assertEqual(len(set(run["plan"]["placement"].values())), 2)

    def test_hbm_gpu_and_hbm_npu_transfer_hidden_activation_explicitly(self):
        for accelerator in ("gpu", "npu"):
            with self.subTest(accelerator=accelerator):
                run = platform_demo(("hbm", accelerator))["runs"]["heterogeneous"]
                self.assertEqual(run["plan"]["placement"], {
                    "linear1": "hbm", "activation": accelerator, "linear2": accelerator,
                })
                activation_copy = [item for item in run["plan"]["transfers"]
                                   if item["tensor"] == "hidden"]
                self.assertEqual(len(activation_copy), 1)
                transfer = activation_copy[0]
                self.assertEqual(transfer["source"], "hbm")
                self.assertEqual(transfer["destination"], accelerator)
                self.assertEqual(transfer["path"], ["hbm", accelerator])
                self.assertEqual(transfer["payload_bytes"], 48)
                self.assertEqual(transfer["logical_link_bytes"], 48)
                self.assertFalse(transfer["streamed"])
                self.assertEqual(run["plan"]["logical_bytes_by_link"][f"hbm->{accelerator}"], 48)
                self.assertEqual(
                    {event["kind"] for event in run["kernel_events"]},
                    {"weight_load", "bank_mac", EVENTS[accelerator], "vector_relu"},
                )

    def test_complete_platform_keeps_inactive_devices_and_declares_demo_links(self):
        experiment = platform_demo(KINDS)
        expected_profiles = {
            "sram": (256, 4, 3, 2), "hbm": (4096, 8, 4, 4),
            "gpu": (8192, 8, 8, 8), "npu": (2048, 4, 4, 4),
        }
        endpoints = {"host", *KINDS}
        for run in experiment["runs"].values():
            profiles = run["plan"]["platform"]["devices"]
            self.assertEqual({profile["kind"] for profile in profiles}, set(KINDS))
            for profile in profiles:
                self.assertEqual(tuple(profile[key] for key in ("capacity_bytes", "tile_k", "tile_n", "units")),
                                 expected_profiles[profile["kind"]])
            self.assertEqual(set(run["plan"]["platform"]["links"]),
                             {(a, b) for a in endpoints for b in endpoints if a != b})
            inactive = set(KINDS) - set(run["plan"]["placement"].values())
            for kind in inactive:
                self.assertEqual(run["plan"]["peak_live_bytes"][kind], 0)
        mixed = experiment["runs"]["heterogeneous"]
        self.assertEqual(mixed["plan"]["placement"], {
            "linear1": "hbm", "activation": "gpu", "linear2": "gpu",
        })
        self.assertEqual(len(experiment["placement_search"]["candidates"]), 64)
        self.assertEqual(experiment["placement_search"]["rejected"], [])

    def test_default_behavior_and_workload_identity_are_preserved_across_options(self):
        original = demo()
        self.assertEqual(original, platform_demo(("sram", "hbm")))
        self.assertEqual(set(original["runs"]), {"sram_only", "hbm_only", "heterogeneous"})
        for kinds in (("gpu",), ("npu",), ("hbm", "gpu"), ("hbm", "npu"), KINDS):
            experiment = platform_demo(kinds)
            for field in ("workload_sha256", "graph", "inputs"):
                self.assertEqual(experiment[field], original[field])
            self.assertEqual(json.dumps(experiment, sort_keys=True),
                             json.dumps(platform_demo(kinds), sort_keys=True))
            self.assertEqual(experiment, platform_demo(tuple(reversed(kinds))))

    def test_invalid_kind_options_are_rejected_before_execution(self):
        invalid = (None, True, "hbm", [], ["hbm", "gpu"], (), (True,), (["hbm"],),
                   ("hbm", "hbm"), ("unknown",), ("GPU",), ("hbm", None),
                   ("sram", "hbm", "gpu", "npu", "extra"))
        for kinds in invalid:
            with self.subTest(kinds=kinds), self.assertRaisesRegex(ValueError, "kinds"):
                platform_demo(kinds)


class AcceleratorReferenceTests(unittest.TestCase):
    def test_gpu_and_npu_tiled_outputs_match_untiled_scalar_calculation(self):
        rng = random.Random(191)
        for kind in ("gpu", "npu"):
            for tile_k, tile_n, units in ((1, 1, 1), (3, 2, 3), (8, 8, 8)):
                target = ReferenceDevice("target", kind, 4096, tile_k, tile_n, units)
                for k_size, n_size in ((1, 1), (7, 5), (9, 11)):
                    with self.subTest(kind=kind, tile=(tile_k, tile_n), shape=(k_size, n_size)):
                        x = tuple(rng.randint(-9, 9) for _ in range(k_size))
                        data = tuple(rng.randint(-9, 9) for _ in range(k_size * n_size))
                        weights = Tensor("w", (k_size, n_size), data)
                        expected = tuple(sum(x[k] * data[k * n_size + n] for k in range(k_size))
                                         for n in range(n_size))
                        result = evaluate_matvec(x, weights, target)
                        self.assertEqual(result.values, expected)
                        compute = [event for event in result.events if event["kind"] != "weight_load"]
                        self.assertEqual({event["kind"] for event in compute}, {EVENTS[kind]})
                        self.assertEqual(sum(event["macs"] for event in compute), k_size * n_size)
                        self.assertTrue(all(0 <= event["unit"] < units for event in compute))
                        self.assertEqual(evaluate_relu(expected, target).values,
                                         tuple(max(0, value) for value in expected))

    def test_gpu_and_npu_preserve_exact_integer_and_overflow_contract(self):
        maximum = 2**63 - 1
        for kind in ("gpu", "npu"):
            target = ReferenceDevice("target", kind, 1024, 1, 1, 1)
            weights = Tensor("w", (3, 1), (maximum, -maximum, 1))
            self.assertEqual(evaluate_matvec((maximum, maximum, -3), weights, target).values, (-3,))
            with self.assertRaisesRegex(ValueError, "signed int64"):
                evaluate_matvec((maximum,), Tensor("w", (1, 1), (2,)), target)
            for value in (True, 1.0):
                with self.assertRaisesRegex(ValueError, "signed int64"):
                    evaluate_matvec((value,), Tensor("w", (1, 1), (1,)), target)

    def test_device_kind_validation_rejects_invalid_types_and_names(self):
        for kind in (True, [], ("gpu",), None, "GPU", "unknown"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "reference kind"):
                ReferenceDevice("target", kind, 4096, 4, 4, 4)


if __name__ == "__main__":
    unittest.main()

"""Typed heterogeneous execution, accounting, and evidence rejection gates.

The fixture backend below is an explicitly synthetic unit-test backend. It
computes integer outputs independently of the typed reference, emits fixed
modeled costs, and does not execute or claim a native hardware simulator.
"""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pim_foundry import typed, unified


def graph_fixture():
    return typed.Graph((typed.Tensor("x", (2,), "int8"),
                        typed.Tensor("w", (2, 2), "int8", (-2, 5, 3, -4)),
                        typed.Tensor("a", (2,), "int32"), typed.Tensor("y", (2,), "int32")),
                       (typed.Node("linear", "matvec", ("x", "w"), "a"),
                        typed.Node("relu", "relu", ("a",), "y")), ("x",), ("y",))


def config_fixture(names=("gpu", "npu"), backend="analytical", capacity=1000):
    options = {"macs_per_ns": 1, "elements_per_ns": 1, "local_bytes_per_ns": 1,
               "launch_ns": 0, "mac_energy_pj": 2, "element_energy_pj": 3, "local_energy_pj_per_byte": 4}
    return {"schema_version": 1,
            "devices": [{"name": name, "kind": "gpu" if name == "gpu" else "npu", "backend": backend,
                         "capacity_bytes": capacity, "idle_power_w": 0,
                         "options": copy.deepcopy(options) if backend == "analytical" else {}}
                        for name in names],
            "links": [{"source": source, "destination": target, "bytes_per_ns": 2,
                       "latency_ns": 1, "energy_pj_per_byte": 5}
                      for source in ("host", *names) for target in ("host", *names) if source != target]}


class UnitFixtureBackend:
    """Independent integer arithmetic with synthetic fixed costs, for tests only."""
    name = "test-independent-integer"
    calls = []

    def __init__(self, options):
        self.options = options

    def validate(self, node, tensors):
        if node.op not in ("matvec", "add", "relu"):
            raise ValueError("unit fixture only supports simple integer operations")
        if tensors[node.output].dtype != "int32":
            raise ValueError("unit fixture output must be int32")

    def run(self, node, tensors, values, directory, available_bytes):
        self.calls.append((node.name, {name: tuple(values[name]) for name in node.inputs}, available_bytes))
        args = [values[name] for name in node.inputs]
        if node.op == "matvec":
            k, n = tensors[node.inputs[1]].shape
            actual = [sum(args[0][index] * args[1][index * n + col] for index in range(k)) for col in range(n)]
        elif node.op == "add":
            actual = [a + b for a, b in zip(*args)]
        else:
            actual = [max(value, 0) for value in args[0]]
        if self.options.get("corrupt"):
            actual[0] += 1
        directory.mkdir(parents=True)
        result = {"values": actual, "duration_ns": self.options.get("duration", 7),
                  "energy_pj": self.options.get("energy", 11), "workspace_bytes": self.options.get("workspace", 0),
                  "evidence": "explicit_synthetic_unit_fixture", "instruction_lowering_verified": False,
                  "native_arithmetic": False, "profile_provenance": "unit_test_only"}
        if self.options.get("override_identity"):
            result.update(node="forged-node", device="forged-device", backend="forged-backend", matches_scalar_reference=False)
        (directory / "fixture.json").write_text(json.dumps({"fixture": True, "node": node.name}))
        return result


class UnifiedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry = mock.patch.dict(unified._BACKENDS, {UnitFixtureBackend.name: UnitFixtureBackend})
        self.registry.start()
        UnitFixtureBackend.calls = []

    def tearDown(self):
        self.registry.stop()
        self.temp.cleanup()

    def run_case(self, *, graph=None, config=None, placement=None, name="result"):
        return unified.run(graph or graph_fixture(), {"x": (-7, 2)}, config or config_fixture(),
                           placement or {"linear": "gpu", "relu": "npu"}, self.root / name)

    def test_analytical_dtype_bytes_costs_outputs_and_liveness(self):
        result = self.run_case()
        self.assertEqual(result["outputs"], {"y": [20, 0]})
        self.assertTrue(result["quality"]["passed"])
        self.assertEqual(result["logical_transfer_bytes"], 22)
        self.assertEqual(result["modeled_latency_ns"], 45)
        self.assertEqual(result["modeled_energy_pj"], 244)
        self.assertEqual(result["peak_reserved_bytes"], {"gpu": 14, "npu": 16})
        transfers = [event for event in result["events"] if event["kind"] == "transfer"]
        self.assertEqual([(event["tensor"], event["bytes"]) for event in transfers], [("x", 2), ("w", 4), ("a", 8), ("y", 8)])
        self.assertFalse(result["hardware_execution"])
        self.assertFalse(result["performance_comparison_eligible"])
        self.assertTrue(unified.verify(self.root / "result")["passed"])

    def test_directed_multihop_routes_charge_each_hop(self):
        config = config_fixture()
        config["links"] = [link for link in config["links"] if (link["source"], link["destination"]) != ("gpu", "npu")]
        result = self.run_case(config=config)
        transfers = [event for event in result["events"] if event.get("tensor") == "a"]
        self.assertEqual([(event["source"], event["destination"]) for event in transfers], [("gpu", "host"), ("host", "npu")])
        self.assertEqual(result["logical_transfer_bytes"], 30)
        self.assertEqual(result["modeled_latency_ns"], 50)
        self.assertEqual(transfers[0]["payload_sha256"], transfers[1]["payload_sha256"])

    def test_missing_reverse_link_retains_failure(self):
        config = config_fixture()
        config["links"] = [link for link in config["links"] if link["destination"] != "host"]
        with self.assertRaisesRegex(ValueError, "no transfer route"):
            self.run_case(config=config)
        failure = json.loads((self.root / "result/failure.json").read_text())
        self.assertEqual(failure["completed_nodes"], 2)
        self.assertFalse((self.root / "result/manifest.json").exists())

    def test_fanout_duplicate_inputs_and_last_use_memory(self):
        graph = typed.Graph((typed.Tensor("x", (2,), "int8"), typed.Tensor("w", (2, 2), "int8", (-2, 5, 3, -4)),
                             typed.Tensor("a", (2,), "int32"), typed.Tensor("b", (2,), "int32"),
                             typed.Tensor("c", (2,), "int32"), typed.Tensor("y", (2,), "int32")),
                            (typed.Node("linear", "matvec", ("x", "w"), "a"),
                             typed.Node("double", "add", ("a", "a"), "b"), typed.Node("relu", "relu", ("a",), "c"),
                             typed.Node("sum", "add", ("b", "c"), "y")), ("x",), ("y",))
        result = self.run_case(graph=graph, config=config_fixture(("gpu",), capacity=24),
                               placement={node.name: "gpu" for node in graph.nodes})
        self.assertEqual(result["outputs"], {"y": [60, -86]})
        self.assertEqual(result["logical_transfer_bytes"], 14)
        self.assertEqual(result["peak_reserved_bytes"]["gpu"], 24)
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.run_case(graph=graph, config=config_fixture(("gpu",), capacity=23),
                          placement={node.name: "gpu" for node in graph.nodes}, name="too-small")

    def test_capacity_includes_input_output_and_backend_workspace(self):
        config = config_fixture(backend=UnitFixtureBackend.name)
        config["devices"][0]["capacity_bytes"] = 16
        config["devices"][0]["options"]["workspace"] = 3
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.run_case(config=config)
        self.assertEqual(UnitFixtureBackend.calls[0][2], 2)
        self.assertTrue((self.root / "result/failure.json").exists())

    def test_independent_backend_values_consumed_by_next_backend(self):
        config = config_fixture(backend=UnitFixtureBackend.name)
        result = self.run_case(config=config)
        self.assertEqual(UnitFixtureBackend.calls[0][1], {"x": (-7, 2), "w": (-2, 5, 3, -4)})
        self.assertEqual(UnitFixtureBackend.calls[1][1], {"a": (20, -43)})
        self.assertEqual(result["outputs"], {"y": [20, 0]})
        self.assertEqual(result["nodes"][0]["values"], [20, -43])
        self.assertEqual(result["nodes"][0]["evidence"], "explicit_synthetic_unit_fixture")

    def test_wrong_backend_value_fails_before_downstream(self):
        config = config_fixture(backend=UnitFixtureBackend.name)
        config["devices"][0]["options"]["corrupt"] = True
        with self.assertRaisesRegex(ValueError, "differs from"):
            self.run_case(config=config)
        self.assertEqual(len(UnitFixtureBackend.calls), 1)
        failure = json.loads((self.root / "result/failure.json").read_text())
        self.assertEqual(failure["completed_nodes"], 0)
        self.assertFalse((self.root / "result/result.json").exists())

    def test_no_implicit_dtype_fallback_to_reference_for_native_backend(self):
        config = config_fixture()
        config["devices"][0].update(kind="hbm", backend="hbm-simd16", options={"runtime_descriptor": "/not-used",
                                     "packing_bytes_per_ns": 1, "packing_energy_pj_per_byte": 0})
        with self.assertRaisesRegex(ValueError, "FP16"):
            self.run_case(config=config)
        self.assertFalse((self.root / "result").exists())

    def test_backend_numeric_accounting_rejects_negative_nonfinite_values(self):
        for key, value in (("workspace", -1), ("workspace", 0.5), ("energy", -1), ("energy", float("nan")),
                           ("duration", -1), ("duration", float("inf"))):
            config = config_fixture(backend=UnitFixtureBackend.name)
            config["devices"][0]["options"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.run_case(config=config, name=f"bad-{key}-{repr(value)}")

    def test_backend_cannot_override_record_identity(self):
        config = config_fixture(backend=UnitFixtureBackend.name)
        config["devices"][0]["options"]["override_identity"] = True
        try:
            result = self.run_case(config=config)
        except ValueError:
            return
        first = result["nodes"][0]
        self.assertEqual(first["node"], "linear")
        self.assertEqual(first["device"], "gpu")
        self.assertEqual(first["backend"], UnitFixtureBackend.name)
        self.assertTrue(first["matches_scalar_reference"])

    def test_explicit_cast_changes_transfer_storage_width(self):
        graph = typed.Graph((typed.Tensor("x", (2,), "int8"), typed.Tensor("w", (2, 2), "int8", (-2, 5, 3, -4)),
                             typed.Tensor("a", (2,), "int32"), typed.Tensor("f", (2,), "float16"),
                             typed.Tensor("y", (2,), "float16")),
                            (typed.Node("linear", "matvec", ("x", "w"), "a"),
                             typed.Node("half", "cast", ("a",), "f", typed.Arithmetic("float32")),
                             typed.Node("relu", "relu", ("f",), "y", typed.Arithmetic("float32"))), ("x",), ("y",))
        result = self.run_case(graph=graph, placement={"linear": "gpu", "half": "gpu", "relu": "npu"})
        self.assertEqual(result["outputs"], {"y": [20.0, 0.0]})
        transfers = [event for event in result["events"] if event.get("tensor") == "f"]
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0]["bytes"], 4)
        self.assertEqual(result["logical_transfer_bytes"], 14)

    def test_idle_power_all_configured_devices_charged(self):
        config = config_fixture()
        config["devices"][0]["idle_power_w"] = 2
        config["devices"][1]["idle_power_w"] = 3
        result = self.run_case(config=config, placement={"linear": "gpu", "relu": "gpu"})
        self.assertEqual(result["idle_energy_pj"], 5 * result["modeled_latency_ns"] * 1000)

    def test_unknown_energy_is_not_zero_or_complete(self):
        config = config_fixture(backend=UnitFixtureBackend.name)
        config["devices"][0]["options"]["energy"] = None
        result = self.run_case(config=config)
        self.assertFalse(result["energy_complete"])
        self.assertIsNone(result["modeled_energy_pj"])
        self.assertEqual(result["protocol"]["energy_missing_components"], ["linear"])

    def test_compare_requires_same_workload_platform_protocol_and_quality(self):
        original = self.run_case()
        self.assertTrue(unified.compare(original, original)["same_model_comparable"])
        for key, value in (("workload_sha256", "0" * 64), ("platform_sha256", "1" * 64),
                           ("protocol", {}), ("quality", {"passed": False})):
            modified = copy.deepcopy(original)
            modified[key] = value
            self.assertFalse(unified.compare(original, modified)["same_model_comparable"])
        faster = copy.deepcopy(original)
        faster["modeled_latency_ns"] /= 2
        comparison = unified.compare(original, faster)
        self.assertEqual(comparison["modeled_latency_ratio"], 2)
        self.assertFalse(comparison["hardware_performance_comparison_eligible"])

    def test_compare_rejects_missing_identity_failed_status_and_invalid_latencies(self):
        original = self.run_case()
        variants = [{"quality": {"passed": True}, "modeled_latency_ns": 1}]
        for key, value in (("status", "failed"), ("modeled_latency_ns", -1), ("modeled_latency_ns", float("nan")),
                           ("modeled_latency_ns", float("inf")), ("modeled_latency_ns", True)):
            modified = copy.deepcopy(original)
            modified[key] = value
            variants.append(modified)
        for variant in variants:
            with self.subTest(variant=str(variant)[:80]):
                try:
                    comparison = unified.compare(variant, variant)
                except ValueError:
                    continue
                self.assertFalse(comparison["same_model_comparable"])

    def test_manifest_tamper_missing_extra_and_traversal_rejected(self):
        self.run_case()
        directory = self.root / "result"
        original = (directory / "result.json").read_bytes()
        (directory / "result.json").write_bytes(original + b" ")
        with self.assertRaises(ValueError):
            unified.verify(directory)
        (directory / "result.json").write_bytes(original)
        (directory / "extra.txt").write_text("unaccounted")
        with self.assertRaises(ValueError):
            unified.verify(directory)
        (directory / "extra.txt").unlink()
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["files"]["../outside"] = "0" * 64
        (directory / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            unified.verify(directory)

    def test_manifest_rejects_empty_bundle(self):
        directory = self.root / "empty"
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {}}))
        with self.assertRaises(ValueError):
            unified.verify(directory)

    def test_manifest_does_not_follow_symlink_parent(self):
        directory, outside = self.root / "bundle", self.root / "outside"
        directory.mkdir()
        outside.mkdir()
        secret = outside / "secret"
        secret.write_text("outside artifact root")
        (directory / "alias").symlink_to(outside, target_is_directory=True)
        (directory / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {
            "alias/secret": hashlib.sha256(secret.read_bytes()).hexdigest()}}))
        hashed = []
        original_hash = unified.file_sha256
        def record_hash(path):
            hashed.append(Path(path).resolve())
            return original_hash(path)
        with mock.patch.object(unified, "file_sha256", side_effect=record_hash), self.assertRaises(ValueError):
            unified.verify(directory)
        self.assertNotIn(secret.resolve(), hashed)

    def test_no_overwrite_and_placement_validation(self):
        self.run_case()
        original = (self.root / "result/result.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "exists"):
            self.run_case()
        self.assertEqual((self.root / "result/result.json").read_bytes(), original)
        for placement in ({"linear": "gpu"}, {"linear": "gpu", "relu": "unknown"},
                          {"linear": "gpu", "relu": "npu", "extra": "gpu"}):
            with self.assertRaises(ValueError):
                self.run_case(placement=placement, name="bad-placement")
        self.assertFalse((self.root / "bad-placement").exists())

    def test_strict_config_schema_option_names_types_and_duplicates(self):
        variants = []
        for mutate in (lambda c: c.update(schema_version=True), lambda c: c.update(extra=True),
                       lambda c: c["devices"][0]["options"].update(macs_per_nanosecond=3),
                       lambda c: c["devices"][0].update(capacity_bytes=True),
                       lambda c: c["devices"][0]["options"].update(macs_per_ns=0),
                       lambda c: c["devices"][0].update(idle_power_w=-1),
                       lambda c: c["devices"].append(copy.deepcopy(c["devices"][0])),
                       lambda c: c["links"].append(copy.deepcopy(c["links"][0])),
                       lambda c: c["links"][0].update(source="unknown"),
                       lambda c: c["links"][0].update(bytes_per_ns=float("inf"))):
            config = config_fixture()
            mutate(config)
            variants.append(config)
        for index, config in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                unified.validate_config(config)

    def test_backend_registration_rejects_override_and_noncallable(self):
        for name, factory in (("analytical", UnitFixtureBackend), ("", UnitFixtureBackend), ("bad-factory", 1)):
            with self.assertRaises(ValueError):
                unified.register_backend(name, factory)


if __name__ == "__main__":
    unittest.main()

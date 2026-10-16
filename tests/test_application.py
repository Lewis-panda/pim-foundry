"""Application semantics stay independent of deployment and evidence scopes."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pim_foundry import application as app, typed, unified


def graph_fixture(dtype="float16"):
    arithmetic = typed.Arithmetic("float32")
    return typed.Graph((typed.Tensor("x", (2,), dtype),
                        typed.Tensor("weights", (2, 2), dtype, (-2.0, 5.0, 3.0, -4.0)),
                        typed.Tensor("hidden", (2,), dtype), typed.Tensor("y", (2,), dtype)),
                       (typed.Node("projection", "matvec", ("x", "weights"), "hidden", arithmetic),
                        typed.Node("activation", "relu", ("hidden",), "y", arithmetic)), ("x",), ("y",))


def platform_fixture(names=("device_a", "device_b")):
    options = {"macs_per_ns": 1, "elements_per_ns": 1, "local_bytes_per_ns": 1,
               "launch_ns": 0, "mac_energy_pj": 2, "element_energy_pj": 3, "local_energy_pj_per_byte": 4}
    return {"schema_version": 1,
            "devices": [{"name": name, "kind": "cpu", "backend": "analytical", "capacity_bytes": 1024,
                         "idle_power_w": 0, "options": copy.deepcopy(options)} for name in names],
            "links": [{"source": source, "destination": target, "bytes_per_ns": 2,
                       "latency_ns": 1, "energy_pj_per_byte": 5}
                      for source in ("host", *names) for target in ("host", *names) if source != target]}


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        unified.save(self.root / "graph.json", typed.graph_to_dict(graph_fixture()))
        unified.save(self.root / "inputs.json", {"x": [-7.0, 2.0]})
        self.manifest = {
            "schema": app.APP_SCHEMA, "id": "portable-projection", "adapter": "typed-graph/v1",
            "assets": {"graph": app.asset(self.root / "graph.json", "model", "pim-foundry.typed-graph.v1"),
                       "inputs": app.asset(self.root / "inputs.json", "inputs")},
            "parameters": {}, "scenario": dict(app.STATELESS),
            "quality": {"kind": "numerical_equivalence", "reference": "typed_scalar/v1", "atol": 0.0, "rtol": 0.0},
        }
        self.experiment = {"schema": app.EXPERIMENT_SCHEMA, "platform": platform_fixture(),
                           "mapping": {"strategy": "first_feasible", "device_preference": ["device_a", "device_b"], "max_candidates": 128},
                           "variant": {}}
        self.app_path, self.experiment_path = self.root / "application.json", self.root / "experiment.json"

    def write(self):
        unified.save(self.app_path, self.manifest)
        unified.save(self.experiment_path, self.experiment)

    def run_case(self, directory="result"):
        self.write()
        return app.run(self.app_path, self.experiment_path, self.root / directory)

    def test_shared_graph_outputs_quality_and_scoped_metrics(self):
        result = self.run_case()
        self.assertEqual(result["outputs"], {"y": [20.0, 0.0]})
        self.assertEqual(result["quality"]["numerical"]["status"], "passed")
        self.assertEqual(result["quality"]["task"]["status"], "not_evaluated")
        self.assertEqual(result["metrics"]["latency"]["unit"], "ns")
        self.assertFalse(result["hardware_performance_comparison_eligible"])
        self.assertTrue(app.verify(self.root / "result")["passed"])

    def test_same_application_runs_with_arbitrary_device_names(self):
        first = self.run_case("first")
        self.experiment["platform"] = platform_fixture(("memory_processor", "tensor_engine"))
        self.experiment["mapping"]["device_preference"] = ["tensor_engine", "memory_processor"]
        second = self.run_case("second")
        self.assertEqual(first["task_sha256"], second["task_sha256"])
        self.assertNotEqual(first["execution_sha256"], second["execution_sha256"])
        self.assertEqual(first["outputs"], second["outputs"])
        self.assertEqual(second["diagnostics"]["placement"], {"projection": "tensor_engine", "activation": "tensor_engine"})
        self.assertIn("mismatch:platform_sha256", app.compare_results(first, second)["reasons"])

    def test_explicit_multi_node_placement(self):
        self.experiment["mapping"] = {"strategy": "explicit", "placement": {"projection": "device_a", "activation": "device_b"}}
        result = self.run_case()
        self.assertEqual(result["diagnostics"]["placement"], self.experiment["mapping"]["placement"])

    def test_stateful_graph_scenario_rejected(self):
        self.manifest["scenario"] = {"kind": "autoregressive_decode", "state_policy": "retain_kv", "initial_state": "warm"}
        self.write()
        with self.assertRaisesRegex(ValueError, "cold stateless_single_batch"):
            app.inspect(self.app_path, self.experiment_path)

    def test_variant_cannot_implicitly_change_graph_semantics(self):
        self.experiment["variant"] = {"precision": "int8"}
        self.write()
        with self.assertRaisesRegex(ValueError, "variant must be empty"):
            app.inspect(self.app_path, self.experiment_path)

    def test_asset_tampering_detected_before_execution(self):
        self.write()
        unified.save(self.root / "inputs.json", {"x": [1.0, 1.0]})
        with self.assertRaisesRegex(ValueError, "modified application asset: inputs"):
            app.inspect(self.app_path, self.experiment_path)

    def test_model_role_cannot_be_hidden_as_implementation(self):
        self.manifest["assets"]["graph"]["role"] = "implementation"
        self.write()
        with self.assertRaisesRegex(ValueError, "semantic asset roles"):
            app.inspect(self.app_path, self.experiment_path)

    def test_asset_formats_must_match_the_adapter_contract(self):
        for name, invalid in (("graph", "onnx"), ("inputs", "pickle")):
            original = self.manifest["assets"][name]["format"]
            self.manifest["assets"][name]["format"] = invalid
            self.write()
            with self.assertRaisesRegex(ValueError, "format"):
                app.inspect(self.app_path, self.experiment_path)
            self.manifest["assets"][name]["format"] = original

    def test_task_identity_ignores_paths_labels_and_implementation(self):
        original = copy.deepcopy(self.manifest)
        original["assets"]["implementation"] = {"path": "/one/method.py", "role": "implementation", "format": "python", "sha256": "a" * 64}
        changed = copy.deepcopy(original)
        changed["id"] = "renamed"
        changed["assets"]["graph"]["path"] = "/other/place/model.json"
        changed["assets"]["implementation"]["sha256"] = "b" * 64
        self.assertEqual(app.task_identity(original), app.task_identity(changed))
        changed["assets"]["graph"]["sha256"] = "c" * 64
        self.assertNotEqual(app.task_identity(original), app.task_identity(changed))

    def test_quality_tolerances_and_inputs_affect_semantic_identity(self):
        original = app.task_identity(self.manifest)
        changed = copy.deepcopy(self.manifest)
        changed["quality"]["atol"] = 1.0
        self.assertNotEqual(original, app.task_identity(changed))
        changed = copy.deepcopy(self.manifest)
        changed["assets"]["inputs"]["sha256"] = "d" * 64
        self.assertNotEqual(original, app.task_identity(changed))

    def test_unsupported_native_dtype_has_node_diagnostic(self):
        graph = graph_fixture("float32")
        config = {"schema_version": 1, "devices": [{"name": "memory", "kind": "hbm", "backend": "hbm-simd16",
                  "capacity_bytes": 4096, "idle_power_w": 0,
                  "options": {"runtime_descriptor": "/not-needed-for-preflight/runtime.json",
                              "packing_bytes_per_ns": 1, "packing_energy_pj_per_byte": 0}}],
                  "links": [{"source": a, "destination": b, "bytes_per_ns": 1, "latency_ns": 0, "energy_pj_per_byte": 0}
                            for a, b in (("host", "memory"), ("memory", "host"))]}
        result = app.plan_graph(graph, config, {"strategy": "first_feasible", "device_preference": ["memory"], "max_candidates": 8})
        self.assertFalse(result["passed"])
        self.assertIn("FP16", result["nodes"]["projection"]["memory"]["reasons"][0])
        self.assertFalse(result["nodes"]["activation"]["memory"]["supported"])

    def test_missing_host_input_routes_reject_before_execution(self):
        config = self.experiment["platform"]
        config["links"] = [link for link in config["links"] if link["source"] != "host"]
        self.write()
        diagnostic = app.inspect(self.app_path, self.experiment_path)["diagnostics"]
        self.assertFalse(diagnostic["passed"])
        self.assertIn("missing_host_input_route", diagnostic["nodes"]["projection"]["device_a"]["reasons"])

    def test_missing_output_routes_reject_before_execution(self):
        config = self.experiment["platform"]
        config["links"] = [link for link in config["links"] if link["destination"] != "host"]
        self.write()
        diagnostic = app.inspect(self.app_path, self.experiment_path)["diagnostics"]
        self.assertFalse(diagnostic["passed"])
        self.assertIn("missing_output_host_route", diagnostic["nodes"]["activation"]["device_a"]["reasons"])

    def test_missing_interdevice_route_rejects_explicit_assignment(self):
        config = self.experiment["platform"]
        config["links"] = [link for link in config["links"] if (link["source"], link["destination"]) in
                           {("host", "device_a"), ("device_b", "host")}]
        self.experiment["mapping"] = {"strategy": "explicit", "placement": {"projection": "device_a", "activation": "device_b"}}
        self.write()
        diagnostic = app.inspect(self.app_path, self.experiment_path)["diagnostics"]
        self.assertFalse(diagnostic["passed"])

    def test_bounded_search_reports_incomplete_search(self):
        self.experiment["mapping"]["max_candidates"] = 1
        self.write()
        diagnostic = app.inspect(self.app_path, self.experiment_path)["diagnostics"]
        self.assertFalse(diagnostic["passed"])
        self.assertTrue(diagnostic["search_limit_reached"])
        self.assertEqual(diagnostic["reason"], "search_budget_exhausted")

    def test_preflight_failure_keeps_diagnostic_artifact(self):
        self.experiment["platform"]["devices"][0]["capacity_bytes"] = 1
        self.experiment["platform"]["devices"][1]["capacity_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "preflight failed"):
            self.run_case()
        self.assertTrue((self.root / "result/failure.json").is_file())
        self.assertFalse((self.root / "result/result.json").exists())

    def test_bundle_asset_tamper_and_extra_file_rejected(self):
        self.run_case()
        extra = self.root / "result/undeclared.txt"
        extra.write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "does not cover"):
            app.verify(self.root / "result")
        extra.unlink()
        (self.root / "result/assets/inputs.json").write_text('{"x": [1, 1]}')
        with self.assertRaisesRegex(ValueError, "modified artifact"):
            app.verify(self.root / "result")

    def test_bundle_engine_tamper_rejected(self):
        self.run_case()
        (self.root / "result/engine/pim_foundry/application.py").write_text("# replaced\n")
        with self.assertRaisesRegex(ValueError, "modified artifact"):
            app.verify(self.root / "result")

    def test_replay_from_snapshot_assets_after_original_inputs_deleted(self):
        self.run_case()
        (self.root / "inputs.json").unlink()
        replay = app.replay(self.root / "result", self.root / "replay")
        self.assertTrue(replay["passed"], replay)
        self.assertTrue(app.verify(self.root / "replay")["passed"])

    def test_replay_executes_bundled_engine_after_installed_engine_changes(self):
        result = self.run_case()
        original = app._engine_identity
        installed = Path(app.__file__).parent.resolve()

        def changed_installed_engine(directory):
            identity = original(directory)
            if Path(directory).resolve() == installed:
                identity["upgraded_engine_marker.py"] = "a" * 64
            return identity

        # The child actually runs the frozen engine; only the installed-version
        # fingerprint differs. This verifies outputs and execution provenance,
        # rather than asserting that a particular subprocess call was made.
        with mock.patch.object(app, "_engine_identity", side_effect=changed_installed_engine):
            replay = app.replay(self.root / "result", self.root / "replay")
        self.assertTrue(replay["passed"], replay)
        self.assertTrue(replay["checks"]["execution_sha256"])
        after = json.loads((self.root / "replay/result.json").read_text())
        self.assertEqual(result["execution_sha256"], after["execution_sha256"])

    def test_conditional_ratio_preserves_quality_and_hardware_limits(self):
        first = self.run_case()
        comparison = app.compare_results(first, copy.deepcopy(first))
        self.assertTrue(comparison["same_model_comparable"])
        self.assertEqual(comparison["left_over_right_ratio"], 1.0)
        self.assertFalse(comparison["quality_preserving_comparison_eligible"])
        self.assertFalse(comparison["hardware_performance_comparison_eligible"])

    def test_comparison_rejects_task_and_accounting_changes(self):
        first = self.run_case()
        for field in ("task_sha256", "platform_sha256", "cost_contract_sha256"):
            changed = copy.deepcopy(first)
            changed[field] = "f" * 64
            self.assertIn("mismatch:" + field, app.compare_results(first, changed)["reasons"])

    def test_comparison_rejects_metric_unit_scope_quality_and_zero(self):
        first = self.run_case()
        for field, value in (("unit", "cycles"), ("scope", "kernel_only")):
            changed = copy.deepcopy(first)
            changed["metrics"]["latency"][field] = value
            self.assertIn("mismatch:metric_scope_unit_or_origin", app.compare_results(first, changed)["reasons"])
        changed = copy.deepcopy(first)
        changed["quality"]["numerical"]["status"] = "failed"
        self.assertIn("right:quality_failed", app.compare_results(first, changed)["reasons"])
        changed = copy.deepcopy(first)
        changed["metrics"]["latency"]["value"] = 0
        self.assertIn("nonpositive_denominator", app.compare_results(first, changed)["reasons"])

    def test_malformed_comparison_rejects_without_crashing(self):
        first = self.run_case()
        for malformed in (None, [], "passed", {"numerical": {"status": "passed"}, "task": {}}):
            changed = copy.deepcopy(first)
            changed["quality"] = malformed
            self.assertFalse(app.compare_results(first, changed)["same_model_comparable"])

    def test_custom_registration_requires_explicit_trusted_object(self):
        with self.assertRaisesRegex(ValueError, "new name"):
            app.register_adapter("typed-graph/v1", object())
        with self.assertRaisesRegex(ValueError, "prepare/inspect/execute"):
            app.register_adapter("not-a-runtime-import", object())


if __name__ == "__main__":
    unittest.main()

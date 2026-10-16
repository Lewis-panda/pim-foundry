import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pim_foundry import application, comparison as cmp, typed, unified


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        graph = typed.Graph((typed.Tensor("x", (2,), "float32"),
                             typed.Tensor("w", (2, 2), "float32", (1., 0., 0., 1.)),
                             typed.Tensor("y", (2,), "float32")),
                            (typed.Node("p", "matvec", ("x", "w"), "y", typed.Arithmetic("float32")),), ("x",), ("y",))
        write(self.root / "graph.json", typed.graph_to_dict(graph))
        write(self.root / "input.json", {"x": [1., -2.]})
        manifest = {"schema": application.APP_SCHEMA, "id": "comparison-test", "adapter": "typed-graph/v1",
                    "assets": {"graph": application.asset(self.root / "graph.json", "model", "pim-foundry.typed-graph.v1"),
                               "inputs": application.asset(self.root / "input.json", "inputs")},
                    "parameters": {}, "scenario": dict(application.STATELESS),
                    "quality": {"kind": "numerical_equivalence", "reference": "typed_scalar/v1", "atol": 0, "rtol": 0}}
        options = {"macs_per_ns": 1, "elements_per_ns": 1, "local_bytes_per_ns": 1, "launch_ns": 0,
                   "mac_energy_pj": 1, "element_energy_pj": 1, "local_energy_pj_per_byte": 1}
        self.platform = {"schema_version": 1, "devices": [{"name": "cpu", "kind": "cpu", "backend": "analytical",
                          "capacity_bytes": 1024, "idle_power_w": 0, "options": options}],
                         "links": [{"source": "host", "destination": "cpu", "bytes_per_ns": 1, "latency_ns": 0, "energy_pj_per_byte": 0},
                                   {"source": "cpu", "destination": "host", "bytes_per_ns": 1, "latency_ns": 0, "energy_pj_per_byte": 0}]}
        write(self.root / "application.json", manifest)
        experiment = {"schema": application.EXPERIMENT_SCHEMA, "platform": self.platform,
                      "mapping": {"strategy": "explicit", "placement": {"p": "cpu"}}, "variant": {}}
        write(self.root / "experiment.json", experiment)
        self.bundle = self.root / "base"
        self.replay = self.root / "replay"
        self.result = application.run(self.root / "application.json", self.root / "experiment.json", self.bundle)
        application.run(self.root / "application.json", self.root / "experiment.json", self.replay)
        values = {key: 0 for key in cmp.UNITS}
        values.update(cpu_count=1, capacity=1024, compute=2e9, external_bandwidth=1e9, internal_bandwidth=1e9)
        self.resource = {"schema": cmp.RESOURCE_SCHEMA, "platform_sha256": self.result["platform_sha256"],
                         "resources": {name: {"value": values[name], "unit": unit, "basis": "model_assumption", "description": "test fixture"}
                                       for name, unit in cmp.UNITS.items()}, "compute_convention": "FP32; MAC=2 ops",
                         "cost_basis": "synthetic fixture; no real prices", "sources": [cmp.file_ref(self.bundle / "experiment.json")]}
        resource = write(self.root / "resources.json", self.resource)
        scope = self.result["metrics"]["latency"]["scope"]
        self.evidence = [write(self.root / "correctness.json", {
            "schema": cmp.EVIDENCE_SCHEMA, "kind": "correctness", "basis": "observed", "status": "passed",
            "result_sha256": [application.file_sha256(self.bundle / "result.json")], "scope": scope,
            "sources": [cmp.file_ref(self.bundle / "result.json")], "details": {"reference": "typed_scalar/v1"}}),
            write(self.root / "replay-evidence.json", cmp.replay_evidence(self.bundle, self.replay, scope))]
        repeats = [{"path": str(path), "manifest_sha256": application.file_sha256(path / "manifest.json")}
                   for path in (self.bundle, self.replay)]
        self.protocol = {"schema": cmp.PROTOCOL_SCHEMA, "id": "test", "track": "same_platform", "metric": "latency",
                         "quality_requirement": "numerical", "resources": {"left": cmp.file_ref(resource), "right": cmp.file_ref(resource)},
                         "fixed_axes": [], "budget_constraints": [], "cost_model_bindings": [],
                         "evidence_sha256": [application.file_sha256(p) for p in self.evidence],
                         "repetitions": {"left": repeats, "right": repeats},
                         "statistics": {"method": "deterministic", "seed": 0, "resamples": 0, "confidence": .95, "sampling_unit": "independent_runs"}}

    def compare(self):
        write(self.root / "protocol.json", self.protocol)
        return cmp.compare(self.bundle, self.bundle, self.root / "protocol.json", self.evidence)

    def test_verified_deterministic_numerical_comparison(self):
        result = self.compare()
        self.assertTrue(result["qualified"])
        self.assertEqual(result["left_over_right_ratio"], 1)
        self.assertFalse(result["hardware_performance_comparison_eligible"])
        self.assertEqual(result["statistics"]["left"]["n"], 2)

    def test_missing_task_quality_blocks_qualification(self):
        self.protocol["quality_requirement"] = "task"
        result = self.compare()
        self.assertFalse(result["qualified"])
        self.assertIn("left:task_quality_not_passed", result["qualification_gaps"])

    def test_unfair_capacity_budget_is_rejected(self):
        self.protocol["track"] = "equal_budget"
        self.protocol["budget_constraints"] = [{"resource": "capacity", "relation": "at_most", "value": 512, "relative_tolerance": 0}]
        result = self.compare()
        self.assertFalse(result["comparable"])
        self.assertIsNone(result["left_over_right_ratio"])

    def test_equal_budget_requires_a_real_constraint(self):
        self.protocol["track"] = "equal_budget"
        with self.assertRaisesRegex(ValueError, "explicit shared constraint"):
            self.compare()

    def test_known_study_failure_blocks_intact_results(self):
        write(self.root / "summary.json", {"status": "reproducibility_failed"})
        result = self.compare()
        self.assertFalse(result["comparable"])
        self.assertIn("left:study_reproducibility_failed", result["blocking_reasons"])

    def test_author_claim_does_not_qualify_observed_correctness(self):
        value = json.loads(self.evidence[0].read_text())
        value["basis"] = "author_reported"
        write(self.evidence[0], value)
        self.protocol["evidence_sha256"] = [application.file_sha256(p) for p in self.evidence]
        self.assertFalse(self.compare()["qualified"])

    def test_failed_attached_evidence_blocks_comparison(self):
        value = json.loads(self.evidence[0].read_text())
        value["status"] = "failed"
        write(self.evidence[0], value)
        self.protocol["evidence_sha256"] = [application.file_sha256(p) for p in self.evidence]
        result = self.compare()
        self.assertIn("left:failed_evidence:correctness", result["blocking_reasons"])

    def test_replay_status_is_recomputed_from_sealed_bundles(self):
        value = json.loads(self.evidence[1].read_text())
        value["status"] = "failed"
        write(self.evidence[1], value)
        self.protocol["evidence_sha256"] = [application.file_sha256(p) for p in self.evidence]
        with self.assertRaisesRegex(ValueError, "replay evidence disagrees"):
            self.compare()

    def test_duplicate_run_does_not_increase_sample_count(self):
        self.protocol["repetitions"]["left"] = [self.protocol["repetitions"]["left"][0]] * 2
        with self.assertRaisesRegex(ValueError, "same run"):
            self.compare()

    def test_resource_tampering_detected(self):
        write(self.root / "resources.json", {})
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.compare()

    def test_rehashed_false_resource_declaration_disagrees_with_execution(self):
        self.resource["resources"]["capacity"]["value"] = 512
        ref = cmp.file_ref(write(self.root / "resources.json", self.resource))
        self.protocol["resources"] = {"left": ref, "right": ref}
        result = self.compare()
        self.assertIn("left:resource_disclosure_disagrees_with_executed_config:capacity", result["blocking_reasons"])

    def test_undisclosed_budget_cannot_be_treated_as_zero(self):
        self.resource["resources"]["cost"].update(value=None, basis="undisclosed")
        write(self.root / "resources.json", self.resource)
        ref = cmp.file_ref(self.root / "resources.json")
        self.protocol["resources"] = {"left": ref, "right": ref}
        self.protocol["track"] = "equal_budget"
        self.protocol["budget_constraints"] = [{"resource": "cost", "relation": "at_most", "value": 1, "relative_tolerance": 0}]
        self.assertFalse(self.compare()["comparable"])

    def test_digest_coverage_detects_bundle_mutation(self):
        write(self.bundle / "result.json", self.result | {"status": "failed"})
        with self.assertRaisesRegex(ValueError, "modified artifact"):
            self.compare()

    def rewrite_replay_result(self, change):
        value = json.loads((self.replay / "result.json").read_text())
        change(value)
        write(self.replay / "result.json", value)
        (self.replay / "manifest.json").unlink()
        application._seal(self.replay)
        self.protocol["repetitions"] = {"left": [], "right": []}
        self.protocol["evidence_sha256"] = []

    def test_metric_scope_cannot_be_relaxed_by_whole_system_track(self):
        self.rewrite_replay_result(lambda r: r["metrics"]["latency"].update(scope="compute_only"))
        self.protocol["track"] = "whole_system"
        path = write(self.root / "protocol.json", self.protocol)
        result = cmp.compare(self.bundle, self.replay, path)
        self.assertIn("mismatch:metric_unit_scope_or_aggregation", result["blocking_reasons"])

    def test_different_cost_models_need_exact_declared_accounting_bindings(self):
        def change(value):
            value["cost_contract"]["model_variant"] = "different-cost-model"
            value["cost_contract_sha256"] = unified.digest(value["cost_contract"])
            value["metrics"]["latency"]["origin"] = "other_declared_model"
        self.rewrite_replay_result(change)
        self.protocol["track"] = "whole_system"
        path = write(self.root / "protocol.json", self.protocol)
        self.assertFalse(cmp.compare(self.bundle, self.replay, path)["comparable"])
        for side, bundle in (("left", self.bundle), ("right", self.replay)):
            value = json.loads((bundle / "result.json").read_text())
            binding = {"schema": "pim-foundry.cost-model-binding.v1", "cost_contract": value["cost_contract"],
                       "metric": {k: v for k, v in value["metrics"]["latency"].items() if k != "value"},
                       "accounting": {"included": ["device_graph"], "excluded": ["compile"]},
                       "rationale": "fault-injection test: the same modeled boundary with a changed cost model"}
            self.protocol["cost_model_bindings"].append(cmp.file_ref(write(self.root / (side + "-binding.json"), binding)))
        write(path, self.protocol)
        self.assertTrue(cmp.compare(self.bundle, self.replay, path)["comparable"])
        binding_path = self.root / "right-binding.json"
        binding = json.loads(binding_path.read_text())
        binding["accounting"]["included"] = ["compute_only"]
        write(binding_path, binding)
        self.protocol["cost_model_bindings"][1] = cmp.file_ref(binding_path)
        write(path, self.protocol)
        self.assertFalse(cmp.compare(self.bundle, self.replay, path)["comparable"])

    def test_same_cost_origin_override_still_validates_binding_content(self):
        self.rewrite_replay_result(lambda r: r["metrics"]["latency"].update(origin="different_origin"))
        self.protocol["track"] = "whole_system"
        invalid = cmp.file_ref(write(self.root / "invalid-binding.json", {}))
        self.protocol["cost_model_bindings"] = [invalid, invalid]
        with self.assertRaisesRegex(ValueError, "cost-model binding"):
            cmp.compare(self.bundle, self.replay, write(self.root / "protocol.json", self.protocol))


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_is_seeded_and_requires_five_runs(self):
        options = {"method": "bootstrap_percentile", "seed": 41, "resamples": 1000}
        self.assertEqual(cmp.summarize_samples([1, 2, 3, 4, 8], **options), cmp.summarize_samples([1, 2, 3, 4, 8], **options))
        with self.assertRaisesRegex(ValueError, "five"):
            cmp.summarize_samples([1, 2, 3, 4], **options)

    def test_deterministic_variation_fails_reproduction(self):
        result = cmp.summarize_samples([3, 3, 4])
        self.assertFalse(result["reproducibility_passed"])
        self.assertIsNone(result["confidence_interval"])
        self.assertFalse(cmp.summarize_samples([3])["reproducibility_passed"])

    def test_nonfinite_and_boolean_samples_rejected(self):
        for values in ([True, True], [float("nan")], [float("inf")], [-1]):
            with self.assertRaises(ValueError):
                cmp.summarize_samples(values)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.metric = {"name": "energy", "unit": "pJ", "scope": "device_graph", "aggregation": "one_batch"}
        model = cmp.file_ref(write(self.root / "model.json", {"synthetic": "model"}))
        platform = cmp.file_ref(write(self.root / "platform.json", {"synthetic": "platform"}))
        self.dataset = {"schema": cmp.CALIBRATION_SCHEMA, "metric": self.metric, "model_sha256": model["sha256"],
                        "platform_sha256": platform["sha256"], "model_source": model, "platform_source": platform,
                        "max_relative_error": .1, "samples": []}
        for index in range(4):
            records = {}
            input_path = self.root / f"input-{index}.txt"
            input_path.write_text(str(index))
            for kind in ("prediction", "measurement"):
                log = self.root / f"{kind}-{index}.txt"
                log.write_text(f"synthetic test fixture {kind} {index}; not hardware data\n")
                value = {"schema": "pim-foundry.metric-record.v1", "input_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                         "input_source": cmp.file_ref(input_path), "model_sha256": model["sha256"], "platform_sha256": platform["sha256"], "metric": self.metric, "value": 105 if kind == "prediction" else 100,
                         "basis": "observed" if kind == "prediction" else "externally_measured", "raw_log": cmp.file_ref(log)}
                records[kind] = cmp.file_ref(write(self.root / f"{kind}-{index}.json", value))
            self.dataset["samples"].append({"split": "train" if index < 2 else "held_out", **records})

    def evaluate(self):
        return cmp.evaluate_calibration(write(self.root / "dataset.json", self.dataset))

    def test_generic_energy_held_out_error(self):
        result = self.evaluate()
        self.assertTrue(result["passed"])
        self.assertEqual(result["held_out_max_relative_error"], .05)
        self.assertFalse(result["independently_authenticated"])

    def test_train_held_out_input_overlap_rejected(self):
        for kind in ("prediction", "measurement"):
            path = self.root / f"{kind}-2.json"
            value = json.loads(path.read_text())
            value["input_sha256"] = hashlib.sha256(b"0").hexdigest()
            value["input_source"] = cmp.file_ref(self.root / "input-0.txt")
            write(path, value)
            self.dataset["samples"][2][kind] = cmp.file_ref(path)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.evaluate()

    def test_changed_metric_boundary_rejected(self):
        path = self.root / "measurement-2.json"
        value = json.loads(path.read_text())
        value["metric"]["scope"] = "chip_compute_only"
        write(path, value)
        self.dataset["samples"][2]["measurement"] = cmp.file_ref(path)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.evaluate()

    def test_failed_held_out_threshold_is_not_overridden_by_training(self):
        path = self.root / "prediction-2.json"
        value = json.loads(path.read_text())
        value["value"] = 130
        write(path, value)
        self.dataset["samples"][2]["prediction"] = cmp.file_ref(path)
        self.assertFalse(self.evaluate()["passed"])

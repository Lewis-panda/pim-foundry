"""Comparison declarations are checked; measurements are never certified here."""

from dataclasses import replace
import json
import math
import unittest

from pim_foundry.benchmark import (
    BenchmarkContract, BenchmarkResult, END_TO_END_COSTS, LOGICAL_COSTS,
    compare, content_hash, reference_records,
)
from pim_foundry.research import demo


HASH_A, HASH_B, HASH_C = (character * 64 for character in "abc")


def contract(**changes):
    base = BenchmarkContract(
        benchmark_id="declared-test-workload/v1", workload_sha256=HASH_A,
        request_stream_sha256=HASH_B, numeric_policy="exact_integer",
        quality_metric="exact_match_fraction", quality_direction="at_least",
        quality_threshold=1, scenario="one_request", initial_state="cold",
        measurement_scope="logical_dataflow", included_costs=LOGICAL_COSTS,
        measurement_protocol_sha256=HASH_C,
    )
    return replace(base, **changes)


def result(**changes):
    base = BenchmarkResult(
        run_id="left", contract=contract(), evidence="functional",
        component_evidence=(("compute", "functional"),),
        metric="logical_transfer_bytes", samples=(80,), quality_value=1,
        system_manifest_sha256=HASH_A, engine_revision_sha256=HASH_B,
        raw_artifact_sha256=HASH_C,
    )
    return replace(base, **changes)


def physical(**changes):
    base = result(
        contract=contract(measurement_scope="end_to_end",
                          included_costs=(*END_TO_END_COSTS, "model_load"),
                          statistic="mean", min_samples=3),
        evidence="hardware_measurement",
        component_evidence=(("system", "hardware_measurement"),),
        metric="latency_ns", samples=(10, 20, 30),
    )
    return replace(base, **changes)


class SerializationAndValidationTests(unittest.TestCase):
    def test_dictionary_and_json_roundtrips_preserve_record(self):
        original = physical()
        self.assertEqual(BenchmarkResult.from_dict(original.to_dict()), original)
        encoded = json.dumps(original.to_dict(), allow_nan=False)
        self.assertEqual(BenchmarkResult.from_dict(json.loads(encoded)), original)
        self.assertEqual(original.to_dict(), json.loads(encoded))

    def test_hash_is_deterministic_for_object_key_order_and_sensitive_to_values(self):
        self.assertEqual(content_hash({"a": [1, 2], "b": 3}),
                         content_hash({"b": 3, "a": [1, 2]}))
        self.assertNotEqual(content_hash({"a": [1, 2]}), content_hash({"a": [2, 1]}))
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                content_hash({"value": invalid})

    def test_malformed_schema_and_fields_are_rejected(self):
        valid = json.loads(json.dumps(result().to_dict()))
        invalid = [None, [], {}, {**valid, "schema_version": True},
                   {**valid, "schema_version": 2}, {**valid, "schema_version": "1"},
                   {**valid, "unexpected": 1}, {**valid, "metric": []},
                   {**valid, "contract": list(valid["contract"].items())},
                   {**valid, "contract": None}, {**valid, "samples": "80"},
                   {**valid, "samples": []}, {**valid, "component_evidence": {}},
                   {**valid, "component_evidence": ["compute:functional"]},
                   {**valid, "contract": {**valid["contract"], "included_costs": "compute"}},
                   {key: value for key, value in valid.items() if key != "raw_artifact_sha256"}]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                BenchmarkResult.from_dict(record)

    def test_hash_fields_require_full_lowercase_sha256(self):
        bad_hashes = (None, "", "a" * 63, "a" * 65, "A" * 64, "g" * 64, True)
        contract_hashes = ("workload_sha256", "request_stream_sha256", "measurement_protocol_sha256")
        result_hashes = ("system_manifest_sha256", "engine_revision_sha256", "raw_artifact_sha256")
        for field in contract_hashes:
            for value in bad_hashes:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    contract(**{field: value})
        for field in result_hashes:
            for value in bad_hashes:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    result(**{field: value})

    def test_numbers_must_be_finite_and_not_boolean(self):
        invalid = (True, False, math.nan, math.inf, -math.inf, "1", None, 10**1000)
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    contract(quality_threshold=value)
                with self.assertRaises(ValueError):
                    result(quality_value=value)
                with self.assertRaises(ValueError):
                    physical(samples=(value,))
        with self.assertRaisesRegex(ValueError, "summary"):
            physical(contract=replace(physical().contract, statistic="total"),
                     samples=(1e308, 1e308))

    def test_sample_sign_type_and_container(self):
        for samples in ((), [], [1], (-1,), (True,), (1.0,)):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                result(samples=samples)
        self.assertEqual(result(samples=(0,)).value, 0)
        for metric in ("latency_ns", "energy_j", "throughput_per_s"):
            for samples in ((0,), (-1,)):
                with self.subTest(metric=metric, samples=samples), self.assertRaises(ValueError):
                    physical(metric=metric, samples=samples)

    def test_contract_enums_counts_and_cost_names_are_validated(self):
        changes = ({"quality_direction": "higher"}, {"initial_state": "unknown"},
                   {"measurement_scope": "all"}, {"comparison_track": "fastest"},
                   {"statistic": "average"}, {"min_samples": True}, {"min_samples": 0},
                   {"min_samples": 1.5}, {"included_costs": ()},
                   {"included_costs": [*LOGICAL_COSTS]},
                   {"included_costs": (*LOGICAL_COSTS, LOGICAL_COSTS[0])})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                contract(**change)

    def test_evidence_requires_consistent_named_component_modes(self):
        invalid = ({"evidence": "certified"}, {"component_evidence": ()},
                   {"component_evidence": (("unit", "functional"), ("unit", "functional"))},
                   {"component_evidence": (("unit", "unknown"),)},
                   {"component_evidence": ((" ", "functional"),)},
                   {"component_evidence": (("unit", "hardware_measurement"),)},
                   {"evidence": "hybrid_estimate", "component_evidence": (("unit", "functional"),)},
                   {"component_evidence": (("a", "functional"), ("b", "analytical"))})
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                result(**change)


class ComparisonContractTests(unittest.TestCase):
    def test_quality_threshold_is_inclusive_in_both_directions(self):
        for direction, passing, failing in (("at_least", 0.95, 0.89), ("at_most", 0.85, 0.91)):
            c = contract(quality_threshold=0.9, quality_direction=direction)
            boundary = result(contract=c, quality_value=0.9)
            with self.subTest(direction=direction):
                self.assertTrue(compare(boundary, result(contract=c, quality_value=passing))["metadata_compatible"])
                rejected = compare(boundary, result(contract=c, quality_value=failing))
                self.assertFalse(rejected["metadata_compatible"])
                self.assertIn("right:quality_threshold_failed", rejected["reasons"])
                self.assertIsNone(rejected["observations"])

    def test_protocol_workload_stream_numeric_scenario_and_initial_state_mismatch(self):
        changes = {"benchmark_id": "other/v1", "workload_sha256": HASH_B,
                   "request_stream_sha256": HASH_A, "numeric_policy": "int8_saturating",
                   "quality_metric": "another_metric", "quality_direction": "at_most",
                   "quality_threshold": 0.5, "scenario": "many_requests",
                   "initial_state": "warm", "measurement_protocol_sha256": HASH_A,
                   "statistic": "mean", "min_samples": 2}
        left = result()
        for field, value in changes.items():
            right = result(contract=replace(left.contract, **{field: value}))
            with self.subTest(field=field):
                checked = compare(left, right)
                self.assertFalse(checked["metadata_compatible"])
                self.assertIn(f"contract_mismatch:{field}", checked["reasons"])

    def test_cost_order_is_irrelevant_but_additional_cost_changes_contract(self):
        left = result()
        reordered = result(contract=contract(included_costs=tuple(reversed(LOGICAL_COSTS))))
        self.assertTrue(compare(left, reordered)["metadata_compatible"])
        expanded = result(contract=contract(included_costs=(*LOGICAL_COSTS, "additional_copy")))
        self.assertIn("contract_mismatch:included_costs", compare(left, expanded)["reasons"])

    def test_end_to_end_requires_every_declared_cost_and_cold_model_loading(self):
        for omitted in END_TO_END_COSTS:
            costs = tuple(name for name in END_TO_END_COSTS if name != omitted)
            with self.subTest(omitted=omitted), self.assertRaisesRegex(ValueError, "required costs"):
                contract(measurement_scope="end_to_end", initial_state="warm", included_costs=costs)
        with self.assertRaisesRegex(ValueError, "model_load"):
            contract(measurement_scope="end_to_end", included_costs=END_TO_END_COSTS)
        warm = contract(measurement_scope="end_to_end", initial_state="warm", included_costs=END_TO_END_COSTS)
        cold = contract(measurement_scope="end_to_end", included_costs=(*END_TO_END_COSTS, "model_load"))
        self.assertEqual(warm.initial_state, "warm")
        self.assertEqual(cold.initial_state, "cold")

    def test_logical_scope_requires_all_transfer_cost_categories(self):
        for omitted in LOGICAL_COSTS:
            costs = tuple(name for name in LOGICAL_COSTS if name != omitted)
            with self.subTest(omitted=omitted), self.assertRaisesRegex(ValueError, "required costs"):
                contract(included_costs=costs)

    def test_resource_budget_tracks_are_explicit_and_must_match(self):
        with self.assertRaises(ValueError):
            contract(comparison_track="equal_budget")
        with self.assertRaises(ValueError):
            contract(resource_budget_sha256=HASH_A)
        left = result(contract=contract(comparison_track="equal_budget", resource_budget_sha256=HASH_A))
        same = replace(left, run_id="right", system_manifest_sha256=HASH_B)
        self.assertTrue(compare(left, same)["metadata_compatible"])
        changed = replace(same, contract=replace(same.contract, resource_budget_sha256=HASH_B))
        self.assertIn("contract_mismatch:resource_budget_sha256", compare(left, changed)["reasons"])
        self.assertIn("contract_mismatch:comparison_track", compare(left, result())["reasons"])

    def test_metric_and_evidence_classes_must_match(self):
        left = physical()
        self.assertIn("metric_mismatch", compare(left, replace(left, metric="energy_j"))["reasons"])
        modeled = replace(left, evidence="analytical", component_evidence=(("system", "analytical"),),
                          calibration_protocol_sha256=HASH_A, calibration_artifact_sha256=HASH_B)
        self.assertIn("evidence_class_mismatch", compare(left, modeled)["reasons"])

    def test_system_engine_and_raw_artifacts_may_differ_without_claiming_audit(self):
        left = result()
        right = replace(left, run_id="different_system", samples=(48,),
                        system_manifest_sha256=HASH_B, engine_revision_sha256=HASH_C,
                        raw_artifact_sha256=HASH_A)
        checked = compare(left, right)
        self.assertTrue(checked["metadata_compatible"])
        self.assertEqual(checked["observations"]["left"], 80)
        self.assertEqual(checked["observations"]["right"], 48)
        self.assertFalse(checked["performance_ranking_produced"])
        self.assertFalse(checked["independently_audited"])
        self.assertFalse(checked["statistical_significance_established"])
        self.assertGreater(len(checked["external_review_required"]), 0)
        self.assertNotIn("winner", checked)


class EvidenceAndStatisticsTests(unittest.TestCase):
    def test_logical_bytes_require_single_total_observation_and_logical_scope(self):
        invalid = (result(samples=(40, 40)),
                   result(contract=contract(statistic="mean")),
                   result(contract=contract(measurement_scope="kernel", included_costs=("compute",))))
        for observation in invalid:
            with self.subTest(observation=observation):
                checked = compare(observation, observation)
                self.assertFalse(checked["metadata_compatible"])
                self.assertTrue(any("logical_bytes_require" in reason for reason in checked["reasons"]))

    def test_functional_components_cannot_support_physical_measurements(self):
        physical_contract = physical().contract
        single = result(contract=physical_contract, metric="latency_ns", samples=(10, 20, 30))
        hybrid = replace(single, evidence="hybrid_estimate",
                         component_evidence=(("compute", "hardware_measurement"), ("link", "functional")),
                         calibration_protocol_sha256=HASH_A, calibration_artifact_sha256=HASH_B)
        for observation in (single, hybrid):
            with self.subTest(evidence=observation.evidence):
                self.assertIn("left:functional_evidence_cannot_support_physical_metrics",
                              compare(observation, observation)["reasons"])

    def test_physical_metric_requires_kernel_or_end_to_end_scope(self):
        observation = physical(contract=contract())
        self.assertIn("left:physical_metrics_require_a_physical_measurement_scope",
                      compare(observation, observation)["reasons"])

    def test_modeled_performance_requires_calibration_but_measurement_does_not(self):
        for evidence in ("analytical", "cycle_simulation", "hybrid_estimate"):
            components = (("compute", evidence),) if evidence != "hybrid_estimate" else (
                ("compute", "cycle_simulation"), ("link", "analytical"))
            observation = physical(evidence=evidence, component_evidence=components)
            with self.subTest(evidence=evidence):
                checked = compare(observation, observation)
                self.assertIn("left:performance_model_calibration_not_declared", checked["reasons"])
                calibrated = replace(observation, calibration_protocol_sha256=HASH_A,
                                     calibration_artifact_sha256=HASH_B)
                self.assertTrue(compare(calibrated, calibrated)["metadata_compatible"])
        self.assertTrue(compare(physical(), physical())["metadata_compatible"])

    def test_calibration_pairs_and_protocol_mismatch(self):
        for changes in ({"calibration_protocol_sha256": HASH_A},
                        {"calibration_artifact_sha256": HASH_A},
                        {"calibration_protocol_sha256": "bad", "calibration_artifact_sha256": HASH_A}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                physical(**changes)
        left = physical(calibration_protocol_sha256=HASH_A, calibration_artifact_sha256=HASH_B)
        different_artifact = replace(left, calibration_artifact_sha256=HASH_C)
        self.assertTrue(compare(left, different_artifact)["metadata_compatible"])
        different_protocol = replace(left, calibration_protocol_sha256=HASH_B)
        self.assertIn("calibration_protocol_mismatch", compare(left, different_protocol)["reasons"])

    def test_hybrid_basis_compares_component_roles_and_modes_regardless_of_order(self):
        left = physical(evidence="hybrid_estimate",
                        component_evidence=(("compute", "cycle_simulation"), ("link", "analytical")),
                        calibration_protocol_sha256=HASH_A, calibration_artifact_sha256=HASH_B)
        reordered = replace(left, component_evidence=tuple(reversed(left.component_evidence)))
        self.assertTrue(compare(left, reordered)["metadata_compatible"])
        invalid = (
            (("compute", "analytical"), ("link", "cycle_simulation")),
            (("other_compute", "cycle_simulation"), ("link", "analytical")),
            (("compute", "hardware_measurement"), ("link", "analytical")),
        )
        for components in invalid:
            with self.subTest(components=components):
                checked = compare(left, replace(left, component_evidence=components))
                self.assertIn("hybrid_evidence_basis_mismatch", checked["reasons"])

    def test_sample_count_and_summary_statistics(self):
        samples = tuple(range(100, 0, -1))
        expected = {"total": 5050, "mean": 50.5, "median": 50.5, "p95": 95, "p99": 99}
        for statistic, value in expected.items():
            c = replace(physical().contract, statistic=statistic, min_samples=100)
            observation = physical(contract=c, samples=samples)
            with self.subTest(statistic=statistic):
                self.assertEqual(observation.value, value)
                self.assertTrue(compare(observation, observation)["metadata_compatible"])
                insufficient = replace(observation, samples=samples[:-1])
                self.assertIn("right:insufficient_samples", compare(observation, insufficient)["reasons"])

    def test_repeated_throughput_rates_cannot_be_summed(self):
        observation = physical(metric="throughput_per_s", contract=replace(physical().contract, statistic="total"))
        self.assertIn("left:throughput_rate_cannot_be_summed", compare(observation, observation)["reasons"])
        averaged = replace(observation, contract=replace(observation.contract, statistic="mean"))
        self.assertTrue(compare(averaged, averaged)["metadata_compatible"])


class ReferenceBenchmarkTests(unittest.TestCase):
    def test_demo_records_report_actual_logical_counts_and_exact_quality(self):
        experiment = demo()
        records = reference_records(experiment)
        self.assertEqual(set(records), {"sram_only", "hbm_only", "heterogeneous"})
        for name, record in records.items():
            with self.subTest(name=name):
                run = experiment["runs"][name]
                self.assertEqual(record.metric, "logical_transfer_bytes")
                self.assertEqual(record.samples, (run["plan"]["logical_transfer_bytes"],))
                self.assertEqual(record.quality_value, 1)
                self.assertEqual(record.evidence, "functional")
                self.assertEqual(record.raw_artifact_sha256, content_hash(run))
                self.assertEqual(record.contract.workload_sha256, experiment["workload_sha256"])
                self.assertEqual(BenchmarkResult.from_dict(record.to_dict()), record)
        checked = compare(records["sram_only"], records["heterogeneous"])
        self.assertTrue(checked["metadata_compatible"])
        self.assertEqual(checked["observations"]["unit"], "byte")
        self.assertFalse(checked["performance_ranking_produced"])

    def test_demo_record_generation_is_deterministic_and_failed_quality_is_ineligible(self):
        experiment = demo()
        first = reference_records(experiment)
        self.assertEqual(first, reference_records(experiment))
        changed = json.loads(json.dumps(experiment))
        changed["runs"]["sram_only"]["validation"]["matches_scalar_oracle"] = False
        rejected = reference_records(changed)["sram_only"]
        self.assertEqual(rejected.quality_value, 0)
        self.assertIn("right:quality_threshold_failed", compare(first["sram_only"], rejected)["reasons"])


if __name__ == "__main__":
    unittest.main()

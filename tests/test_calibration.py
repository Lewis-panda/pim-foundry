"""Synthetic fixtures exercise binding/error checks, never hardware evidence.

Every measurement/log created here is deliberately fabricated for unit tests.
The evaluator checks declared evidence and hashes; these fixtures do not establish
hardware accuracy or supply a publishable model-validation dataset.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pim_foundry.benchmark import BenchmarkResult, content_hash
from pim_foundry.calibration import evaluate_validation, qualify_latency
from pim_foundry.config import example_config
from pim_foundry.core import Graph, Node, Tensor
from pim_foundry.experiments import execute_case, simulation_records


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset_path = self.root / "dataset.json"
        self.config = example_config(("gpu",))
        self.predictions, self.measurements = [], []
        self.dataset = {"schema_version": 1, "description": "FABRICATED UNIT-TEST FIXTURE ONLY; no hardware measured",
                        "max_relative_error": 0.1, "samples": []}
        graph = Graph((Tensor("x", (2,)), Tensor("w", (2, 2), (1, 2, 3, 4)), Tensor("y", (2,))),
                      (Node("linear", "matvec", ("x", "w"), "y"),), ("x",), ("y",))
        for index in range(4):
            result = execute_case(graph, {"x": (index + 1, -index)}, self.config,
                                  {"linear": "gpu"}, arrivals_ns=(0, 100))
            raw_name = f"raw_{index}.txt"
            raw = f"FABRICATED TEST FIXTURE {index}; NOT A HARDWARE LOG\n".encode()
            (self.root / raw_name).write_bytes(raw)
            measurement = {
                "schema_version": 1, "evidence": "hardware_measurement",
                "workload_sha256": result["workload_sha256"], "config_sha256": result["config_sha256"],
                "placement": {"linear": "gpu"}, "arrivals_ns": [0, 100], "max_inflight": 1,
                "initial_state": "cold", "scope": "device_graph",
                "latency_ns": list(result["simulation"]["latency_ns"]),
                "provenance": {"hardware": "FABRICATED FIXTURE", "software": "unit test",
                               "method": "fabricated values for parser tests; no physical measurements"},
                "raw_log": {"path": raw_name, "sha256": hashlib.sha256(raw).hexdigest()},
            }
            self.predictions.append(result)
            self.measurements.append(measurement)
            self.write(f"prediction_{index}.json", result)
            self.write(f"measurement_{index}.json", measurement)
            self.dataset["samples"].append({"split": "calibration" if index < 2 else "validation",
                                             "prediction_file": f"prediction_{index}.json",
                                             "measurement_file": f"measurement_{index}.json"})
        self.write("dataset.json", self.dataset)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value, indent=2) + "\n")

    def record(self, index=2):
        return simulation_records("fixture-only", self.predictions[index], self.config,
                                  {"fixture-only": "not a hardware engine"})["latency_ns"]


class ValidationReportTests(FixtureCase):
    def test_actual_prediction_fixtures_roundtrip_and_keep_unaudited_status(self):
        report = evaluate_validation(self.dataset_path)
        self.assertEqual(report["status"], "held_out_validation_report")
        self.assertTrue(report["passed"])
        self.assertFalse(report["independently_audited"])
        self.assertFalse(report["parameters_fitted"])
        self.assertEqual(report["model_sha256"], content_hash(self.config["model"]))
        self.assertEqual(report["config_sha256"], content_hash(self.config))
        self.assertEqual(len(report["source_file_sha256"]), 13)
        self.assertTrue(all(not Path(name).is_absolute() for name in report["source_file_sha256"]))
        for summary in report["split_summaries"].values():
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["distinct_workloads"], 2)
            self.assertEqual(summary["mean_absolute_relative_error"], 0)
            self.assertEqual(summary["max_absolute_relative_error"], 0)
        self.assertEqual(report, evaluate_validation(self.dataset_path))
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_validation_threshold_is_inclusive_and_uses_measured_denominator(self):
        data = deepcopy(self.dataset)
        data["max_relative_error"] = 0.5
        self.write("dataset.json", data)
        measurement = deepcopy(self.measurements[2])
        measurement["latency_ns"] = [2 * value for value in measurement["latency_ns"]]
        self.write("measurement_2.json", measurement)
        report = evaluate_validation(self.dataset_path)
        self.assertTrue(report["passed"])
        self.assertEqual(report["samples"][2]["absolute_relative_error"], 0.5)
        measurement["latency_ns"] = [value + 1 for value in measurement["latency_ns"]]
        self.write("measurement_2.json", measurement)
        self.assertFalse(evaluate_validation(self.dataset_path)["passed"])

    def test_calibration_error_is_diagnostic_and_no_parameters_are_fitted(self):
        measurement = deepcopy(self.measurements[0])
        measurement["latency_ns"] = [value * 10 for value in measurement["latency_ns"]]
        self.write("measurement_0.json", measurement)
        report = evaluate_validation(self.dataset_path)
        self.assertTrue(report["passed"])
        self.assertFalse(report["split_summaries"]["calibration"]["all_within_threshold"])
        self.assertFalse(report["parameters_fitted"])

    def test_median_uses_request_samples_and_handles_even_count(self):
        predicted = self.predictions[2]["simulation"]["latency_ns"][0]
        measurement = deepcopy(self.measurements[2])
        measurement["latency_ns"] = [predicted - 1, predicted + 2]
        self.write("measurement_2.json", measurement)
        sample = evaluate_validation(self.dataset_path)["samples"][2]
        self.assertEqual(sample["measured_median_latency_ns"], predicted + 0.5)
        self.assertAlmostEqual(sample["absolute_relative_error"], 0.5 / (predicted + 0.5))

    def test_policy_and_dataset_hashes_bind_threshold_and_source_bytes(self):
        before = evaluate_validation(self.dataset_path)
        data = deepcopy(self.dataset)
        data["max_relative_error"] = 0.2
        self.write("dataset.json", data)
        changed = evaluate_validation(self.dataset_path)
        self.assertNotEqual(before["measurement_protocol_sha256"], changed["measurement_protocol_sha256"])
        self.assertNotEqual(before["dataset_sha256"], changed["dataset_sha256"])
        self.write("dataset.json", self.dataset)
        path = self.root / "prediction_2.json"
        path.write_text(path.read_text() + "\n")
        reformatted = evaluate_validation(self.dataset_path)
        self.assertEqual(before["measurement_protocol_sha256"], reformatted["measurement_protocol_sha256"])
        self.assertNotEqual(before["dataset_sha256"], reformatted["dataset_sha256"])

    def test_split_leakage_and_insufficient_distinct_workloads_are_rejected(self):
        for index in (1, 2):
            prediction, measurement = deepcopy(self.predictions[index]), deepcopy(self.measurements[index])
            prediction["workload_sha256"] = self.predictions[0]["workload_sha256"]
            measurement["workload_sha256"] = prediction["workload_sha256"]
            self.write(f"prediction_{index}.json", prediction)
            self.write(f"measurement_{index}.json", measurement)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "distinct|disjoint"):
                evaluate_validation(self.dataset_path)
            self.write(f"prediction_{index}.json", self.predictions[index])
            self.write(f"measurement_{index}.json", self.measurements[index])

    def test_one_fixed_model_and_recorded_configuration_are_required(self):
        prediction, measurement = deepcopy(self.predictions[3]), deepcopy(self.measurements[3])
        prediction["config_sha256"] = "0" * 64
        measurement["config_sha256"] = "0" * 64
        self.write("prediction_3.json", prediction)
        self.write("measurement_3.json", measurement)
        with self.assertRaisesRegex(ValueError, "same model and recorded configuration"):
            evaluate_validation(self.dataset_path)
        prediction = deepcopy(self.predictions[3])
        prediction["simulation"]["model"]["devices"]["gpu"]["launch_ns"] += 1
        prediction["model_sha256"] = content_hash(prediction["simulation"]["model"])
        self.write("prediction_3.json", prediction)
        self.write("measurement_3.json", self.measurements[3])
        with self.assertRaisesRegex(ValueError, "same model and recorded configuration"):
            evaluate_validation(self.dataset_path)


class ArtifactValidationTests(FixtureCase):
    def test_dataset_schema_sample_count_split_and_threshold_are_strict(self):
        mutations = [lambda d: d.update(extra=1), lambda d: d.update(schema_version=True),
                     lambda d: d.update(description=" "), lambda d: d.update(samples=d["samples"][:3]),
                     lambda d: d["samples"][0].update(extra=1), lambda d: d["samples"][0].update(split="train")]
        for value in (True, -0.1, 1.01, float("nan"), float("inf"), "0.1", None):
            mutations.append(lambda d, value=value: d.update(max_relative_error=value))
        for mutate in mutations:
            data = deepcopy(self.dataset)
            mutate(data)
            self.write("dataset.json", data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                evaluate_validation(self.dataset_path)

    def test_measurement_metadata_and_provenance_must_match(self):
        mutations = [
            lambda m: m.update(extra=1), lambda m: m.update(schema_version=True),
            lambda m: m.update(evidence="analytical"), lambda m: m.update(scope="end_to_end"),
            lambda m: m.update(workload_sha256="0" * 64), lambda m: m.update(config_sha256="0" * 64),
            lambda m: m.update(placement={"linear": "hbm"}), lambda m: m.update(arrivals_ns=[1, 101]),
            lambda m: m.update(max_inflight=2), lambda m: m.update(initial_state="warm"),
            lambda m: m["provenance"].update(hardware=""), lambda m: m["provenance"].update(extra="x"),
        ]
        for mutate in mutations:
            measurement = deepcopy(self.measurements[0])
            mutate(measurement)
            self.write("measurement_0.json", measurement)
            with self.subTest(measurement=measurement), self.assertRaises(ValueError):
                evaluate_validation(self.dataset_path)

    def test_measurement_latencies_reject_invalid_values_and_lengths(self):
        for values in ([], [1], [True, 1], [0, 1], [-1, 1], [float("inf"), 1],
                       [float("nan"), 1], ["1", 1], [None, 1]):
            measurement = deepcopy(self.measurements[0])
            measurement["latency_ns"] = values
            self.write("measurement_0.json", measurement)
            with self.subTest(values=values), self.assertRaises(ValueError):
                evaluate_validation(self.dataset_path)

    def test_prediction_model_and_internal_latency_consistency_are_checked(self):
        mutations = [lambda p: p.update(model_sha256="0" * 64),
                     lambda p: p["simulation"]["latency_ns"].__setitem__(0, 1),
                     lambda p: p["simulation"].update(makespan_ns=1),
                     lambda p: p["functional"]["validation"].update(matches_scalar_oracle=False),
                     lambda p: p["simulation"]["plan"]["placement"].update(linear="hbm")]
        for mutate in mutations:
            prediction = deepcopy(self.predictions[0])
            mutate(prediction)
            self.write("prediction_0.json", prediction)
            with self.subTest(prediction=prediction["model_sha256"]), self.assertRaises(ValueError):
                evaluate_validation(self.dataset_path)

    def test_paths_reject_traversal_absolute_missing_directory_and_duplicates(self):
        for name in ("../prediction_0.json", str(self.root / "prediction_0.json"),
                     "missing.json", "./prediction_0.json", "folder/../prediction_0.json"):
            data = deepcopy(self.dataset)
            data["samples"][0]["prediction_file"] = name
            self.write("dataset.json", data)
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluate_validation(self.dataset_path)
        data = deepcopy(self.dataset)
        data["samples"][1]["prediction_file"] = "prediction_0.json"
        self.write("dataset.json", data)
        with self.assertRaisesRegex(ValueError, "duplicate artifact"):
            evaluate_validation(self.dataset_path)

    def test_symlink_files_and_parent_components_are_rejected(self):
        (self.root / "alias.json").symlink_to(self.root / "prediction_0.json")
        (self.root / "aliasdir").symlink_to(self.root, target_is_directory=True)
        for name in ("alias.json", "aliasdir/prediction_0.json"):
            data = deepcopy(self.dataset)
            data["samples"][0]["prediction_file"] = name
            self.write("dataset.json", data)
            with self.assertRaisesRegex(ValueError, "symlink"):
                evaluate_validation(self.dataset_path)

    def test_raw_log_hash_tampering_and_reuse_are_rejected(self):
        (self.root / "raw_0.txt").write_text("changed fixture")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            evaluate_validation(self.dataset_path)
        data = (self.root / "raw_1.txt").read_bytes()
        (self.root / "raw_0.txt").write_bytes(data)
        measurement = deepcopy(self.measurements[0])
        measurement["raw_log"]["sha256"] = hashlib.sha256(data).hexdigest()
        self.write("measurement_0.json", measurement)
        with self.assertRaisesRegex(ValueError, "reused"):
            evaluate_validation(self.dataset_path)

    def test_duplicate_json_keys_are_not_silently_collapsed(self):
        self.dataset_path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            evaluate_validation(self.dataset_path)

    def test_overflowed_json_decimal_literal_is_rejected_even_outside_latency_fields(self):
        path = self.root / "prediction_0.json"
        text = path.read_text().replace('"total_pj": ', '"unused_large_number": 1e999, "total_pj": ', 1)
        path.write_text(text)
        with self.assertRaisesRegex(ValueError, "finite range"):
            evaluate_validation(self.dataset_path)


class LatencyQualificationTests(FixtureCase):
    def qualify(self, record=None, index=2):
        self.write("record.json", (record or self.record(index)).to_dict())
        return qualify_latency(self.root / "record.json", self.root / f"prediction_{index}.json", self.dataset_path)

    def test_exact_validation_artifact_gets_hash_bound_evidence_and_stays_analytical(self):
        qualified = BenchmarkResult.from_dict(self.qualify())
        report = evaluate_validation(self.dataset_path)
        self.assertEqual(qualified.evidence, "analytical")
        self.assertEqual(qualified.calibration_protocol_sha256, report["measurement_protocol_sha256"])
        self.assertEqual(qualified.calibration_artifact_sha256, content_hash(report))
        self.assertEqual(qualified.samples, tuple(self.predictions[2]["simulation"]["latency_ns"]))

    def test_failed_validation_and_calibration_only_artifacts_cannot_qualify(self):
        with self.assertRaisesRegex(ValueError, "validation split"):
            self.qualify(index=0)
        measurement = deepcopy(self.measurements[2])
        measurement["latency_ns"] = [value * 10 for value in measurement["latency_ns"]]
        self.write("measurement_2.json", measurement)
        with self.assertRaisesRegex(ValueError, "validation failed"):
            self.qualify()

    def test_wrong_metric_scope_statistic_raw_result_or_samples_cannot_qualify(self):
        base = self.record()
        invalid = (
            replace(base, metric="energy_j"),
            replace(base, contract=replace(base.contract, statistic="p95")),
            replace(base, contract=replace(base.contract, measurement_scope="kernel")),
            replace(base, raw_artifact_sha256="0" * 64),
            replace(base, samples=(1, 1)),
            replace(base, system_manifest_sha256="0" * 64),
            replace(base, contract=replace(base.contract, workload_sha256="0" * 64)),
            replace(base, contract=replace(base.contract, request_stream_sha256="0" * 64)),
            replace(base, contract=replace(base.contract, initial_state="warm")),
            replace(base, contract=replace(base.contract, measurement_protocol_sha256="0" * 64)),
            replace(base, contract=replace(base.contract, numeric_policy="invented_float_policy")),
            replace(base, contract=replace(base.contract, quality_threshold=0.5)),
            replace(base, contract=replace(base.contract, quality_metric="invented_task_accuracy")),
            replace(base, quality_value=0),
        )
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.qualify(record)

    def test_reformatted_copy_outside_validation_file_hash_is_rejected(self):
        self.write("record.json", self.record().to_dict())
        (self.root / "copy.json").write_text(json.dumps(self.predictions[2], separators=(",", ":")))
        with self.assertRaisesRegex(ValueError, "exact prediction artifact"):
            qualify_latency(self.root / "record.json", self.root / "copy.json", self.dataset_path)


if __name__ == "__main__":
    unittest.main()

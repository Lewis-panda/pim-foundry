"""DCC result boundaries; synthetic executables never establish hardware quality."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pim_foundry.backend import REVISION, SOURCE_HASHES
from pim_foundry.dcc_simulator import (
    CONFIG, check_request_counts, parse_statistics, run_dcc_simulator,
)
from pim_foundry.trace import analyze_trace


# Observed output from the pinned DCC demo, embedded so tests do not require a
# locally built simulator or any artifact under runs/paper-dcc-build/.
STATISTICS_TEXT = """MemorySystem:
  impl: PIMDRAM
  total_num_pim_mul_requests: 0
  total_num_pim_add_requests: 0
  memory_system_cycles: 660
  total_num_other_requests: 2048
  total_num_pim_mac_per_bank_requests: 0
  total_num_write_requests: 2048
  total_num_mac_requests: 512
  total_num_pim_mac_all_bank_requests: 0
  total_num_pim_mac_same_bank_requests: 0
  total_num_pim_ldop_requests: 1024
  total_num_pim_write_to_gemv_buffer_requests: 0
  total_num_read_requests: 1024
  total_num_pim_move_to_gemv_buffer_requests: 0
  total_num_pim_softmax_requests: 0
  total_num_pim_set_head_requests: 0
  total_num_pim_dummy_requests: 0
  total_num_pim_move_to_softmax_buffer_requests: 0
  total_num_pim_set_model_requests: 0
"""

EXPECTED_STATISTICS = {
    "total_num_pim_mul_requests": 0,
    "total_num_pim_add_requests": 0,
    "memory_system_cycles": 660,
    "total_num_other_requests": 2048,
    "total_num_pim_mac_per_bank_requests": 0,
    "total_num_write_requests": 2048,
    "total_num_mac_requests": 512,
    "total_num_pim_mac_all_bank_requests": 0,
    "total_num_pim_mac_same_bank_requests": 0,
    "total_num_pim_ldop_requests": 1024,
    "total_num_pim_write_to_gemv_buffer_requests": 0,
    "total_num_read_requests": 1024,
    "total_num_pim_move_to_gemv_buffer_requests": 0,
    "total_num_pim_softmax_requests": 0,
    "total_num_pim_set_head_requests": 0,
    "total_num_pim_dummy_requests": 0,
    "total_num_pim_move_to_softmax_buffer_requests": 0,
    "total_num_pim_set_model_requests": 0,
}

TRACE_ANALYSIS = {"opcode_counts": {
    "LD": 1024, "ST": 2048, "PIM_LD_OP1": 1024, "PIM_MAC_OP1": 512,
    "PIM_ACC_RESET": 1024, "PIM_WB_ACC": 1024, "PIM_BARRIER": 48,
}}


class DccStatisticsTests(unittest.TestCase):
    def test_actual_output_fixture_ignores_logs_and_other_subsystems(self):
        output = (
            "[Ramulator::LoadStoreTrace] [info] Loaded 6704 lines.\n"
            "Frontend:\n  impl: PIMLoadStoreTrace\n  memory_system_cycles: 999\n\n"
            + STATISTICS_TEXT
            + "\nAnotherSubsystem:\n  memory_system_cycles: 123\n"
            "  total_num_read_requests: 0\n"
        )
        self.assertEqual(parse_statistics(output), EXPECTED_STATISTICS)
        self.assertEqual(parse_statistics(STATISTICS_TEXT.replace("\n", "\r\n")), EXPECTED_STATISTICS)

    def test_each_required_statistic_must_be_present_once(self):
        for key, value in EXPECTED_STATISTICS.items():
            line = f"  {key}: {value}\n"
            with self.subTest(key=key, mode="missing"), self.assertRaises(ValueError):
                parse_statistics(STATISTICS_TEXT.replace(line, ""))
            with self.subTest(key=key, mode="duplicate"), self.assertRaises(ValueError):
                parse_statistics(STATISTICS_TEXT + line)

    def test_invalid_integer_statistics_and_nonpositive_cycles_are_rejected(self):
        for invalid in ("-1", "1.5", "1e3", "NaN", "true", "", "42 trailing"):
            for key in ("memory_system_cycles", "total_num_read_requests"):
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    value = EXPECTED_STATISTICS[key]
                    parse_statistics(STATISTICS_TEXT.replace(f"{key}: {value}", f"{key}: {invalid}"))
        with self.assertRaises(ValueError):
            parse_statistics(STATISTICS_TEXT.replace("memory_system_cycles: 660", "memory_system_cycles: 0"))

    def test_missing_or_repeated_memory_system_block_is_rejected(self):
        for text in ("", "Frontend:\n  memory_system_cycles: 660\n",
                     STATISTICS_TEXT.replace("MemorySystem:", "Frontend:"),
                     STATISTICS_TEXT + STATISTICS_TEXT):
            with self.subTest(text=text[:60]), self.assertRaises(ValueError):
                parse_statistics(text)

    def test_request_counter_contract_counts_reset_and_writeback_but_not_barriers(self):
        result = check_request_counts(EXPECTED_STATISTICS, TRACE_ANALYSIS)
        self.assertEqual(result, {name: value for name, value in EXPECTED_STATISTICS.items()
                                  if name != "memory_system_cycles"})
        analysis = deepcopy(TRACE_ANALYSIS)
        analysis["opcode_counts"]["PIM_BARRIER"] = 1000000
        self.assertEqual(check_request_counts(EXPECTED_STATISTICS, analysis), result)

    def test_every_mapped_request_counter_mismatch_is_rejected(self):
        for counter in ("total_num_read_requests", "total_num_write_requests",
                        "total_num_pim_ldop_requests", "total_num_mac_requests", "total_num_other_requests"):
            with self.subTest(counter=counter), self.assertRaises(ValueError):
                statistics = dict(EXPECTED_STATISTICS)
                statistics[counter] += 1
                check_request_counts(statistics, TRACE_ANALYSIS)
        for opcode in ("PIM_ACC_RESET", "PIM_WB_ACC"):
            with self.subTest(opcode=opcode), self.assertRaises(ValueError):
                analysis = deepcopy(TRACE_ANALYSIS)
                analysis["opcode_counts"][opcode] += 1
                check_request_counts(EXPECTED_STATISTICS, analysis)

    def test_unexpected_request_classes_cannot_be_hidden_by_matching_five_counters(self):
        for counter in ("total_num_pim_mul_requests", "total_num_pim_softmax_requests"):
            with self.subTest(counter=counter), self.assertRaises(ValueError):
                statistics = dict(EXPECTED_STATISTICS)
                statistics[counter] = 1
                check_request_counts(statistics, TRACE_ANALYSIS)


class DccRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="synthetic-dcc-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.compiled = self.root / "compiled"
        self.compiled.mkdir()
        # Test the first-party runner independently of author generators. This
        # synthetic stream matches counter fixtures, not a legal PIM program.
        trace = self.compiled / "kernel.trace"
        trace.write_text("".join(
            f"{opcode} 0x0\n" * count
            for opcode, count in TRACE_ANALYSIS["opcode_counts"].items()
        ))
        (self.compiled / "kernel.yaml").write_text(CONFIG)
        self.compile_report = {
            "schema_version": 1,
            "backend": "dcc-hbm-gemv-trace",
            "fixture": "synthetic runner-contract input; no author execution",
            "upstream": {"revision": REVISION, "sha256": dict(SOURCE_HASHES)},
            "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "analysis": analyze_trace(trace),
        }
        (self.compiled / "report.json").write_text(json.dumps(self.compile_report))
        self.assertEqual(self.compile_report["analysis"]["opcode_counts"], TRACE_ANALYSIS["opcode_counts"])
        self.executable = self.root / "synthetic simulator"
        self.library = self.root / "libramulator.so"
        self.library.write_bytes(b"Synthetic unit fixture; not a shared library.\n")
        self.output = self.root / "run output"

    def make_executable(self, stdout=None, stderr="synthetic fixture stderr\n", *, before="", after="", code=0):
        if stdout is None:
            stdout = "[synthetic fixture] Loaded 6704 lines.\n" + STATISTICS_TEXT
        self.executable.write_text(
            f"#!{sys.executable}\n"
            '"""Synthetic process fixture; does not simulate PIM."""\n'
            "import os, sys, time\nfrom pathlib import Path\n"
            + before + "\n"
            + f"sys.stdout.write({stdout!r}); sys.stdout.flush()\n"
            + f"sys.stderr.write({stderr!r}); sys.stderr.flush()\n"
            + after + "\n"
            + f"raise SystemExit({code})\n"
        )
        self.executable.chmod(0o700)

    def run_synthetic(self, **options):
        return run_dcc_simulator(self.compiled, self.executable, self.library, self.output, **options)

    def assert_evidence_limits(self, report):
        self.assertIs(report["numerical_correctness_verified"], False)
        self.assertIs(report["performance_comparison_eligible"], False)
        self.assertIs(report["hardware_execution"], False)
        self.assertEqual(json.loads((self.output / "report.json").read_text()), report)

    def test_success_checks_requests_preserves_hashes_and_does_not_promote_quality(self):
        self.make_executable(before=(
            "assert sys.argv[1:] == ['-f', 'kernel.yaml']\n"
            "assert Path('kernel.trace').is_file()\n"
            "assert 'LD_PRELOAD' not in os.environ and 'LD_AUDIT' not in os.environ\n"
            f"assert os.environ['LD_LIBRARY_PATH'] == {str(self.library.parent)!r}\n"
        ))
        with patch.dict("os.environ", {"LD_PRELOAD": "/nonexistent-test.so", "LD_AUDIT": "/bad.so",
                                       "LD_LIBRARY_PATH": "/irrelevant"}):
            report = self.run_synthetic()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["returncode"], 0)
        self.assertEqual(report["statistics"], EXPECTED_STATISTICS)
        self.assertTrue(report["accepted_request_counts_verified"])
        self.assertEqual(Path(report["working_directory"]), self.output)
        self.assertIn("final_write_and_pim_completion_tail_not_drained", report["known_issues"])
        self.assert_evidence_limits(report)
        self.assertFalse(report["runtime"]["build_provenance_independently_attested"])
        self.assertEqual((self.output / "kernel.trace").read_bytes(), (self.compiled / "kernel.trace").read_bytes())
        for artifact, digest in report["artifacts"].items():
            self.assertEqual(hashlib.sha256((self.output / artifact).read_bytes()).hexdigest(), digest)
        self.assertEqual(report["runtime"]["library_sha256"], hashlib.sha256(self.library.read_bytes()).hexdigest())

    def test_failed_process_preserves_both_logs_and_ignores_apparently_valid_statistics(self):
        self.make_executable(code=9)
        report = self.run_synthetic()
        self.assertEqual((report["status"], report["returncode"]), ("failed", 9))
        self.assertIsNone(report["statistics"])
        self.assertFalse(report["accepted_request_counts_verified"])
        self.assertIn("memory_system_cycles: 660", (self.output / "stdout.log").read_text())
        self.assertEqual((self.output / "stderr.log").read_text(), "synthetic fixture stderr\n")
        self.assert_evidence_limits(report)

    def test_timeout_preserves_partial_logs_without_count_or_quality_claim(self):
        self.make_executable(stdout="synthetic partial stdout\n", after="time.sleep(10)")
        report = self.run_synthetic(timeout_seconds=0.2)
        self.assertEqual(report["status"], "timeout")
        self.assertNotEqual(report["returncode"], 0)
        self.assertEqual((self.output / "stdout.log").read_text(), "synthetic partial stdout\n")
        self.assertEqual((self.output / "stderr.log").read_text(), "synthetic fixture stderr\n")
        self.assertFalse(report["accepted_request_counts_verified"])
        self.assert_evidence_limits(report)

    def test_invalid_statistics_or_loaded_line_count_preserves_raw_evidence(self):
        cases = {
            "missing_statistic": "[fixture] Loaded 6704 lines.\n" + STATISTICS_TEXT.replace("  memory_system_cycles: 660\n", ""),
            "request_mismatch": "[fixture] Loaded 6704 lines.\n" + STATISTICS_TEXT.replace("total_num_read_requests: 1024", "total_num_read_requests: 1023"),
            "missing_loaded_count": STATISTICS_TEXT,
            "wrong_loaded_count": "[fixture] Loaded 6703 lines.\n" + STATISTICS_TEXT,
            "duplicate_loaded_count": "[fixture] Loaded 6704 lines.\n[fixture] Loaded 6704 lines.\n" + STATISTICS_TEXT,
        }
        for label, stdout in cases.items():
            with self.subTest(label=label):
                self.output = self.root / label
                self.make_executable(stdout=stdout)
                report = self.run_synthetic()
                self.assertEqual(report["status"], "invalid_output")
                self.assertFalse(report["accepted_request_counts_verified"])
                self.assertEqual((self.output / "stdout.log").read_text(), stdout)
                self.assert_evidence_limits(report)

    def test_existing_file_or_directory_is_never_overwritten(self):
        self.make_executable()
        for kind in ("file", "directory"):
            with self.subTest(kind=kind):
                self.output = self.root / kind
                if kind == "directory":
                    self.output.mkdir()
                    sentinel = self.output / "keep.txt"
                else:
                    sentinel = self.output
                sentinel.write_text("keep this result")
                with self.assertRaises(ValueError):
                    self.run_synthetic()
                self.assertEqual(sentinel.read_text(), "keep this result")

    def test_modified_trace_or_noncanonical_config_is_rejected_before_output_creation(self):
        self.make_executable()
        trace = self.compiled / "kernel.trace"
        original = trace.read_text()
        trace.write_text(original + "LD 0x0\n")
        with self.assertRaises(ValueError):
            self.run_synthetic()
        self.assertFalse(self.output.exists())
        trace.write_text(original)
        config = self.compiled / "kernel.yaml"
        config.write_text(config.read_text().replace("clock_ratio: 1", "clock_ratio: 2", 1))
        with self.assertRaises(ValueError):
            self.run_synthetic()
        self.assertFalse(self.output.exists())

    def test_invalid_report_schema_source_and_analysis_are_rejected_before_output_creation(self):
        self.make_executable()
        path = self.compiled / "report.json"
        for label in ("not_object", "missing_version", "version_bool", "future_version", "upstream_null",
                      "upstream_array", "revision", "source_hash", "trace_hash", "analysis"):
            with self.subTest(label=label):
                report = deepcopy(self.compile_report)
                if label == "not_object":
                    report = []
                elif label == "missing_version":
                    del report["schema_version"]
                elif label == "version_bool":
                    report["schema_version"] = True
                elif label == "future_version":
                    report["schema_version"] = 999
                elif label == "upstream_null":
                    report["upstream"] = None
                elif label == "upstream_array":
                    report["upstream"] = []
                elif label == "revision":
                    report["upstream"]["revision"] = "unknown"
                elif label == "source_hash":
                    report["upstream"]["sha256"] = {}
                elif label == "trace_hash":
                    report["trace_sha256"] = "0" * 64
                else:
                    report["analysis"]["instruction_count"] -= 1
                path.write_text(json.dumps(report))
                self.output = self.root / label
                with self.assertRaises(ValueError):
                    self.run_synthetic()
                self.assertFalse(self.output.exists())

    def test_invalid_timeout_and_runtime_inputs_are_rejected_before_output_creation(self):
        self.make_executable()
        for timeout in (0, -1, 61, True, None, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.run_synthetic(timeout_seconds=timeout)
            self.assertFalse(self.output.exists())
        self.executable.chmod(0o600)
        with self.assertRaises(ValueError):
            self.run_synthetic()
        self.assertFalse(self.output.exists())
        self.executable.chmod(0o700)
        self.library = self.root / "wrong-library-name.so"
        self.library.write_bytes(b"synthetic")
        with self.assertRaises(ValueError):
            self.run_synthetic()
        self.assertFalse(self.output.exists())

    def test_input_mutation_during_execution_prevents_success_claim(self):
        self.make_executable(after="Path('kernel.trace').write_text('LD 0x0\\n')")
        report = self.run_synthetic()
        self.assertEqual(report["status"], "invalid_output")
        self.assertFalse(report["accepted_request_counts_verified"])
        self.assert_evidence_limits(report)

    def test_deleted_input_during_execution_still_preserves_failure_report(self):
        self.make_executable(after="Path('kernel.trace').unlink()")
        report = self.run_synthetic()
        self.assertEqual(report["status"], "invalid_output")
        self.assertFalse(report["accepted_request_counts_verified"])
        self.assertTrue((self.output / "stdout.log").is_file())
        self.assertTrue((self.output / "stderr.log").is_file())
        self.assert_evidence_limits(report)


if __name__ == "__main__":
    unittest.main()

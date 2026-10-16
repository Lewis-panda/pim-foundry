"""File-level workflow checks for immutable experiments and comparison rejection."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from pim_foundry.__main__ import main


class BenchmarkCliTests(unittest.TestCase):
    def call(self, *args):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            status = main(list(args))
        return status, output.getvalue(), error.getvalue()

    def test_generated_records_compare_and_incompatible_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "experiment"
            status, stdout, stderr = self.call("research-demo", "--devices", "hbm,gpu", "--out", str(out))
            self.assertEqual((status, stderr), (0, ""))
            summary = json.loads(stdout)
            self.assertEqual(summary["comparison_pairs"], 3)
            self.assertFalse(summary["hardware_performance_claimed"])
            self.assertEqual(summary, json.loads((out / "summary.json").read_text()))
            left, right = out / "benchmarks/hbm_only.json", out / "benchmarks/heterogeneous.json"
            status, stdout, stderr = self.call("benchmark-check", str(left), str(right))
            self.assertEqual((status, stderr), (0, ""))
            result = json.loads(stdout)
            self.assertTrue(result["metadata_compatible"])
            self.assertEqual(result["observations"]["left"], 672)
            self.assertEqual(result["observations"]["right"], 720)
            self.assertFalse(result["performance_ranking_produced"])
            original = right.read_bytes()
            modified = json.loads(original)
            modified["contract"]["initial_state"] = "warm"
            changed = Path(directory) / "warm.json"
            changed.write_text(json.dumps(modified))
            status, stdout, stderr = self.call("benchmark-check", str(left), str(changed))
            self.assertEqual((status, stderr), (1, ""))
            result = json.loads(stdout)
            self.assertFalse(result["metadata_compatible"])
            self.assertIn("contract_mismatch:initial_state", result["reasons"])
            self.assertIsNone(result["observations"])
            self.assertEqual(right.read_bytes(), original)

    def test_existing_experiment_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "experiment"
            status, _, _ = self.call("research-demo", "--devices", "hbm,npu", "--out", str(out))
            self.assertEqual(status, 0)
            before = {str(path.relative_to(out)): path.read_bytes() for path in out.rglob("*") if path.is_file()}
            status, stdout, stderr = self.call("research-demo", "--out", str(out))
            self.assertEqual((status, stdout), (2, ""))
            self.assertIn("already exists", stderr)
            after = {str(path.relative_to(out)): path.read_bytes() for path in out.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_invalid_devices_do_not_create_an_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            for devices in ("", "hbm,hbm", "hbm,", "hbm,unknown"):
                out = Path(directory) / "experiment"
                status, stdout, stderr = self.call("research-demo", "--devices", devices, "--out", str(out))
                self.assertEqual((status, stdout), (2, ""))
                self.assertIn("kinds", stderr)
                self.assertFalse(out.exists())

    def test_invalid_json_schema_duplicate_keys_and_nonfinite_values_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for content in ('{', '[]', '{"schema_version": true}',
                            '{"schema_version": 1, "schema_version": 2}',
                            '{"sample": NaN}', '{"sample": Infinity}'):
                with self.subTest(content=content):
                    path.write_text(content)
                    status, stdout, stderr = self.call("benchmark-check", str(path), str(path))
                    self.assertEqual((status, stdout), (2, ""))
                    self.assertTrue(stderr.startswith("error:"))


if __name__ == "__main__":
    unittest.main()

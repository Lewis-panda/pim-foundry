"""Completion-contract regressions; native checks skip if the optional build is absent."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_dcc_patched", ROOT / "scripts/build_dcc_patched.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)
DESCRIPTOR = Path(os.environ.get("PIM_FOUNDRY_DCC_RUNTIME", str(ROOT / "runs/dcc-patched-final/runtime.json")))


class PatchContractTests(unittest.TestCase):
    def test_completion_is_explicit_positive_bounded_integer(self):
        for value in (None, False, True, 0, -1, 1.0, 1000001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BUILD.config_template(value)
        config = BUILD.config_template(37)
        self.assertEqual(config.count("pim_completion_cycles:"), 1)
        self.assertEqual(len(set(BUILD.PIM_COMMANDS)), 26)
        for name in BUILD.PIM_COMMANDS:
            self.assertIn(f"      {name}: 37\n", config)

    def test_source_manifest_binds_patch_and_all_original_files(self):
        manifest = json.loads(BUILD.MANIFEST.read_text())
        self.assertEqual(manifest["source_revision"], BUILD.REVISION)
        self.assertEqual(manifest["patch_sha256"], BUILD.sha256(BUILD.PATCH))
        paths = [entry["path"] for entry in manifest["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(len(paths), 133)
        for path in paths:
            self.assertFalse(Path(path).is_absolute())
            self.assertNotIn("..", Path(path).parts)
        changed = [entry["path"] for entry in manifest["files"]
                   if entry["upstream_sha256"] != entry["patched_sha256"]]
        self.assertEqual(set(changed), {"CMakeLists.txt", "src/dram_controller/impl/hbm3_pim_controller.cpp"})


@unittest.skipUnless(DESCRIPTOR.is_file(), "optional native DCC build is not present")
class NativeCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.descriptor = json.loads(DESCRIPTOR.read_text())
        cls.runtime = cls.descriptor["runtimes"]["asan"]
        for field in ("executable", "library"):
            if BUILD.sha256(cls.runtime[field]) != cls.runtime[field + "_sha256"]:
                raise AssertionError("native build differs from descriptor")

    def execute(self, trace, config=None, expect_success=True):
        with tempfile.TemporaryDirectory(prefix="dcc-patched-regression-") as temporary:
            case = Path(temporary)
            (case / "kernel.trace").write_text(trace)
            (case / "kernel.yaml").write_text(config or BUILD.config_template(8))
            environment = dict(os.environ)
            for key in ("LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"):
                environment.pop(key, None)
            environment.update(LD_LIBRARY_PATH=str(Path(self.runtime["library"]).parent),
                               ASAN_OPTIONS="detect_leaks=0:halt_on_error=1",
                               UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
            result = subprocess.run([self.runtime["executable"], "-f", "kernel.yaml"], cwd=case,
                                    env=environment, capture_output=True, text=True, timeout=15)
        if not expect_success:
            self.assertNotEqual(result.returncode, 0)
            return result
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("ERROR: AddressSanitizer", result.stderr)
        self.assertNotIn("runtime error:", result.stderr)
        match = re.findall(r"^  memory_system_cycles: ([0-9]+)$", result.stdout, re.M)
        self.assertEqual(len(match), 1)
        return int(match[0])

    def test_idle_channels_and_empty_trace_are_safe(self):
        self.assertGreaterEqual(self.execute(""), 1)
        self.assertGreater(self.execute("LD 0x0\n"), 1)

    def test_write_recovery_tail_obeys_source_timing(self):
        default = self.execute("ST 0x0\n")
        changed = BUILD.config_template(8).replace("      preset: HBM3_5.2Gbps\n",
                                                        "      preset: HBM3_5.2Gbps\n      nWR: 31\n")
        self.assertEqual(self.execute("ST 0x0\n", changed) - default, 10)

    def test_refresh_completion_is_drained_using_source_nRFC(self):
        config = BUILD.config_template(110).replace("      preset: HBM3_5.2Gbps\n",
            "      preset: HBM3_5.2Gbps\n      nREFI: 100\n      nRFC: 20\n")
        low = self.execute("PIM_WB_ACC 0x0\nPIM_BARRIER 0x0\n", config)
        high = self.execute("PIM_WB_ACC 0x0\nPIM_BARRIER 0x0\n", config.replace("nRFC: 20", "nRFC: 30"))
        self.assertEqual(high - low, 10)

    def test_pim_writeback_tail_obeys_explicit_model(self):
        low = self.execute("PIM_WB_ACC 0x0\n", BUILD.config_template(8))
        high = self.execute("PIM_WB_ACC 0x0\n", BUILD.config_template(28))
        self.assertEqual(high - low, 20)

    def test_fences_preserve_two_completion_intervals(self):
        trace = "PIM_WB_ACC 0x0\nPIM_BARRIER 0x0\nPIM_WB_ACC 0x0\n"
        low = self.execute(trace, BUILD.config_template(8))
        high = self.execute(trace, BUILD.config_template(28))
        self.assertEqual(high - low, 40)

    def test_missing_completion_assumption_is_rejected(self):
        config = BUILD.config_template(8).replace("      WB_ACC: 8\n", "")
        result = self.execute("PIM_WB_ACC 0x0\n", config, expect_success=False)
        self.assertIn("Missing explicit PIM completion latency for WB_ACC", result.stderr)

    def test_existing_native_regression_proof_contains_paper_trace(self):
        report = json.loads((DESCRIPTOR.parent / "regressions.json").read_text())
        self.assertTrue(report["passed"])
        selected = [row for row in report["cases"] if row["case"] == "paper-b1-h32-k128-n512"]
        self.assertEqual(len(selected), 4)
        self.assertEqual({row["variant"] for row in selected}, {"release", "asan"})
        self.assertTrue(all(row["passed"] and row["deterministic_across_builds"] for row in selected))


if __name__ == "__main__":
    unittest.main()

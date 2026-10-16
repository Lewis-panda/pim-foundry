"""Bind HBM timing to the functional instruction stream and exact model inputs."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pim_foundry import banked, native_hbm
from pim_foundry.dcc_simulator import COUNTERS
from pim_foundry.trace import analyze_trace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_test_build", ROOT / "scripts/build_dcc_patched.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-hbm-binding-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.compiled = self.root / "compiled"
        banked.compile_matmul([[1, 2]], [[2, 3], [4, 5]], self.compiled)
        executable, library = self.root / "trusted-test-executable", self.root / "libramulator.so"
        executable.write_bytes(b"synthetic executable never invoked\n")
        library.write_bytes(b"synthetic library never loaded\n")
        executable.chmod(0o755)
        config = BUILD.config_template(8)
        self.descriptor = {
            "schema_version": 1, "runtime_name": "dcc-foundry-explicit-completion-v1",
            "runtimes": {"release": {"executable": str(executable), "library": str(library),
                                      "executable_sha256": sha(executable), "library_sha256": sha(library)}},
            "config_template": config, "config_template_sha256": hashlib.sha256(config.encode()).hexdigest(),
            "model": {"tCK_ps": 769, "pim_completion_cycles": {name: 8 for name in BUILD.PIM_COMMANDS},
                      "calibrated": False, "hardware_measured": False},
        }
        self.runtime_path = self.root / "runtime.json"
        self.save_descriptor()

    def save_descriptor(self):
        self.runtime_path.write_text(json.dumps(self.descriptor) + "\n")

    def rebind_config(self, config):
        self.descriptor["config_template"] = config
        self.descriptor["config_template_sha256"] = hashlib.sha256(config.encode()).hexdigest()
        self.save_descriptor()

    def fake_process(self, mutation=None):
        counts = analyze_trace(self.compiled / "kernel.trace")["opcode_counts"]
        statistics = {name: 0 for name in COUNTERS}
        statistics.update(total_num_read_requests=counts["LD"], total_num_write_requests=counts["ST"],
                          total_num_pim_ldop_requests=counts["PIM_LD_OP1"],
                          total_num_mac_requests=counts["PIM_MAC_OP1"],
                          total_num_other_requests=counts["PIM_WB_ACC"] + counts["PIM_ACC_RESET"])
        stdout = "MemorySystem:\n  impl: PIMDRAM\n  memory_system_cycles: 100\n" + "".join(
            f"  {name}: {value}\n" for name, value in statistics.items())

        def spawn(*args, **kwargs):
            kwargs["stdout"].write(stdout.encode())
            directory = Path(kwargs["cwd"])

            class Process:
                pid = 12345
                returncode = 0

                def wait(self, timeout=None):
                    if mutation:
                        mutation(directory)
                    return 0

            return Process()
        return spawn

    def test_exact_clock_config_and_hash_are_bound(self):
        for change in ("wrong-clock", "wrong-geometry", "hash", "completion"):
            with self.subTest(change=change):
                original = deepcopy(self.descriptor)
                if change == "wrong-clock":
                    self.descriptor["model"]["tCK_ps"] = 770
                elif change == "wrong-geometry":
                    config = self.descriptor["config_template"].replace("      channel: 16", "      channel: 8")
                    self.rebind_config(config)
                elif change == "hash":
                    self.descriptor["config_template_sha256"] = "0" * 64
                else:
                    self.descriptor["model"]["pim_completion_cycles"]["WB_ACC"] = 99
                self.save_descriptor()
                with patch("pim_foundry.native_hbm.subprocess.Popen") as spawn, self.assertRaises(ValueError):
                    native_hbm.execute(self.compiled, self.runtime_path, self.root / change)
                spawn.assert_not_called()
                self.descriptor = original
                self.save_descriptor()

    def test_changed_trace_rejected_before_native_execution(self):
        trace = self.compiled / "kernel.trace"
        trace.write_text(trace.read_text().replace("PIM_LD_OP1 0x00000000", "PIM_LD_OP1 0x00000020", 1))
        with patch("pim_foundry.native_hbm.subprocess.Popen") as spawn, self.assertRaises(ValueError):
            native_hbm.execute(self.compiled, self.runtime_path, self.root / "changed-trace")
        spawn.assert_not_called()

    def test_rehashed_trace_must_still_match_the_program_projection(self):
        trace = self.compiled / "kernel.trace"
        trace.write_text(trace.read_text().replace("PIM_LD_OP1 0x00000000", "PIM_LD_OP1 0x00000020", 1))
        manifest_path = self.compiled / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["kernel.trace"] = sha(trace)
        manifest_path.write_text(json.dumps(manifest))
        with patch("pim_foundry.native_hbm.subprocess.Popen", side_effect=self.fake_process()) as spawn, self.assertRaises(ValueError):
            native_hbm.execute(self.compiled, self.runtime_path, self.root / "rehashed-wrong-projection")
        spawn.assert_not_called()

    def test_rehashed_missing_dependency_barrier_is_rejected(self):
        trace = self.compiled / "kernel.trace"
        lines = trace.read_text().splitlines()
        del lines[1]
        trace.write_text("\n".join(lines) + "\n")
        manifest_path = self.compiled / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["kernel.trace"] = sha(trace)
        manifest_path.write_text(json.dumps(manifest))
        with patch("pim_foundry.native_hbm.subprocess.Popen", side_effect=self.fake_process()) as spawn, self.assertRaises(ValueError):
            native_hbm.execute(self.compiled, self.runtime_path, self.root / "missing-fence")
        spawn.assert_not_called()

    def test_changed_runtime_configuration_and_trace_during_execution_are_rejected(self):
        def mutate_trace(directory):
            path = directory / "kernel.trace"
            path.write_text(path.read_text().replace("PIM_LD_OP1 0x00000000", "PIM_LD_OP1 0x00000020", 1))

        def mutate_config(directory):
            path = directory / "kernel.yaml"
            path.write_text(path.read_text().replace("      WB_ACC: 8", "      WB_ACC: 88"))

        def mutate_descriptor(directory):
            self.runtime_path.write_text(self.runtime_path.read_text() + "\n")

        for name, mutation in (("trace", mutate_trace), ("config", mutate_config), ("descriptor", mutate_descriptor)):
            with self.subTest(name=name):
                self.save_descriptor()
                output = self.root / ("mutation-" + name)
                with patch("pim_foundry.native_hbm.subprocess.Popen", side_effect=self.fake_process(mutation)), self.assertRaises(ValueError):
                    native_hbm.execute(self.compiled, self.runtime_path, output)
                report = json.loads((output / "report.json").read_text())
                self.assertEqual(report["status"], "invalid_output")
                self.assertIsNone(report["native_timing_ns"])

    def test_synthetic_output_preserves_evidence_boundary_and_clock_conversion(self):
        with patch("pim_foundry.native_hbm.subprocess.Popen", side_effect=self.fake_process()):
            report = native_hbm.execute(self.compiled, self.runtime_path, self.root / "valid")
        self.assertEqual(report["outputs"], [[10.0, 13.0]])
        self.assertEqual(report["native_timing_ns"], 76.9)
        self.assertFalse(report["native_arithmetic"])
        self.assertFalse(report["hardware_calibrated"])
        self.assertFalse(report["performance_comparison_eligible"])


@unittest.skipUnless(os.environ.get("PIM_FOUNDRY_TEST_NATIVE_HBM") == "1", "explicit optional native integration")
class NativeIntegrationTests(unittest.TestCase):
    def test_real_asan_banked_trace(self):
        descriptor = Path(os.environ.get("PIM_FOUNDRY_DCC_RUNTIME", ROOT / "runs/dcc-patched-final/runtime.json"))
        with tempfile.TemporaryDirectory(prefix="native-hbm-asan-") as temporary:
            root = Path(temporary)
            banked.compile_matmul([[1, 2]], [[2, 3], [4, 5]], root / "compiled")
            report = native_hbm.execute(root / "compiled", descriptor, root / "native", variant="asan")
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["outputs"], [[10.0, 13.0]])
            self.assertGreater(report["native_timing_ns"], 0)
            self.assertEqual((root / "native/stderr.log").read_text(), "")


if __name__ == "__main__":
    unittest.main()

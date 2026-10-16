"""Contract tests; small CSV/source fixtures are synthetic, not paper evidence."""

import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pim_foundry import starc_adapter as adapter


class StarcAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.definition = self.root / "config.py"
        self.definition.write_text('model_table = {}\nmodel_table["Mistral-7B"] = [32, 4096, 32, 128, 7/2, 4]\n')
        self.manifest = {
            "adapter": adapter.ADAPTER, "parameters": {"model": "Mistral-7B"},
            "assets": {"model_definition": {"role": "model", "sha256": adapter._sha(self.definition)}},
            "scenario": {"kind": "llm_prefill_decode_model", "batch_size": 16, "prompt_tokens": 2048,
                         "generated_tokens": 3, "state": {"kind": "kv_cache", "representation": "shape_and_budget_only",
                         "lifetime": "one_sequence", "reset": "before_each_run"}, "request_schedule": "fixed_batch",
                         "scope": "artifact_declared_system_model"},
            "quality": {"kind": "not_evaluated", "reason": "test has no model quality data"}}

    def prepared(self):
        return adapter.prepare(self.manifest, {"model_definition": self.definition})

    def csv(self, rows, name="output.csv", headers=None):
        path = self.root / name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def output_row(self):
        return {"model": "Mistral-7B", "Lin": "2048", "Lout": "3", "bs": "16", "dtype": "W16A16",
                "s_time": "137.06833917286735", "g_time (ms)": "0.9405744286683656", "g_energy (nJ)": "564791586.345942"}

    def parse(self, path):
        return adapter.parse_csv(path, model="Mistral-7B", lin=2048, lout=3, batch=16)

    def cache_rows(self):
        return [{"L": length, "nhead": 13, "dhead": 128, "dbyte": 2, "pim_type": "BA", "power_constraint": "True",
                 "cycle": 3930, "mac": 6708, "softmax": 13, "mvgb": 936, "mvsb": 1144, "wrgb": 104}
                for length in (2049, 2050)]

    def parse_cache(self, rows):
        return adapter.parse_cache(self.csv(rows, "cache.csv"), lin=2048, lout=3, batch=16, ngpu=8, powerlimit=True)

    def test_prepare_is_hardware_independent_and_explicit_about_missing_quality(self):
        prepared = self.prepared()
        self.assertEqual(prepared["model_definition"]["declared_gqa_size"], 4)
        self.assertEqual(prepared["execution_mode"], "original_artifact_system_model")
        self.assertFalse(prepared["diagnostics"]["lowered_to_common_ir"])
        self.assertNotIn("platform", prepared)

    def test_model_asset_cannot_claim_another_definition(self):
        self.definition.write_text('model_table["Mistral-7B"] = [1, 2, 3, 4, 5, 6]\n')
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.prepared()
        self.manifest["assets"]["model_definition"]["sha256"] = adapter._sha(self.definition)
        with self.assertRaisesRegex(ValueError, "model definition"):
            self.prepared()

    def test_no_execution_of_model_definition_code(self):
        self.definition.write_text('raise RuntimeError("must not execute")\nmodel_table["Mistral-7B"] = [32,4096,32,128,7/2,4]\n')
        self.manifest["assets"]["model_definition"]["sha256"] = adapter._sha(self.definition)
        self.assertEqual(self.prepared()["model"], "Mistral-7B")

    def test_rejects_real_state_or_quality_policy(self):
        self.manifest["scenario"]["state"]["representation"] = "tensor_values"
        with self.assertRaisesRegex(ValueError, "shape-only"):
            self.prepared()
        self.manifest["scenario"]["state"]["representation"] = "shape_and_budget_only"
        self.manifest["quality"]["kind"] = "exact"
        with self.assertRaisesRegex(ValueError, "cannot evaluate"):
            self.prepared()

    def test_generated_tokens_one_and_bool_batch_rejected(self):
        self.manifest["scenario"]["generated_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "generated_tokens"):
            self.prepared()
        self.manifest["scenario"]["generated_tokens"] = 3
        self.manifest["scenario"]["batch_size"] = True
        with self.assertRaisesRegex(ValueError, "batch_size"):
            self.prepared()

    def test_csv_preserves_paper_units_and_batch_scope(self):
        result = self.parse(self.csv([self.output_row()]))
        self.assertEqual(result["decode_steps"], 2)
        self.assertEqual(result["metrics"]["decode_latency"]["value"], 0.9405744286683656)
        self.assertEqual(result["metrics"]["decode_latency"]["unit"], "ms")
        self.assertEqual(result["metrics"]["decode_energy"]["unit"], "nJ")
        self.assertIn("batch_token", result["metrics"]["decode_latency"]["scope"])
        self.assertEqual(set(result["metrics"]["decode_latency"]), {"value", "unit", "scope", "aggregation", "origin"})

    def test_csv_nonfinite_and_nonpositive_rejected(self):
        for value in ("nan", "inf", "-1", "0", "bad"):
            with self.subTest(value=value):
                row = self.output_row()
                row["g_time (ms)"] = value
                with self.assertRaises(ValueError):
                    self.parse(self.csv([row]))

    def test_csv_wrong_scenario_duplicate_rows_and_headers_rejected(self):
        row = self.output_row()
        row["bs"] = "1"
        with self.assertRaisesRegex(ValueError, "requested bs"):
            self.parse(self.csv([row]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.parse(self.csv([self.output_row(), self.output_row()]))
        path = self.root / "duplicate.csv"
        path.write_text("model,model\nx,x\n")
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.parse(path)

    def test_cache_requires_every_step_and_correct_geometry(self):
        result = self.parse_cache(self.cache_rows())
        self.assertEqual(result["rows"], 2)
        self.assertFalse(result["cache_reused_from_other_runs"])
        for key, value in (("L", 2049), ("cycle", 0), ("nhead", 1), ("pim_type", "BG"), ("power_constraint", "False")):
            with self.subTest(key=key):
                rows = self.cache_rows()
                rows[1][key] = value
                with self.assertRaises(ValueError):
                    self.parse_cache(rows)
        with self.assertRaisesRegex(ValueError, "every decode step"):
            self.parse_cache(self.cache_rows()[:1])

    def test_budget_never_silently_repeats_shorter_table(self):
        path = self.root / "budget.txt"
        path.write_text("32\n48\n")
        with self.assertRaisesRegex(ValueError, "shorter"):
            adapter._budget(path, adapter._sha(path), 3)
        path.write_text("32\n48\n64\n")
        self.assertEqual(adapter._budget(path, adapter._sha(path), 3), [32, 48, 64])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            adapter._budget(path, "0" * 64, 3)

    def test_platform_rejects_unimplemented_mapping_and_bool_gpu_count(self):
        self.assertEqual(adapter.validate_platform(adapter.DEFAULT_PLATFORM), adapter.DEFAULT_PLATFORM)
        for key, value in (("ngpu", True), ("gpu", "made-up"), ("pim", "buffer"), ("pipeopt", 1)):
            platform = {**adapter.DEFAULT_PLATFORM, key: value}
            with self.assertRaises(ValueError):
                adapter.validate_platform(platform)

    def test_inspect_does_not_execute_or_create_artifacts(self):
        before = set(self.root.iterdir())
        with mock.patch.object(adapter.subprocess, "Popen", side_effect=AssertionError("preflight must not execute")):
            result = adapter.inspect(self.prepared(), {"mapping": {"strategy": "automatic"}, "platform": adapter.DEFAULT_PLATFORM, "variant": {}})
        self.assertFalse(result["passed"])
        self.assertEqual(set(self.root.iterdir()), before)

    def test_launcher_is_valid_python_and_retains_real_artifacts(self):
        compile(adapter._NATIVE_LAUNCHER, "native-launcher", "exec")
        self.assertIn("trace_sha256", adapter._NATIVE_LAUNCHER)
        self.assertIn("result.returncode", adapter._NATIVE_LAUNCHER)

    def test_runtime_requires_complete_source_and_real_hashes(self):
        path = self.root / "runtime.json"
        descriptor = {"schema": "pim-foundry.starc-runtime.v1", "source_commit": adapter.SOURCE_COMMIT,
                      "source_root": str(self.root), "python": "/missing", "executable": "/missing",
                      "shared_library": "/missing", "source_files": {}, "runtime_files": {}, "patches": [], "validation": {}}
        path.write_text(json.dumps(descriptor))
        with self.assertRaisesRegex(ValueError, "missing application"):
            adapter.validate_runtime(path)

    def native_fixture(self):
        self.csv(self.cache_rows(), "ramulator.out")
        directory = self.root / "ramulator2/calls"
        directory.mkdir(parents=True)
        for i, length in enumerate((2049, 2050)):
            prefix = directory / f"call-{i:05d}"
            prefix.with_suffix(".json").write_text(json.dumps({"returncode": 0, "trace_sha256": hashlib.sha256(b"fixture trace").hexdigest()}))
            prefix.with_suffix(".yaml").write_text(f"Frontend:\n  path: ramulator2/attacc_l{length}_kv0_nattn13.trace\n")
            with gzip.open(prefix.with_suffix(".trace.gz"), "wb") as stream:
                stream.write(b"fixture trace")
            prefix.with_suffix(".stdout").write_text("""memory_system_cycles: 3930
total_num_pim_mac_same_bank_requests: 0
total_num_pim_mac_all_bank_requests: 6708
total_num_pim_mac_per_bank_requests: 0
total_num_pim_softmax_requests: 13
total_num_pim_move_to_gemv_buffer_requests: 936
total_num_pim_move_to_softmax_buffer_requests: 1144
total_num_pim_write_to_gemv_buffer_requests: 104
""")
        return directory

    def test_native_counts_and_preserved_traces_match_each_cache_row(self):
        self.native_fixture()
        result = adapter.verify_native_evidence(self.root)
        self.assertTrue(result["passed"])
        self.assertEqual(result["calls_verified"], 2)

    def test_native_counter_tampering_is_rejected(self):
        directory = self.native_fixture()
        path = directory / "call-00000.stdout"
        path.write_text(path.read_text().replace("3930", "3931"))
        with self.assertRaisesRegex(ValueError, "statistics differ"):
            adapter.verify_native_evidence(self.root)

    def test_native_trace_tampering_is_rejected(self):
        directory = self.native_fixture()
        with gzip.open(directory / "call-00000.trace.gz", "wb") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "trace hash differs"):
            adapter.verify_native_evidence(self.root)


if __name__ == "__main__":
    unittest.main()

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pim_foundry import cimflow_application as cim


class CimflowApplicationTests(unittest.TestCase):
    def manifest(self):
        return {"adapter": cim.ADAPTER, "parameters": {"model": "resnet18"},
                "assets": {"model": {"role": "model", "format": "onnx", "sha256": cim.MODELS["resnet18"]}},
                "scenario": {"kind": "dnn_network_shape_model", "batch_size": 8, "input_shape": [1, 3, 32, 32],
                             "state": "cold_independent_batch", "scope": cim.SCOPE},
                "quality": {"kind": "not_evaluated", "reason": "timing-only original artifact"}}

    def test_refuses_full_model_scope_and_accuracy_claim(self):
        for field, value in (("scope", "whole_model"), ("input_shape", [1, 3, 224, 224])):
            manifest = self.manifest(); manifest["scenario"][field] = value
            with patch.object(cim, "_sha", return_value=cim.MODELS["resnet18"]), self.assertRaises(ValueError):
                cim.prepare(manifest, {"model": Path("unused")})
        manifest = self.manifest(); manifest["quality"]["kind"] = "numerical_equivalence"
        with patch.object(cim, "_sha", return_value=cim.MODELS["resnet18"]), self.assertRaises(ValueError):
            cim.prepare(manifest, {"model": Path("unused")})

    def test_wrong_model_bytes_or_manifest_hash_rejected(self):
        manifest = self.manifest()
        with patch.object(cim, "_sha", return_value="0" * 64), self.assertRaises(ValueError):
            cim.prepare(manifest, {"model": Path("unused")})
        manifest["assets"]["model"]["sha256"] = "0" * 64
        with patch.object(cim, "_sha", return_value=cim.MODELS["resnet18"]), self.assertRaises(ValueError):
            cim.prepare(manifest, {"model": Path("unused")})

    def test_batch_bool_rejected(self):
        manifest = self.manifest(); manifest["scenario"]["batch_size"] = True
        with patch.object(cim, "_sha", return_value=cim.MODELS["resnet18"]), self.assertRaises(ValueError):
            cim.prepare(manifest, {"model": Path("unused")})

    def test_metrics_preserve_native_units_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "report.json"
            p.write_text(json.dumps({"latency_": 3.5, "total_energy_": 17e9, "op_count_": 88, "exec_time_": 100}))
            metrics = cim.parse_report(p, 88)
            self.assertEqual(metrics["latency"]["value"], 3.5)
            self.assertEqual(metrics["latency"]["unit"], "ms")
            self.assertEqual(metrics["energy"]["unit"], "pJ")
            self.assertEqual(metrics["latency"]["scope"], cim.SCOPE)
            with self.assertRaises(ValueError):
                cim.parse_report(p, 89)

    def test_invalid_metric_values_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "report.json"
            for value in (True, 0, -1, "3.5", float("nan")):
                p.write_text(json.dumps({"latency_": value, "total_energy_": 1, "op_count_": 1}))
                with self.assertRaises(ValueError):
                    cim.parse_report(p, 1)

    def test_deployment_cannot_claim_different_cores_or_method(self):
        experiment = {"mapping": {"strategy": "artifact_native"},
                      "platform": {"kind": "cimflow-digital-sram/v1", "cores": 64, "macro_group_size": 8,
                                   "macro_groups": 16, "noc_bytes_per_flit": 8},
                      "variant": {"runtime_descriptor": "unused", "method": "dp", "timeout_seconds": 90}}
        for group, field, value in (("platform", "cores", 32), ("variant", "method", "invented"), ("platform", "noc_bytes_per_flit", True)):
            changed = copy.deepcopy(experiment); changed[group][field] = value
            diagnostic = cim.inspect({"adapter": cim.ADAPTER}, changed)
            self.assertFalse(diagnostic["passed"])


if __name__ == "__main__":
    unittest.main()

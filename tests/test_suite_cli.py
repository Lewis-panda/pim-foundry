"""CLI workflows for external configurations, bundles and safe report content."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import html
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

from pim_foundry.__main__ import main
from pim_foundry.config import example_config, validate_config
from pim_foundry.core import Graph, Node, Tensor
from pim_foundry.experiments import build_suite, write_bundle
from pim_foundry.workloads import WORKLOAD_NAMES, make_workload, workload_from_dict, workload_to_dict


def tiny_workload():
    graph = Graph(
        (Tensor("x", (2,)), Tensor("w", (2, 2), (1, -2, 3, 4)),
         Tensor("hidden", (2,)), Tensor("y", (2,))),
        (Node("linear", "matvec", ("x", "w"), "hidden"),
         Node("relu", "relu", ("hidden",), "y")),
        ("x",), ("y",),
    )
    return graph, {"x": (-1, 2)}


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attributes):
        self.tags.append(tag)


class SuiteCliTests(unittest.TestCase):
    def call(self, *arguments):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            try:
                status = main([str(argument) for argument in arguments])
            except SystemExit as exc:
                status = exc.code
        return status, output.getvalue(), error.getvalue()

    def assert_success(self, *arguments):
        status, output, error = self.call(*arguments)
        self.assertEqual((status, error), (0, ""))
        return json.loads(output)

    def write_inputs(self, directory, *, config=None):
        root = Path(directory)
        config_file, workload_file = root / "platform.json", root / "workload.json"
        write_json(config_file, example_config() if config is None else config)
        write_json(workload_file, workload_to_dict(*tiny_workload()))
        return config_file, workload_file

    def test_init_creates_editable_config_and_all_versioned_workloads(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "project"
            result = self.assert_success("init", "--devices", "hbm,npu", "--out", out)
            self.assertEqual(result["workloads"], list(WORKLOAD_NAMES))
            config = json.loads((out / "platform.json").read_text())
            self.assertEqual([device.kind for device in validate_config(config).devices], ["hbm", "npu"])
            self.assertEqual(config["model"]["provenance"]["kind"], "synthetic")
            for name in WORKLOAD_NAMES:
                self.assertEqual(workload_from_dict(json.loads((out / f"{name}.json").read_text())), make_workload(name))

    def test_workload_export_obeys_seed_scale_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "gemm.json"
            self.assert_success("workload", "gemm", "--scale", "2", "--seed", "14", "--out", out)
            self.assertEqual(workload_from_dict(json.loads(out.read_text())), make_workload("gemm", 2, 14))
            before = out.read_bytes()
            status, output, error = self.call("workload", "gemv", "--out", out)
            self.assertEqual((status, output), (2, ""))
            self.assertTrue(error.startswith("error:"))
            self.assertEqual(out.read_bytes(), before)

    def test_small_suite_exports_report_and_can_verify_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "suite"
            result = self.assert_success("suite", "--workloads", "gemv", "--requests", "1",
                                         "--max-inflight", "1", "--search", "--out", out)
            self.assertEqual(result["evidence"], "analytical")
            self.assertFalse(result["model_validated"])
            self.assertFalse(result["hardware_performance_claimed"])
            self.assertEqual(result["cases"], [{"name": "gemv", "runs": 2, "rejected": 0}])
            self.assertEqual(Path(result["report"]), out / "report.html")
            self.assertTrue((out / "report.html").is_file())
            verification = self.assert_success("verify-bundle", out)
            self.assertTrue(verification["integrity_verified"])
            replay = self.assert_success("replay", out)
            self.assertTrue(replay["results_reproduced"])
            self.assertEqual(replay["replayed_runs"], 2)

    def test_simulate_custom_placement_and_arrival_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config, workload = self.write_inputs(directory)
            placement = Path(directory) / "placement.json"
            arrivals = Path(directory) / "arrivals.json"
            write_json(placement, {"linear": "hbm", "relu": "gpu"})
            write_json(arrivals, [5, 30])
            out = Path(directory) / "custom"
            result = self.assert_success("simulate", "--config", config, "--workload", workload,
                                         "--placement", placement, "--arrivals", arrivals,
                                         "--max-inflight", "2", "--state", "warm", "--out", out)
            self.assertEqual(result["cases"], [{"name": "custom", "runs": 1, "rejected": 0}])
            stored = json.loads((out / "results/case_000/run_000.json").read_text())
            self.assertEqual(stored["functional"]["outputs"], {"y": [5, 10]})
            self.assertEqual(stored["functional"]["plan"]["placement"], {"linear": "hbm", "relu": "gpu"})
            self.assertEqual(stored["simulation"]["arrivals_ns"], [5, 30])
            self.assertEqual(stored["simulation"]["initial_state"], "warm")
            self.assertEqual(self.assert_success("replay", out)["replayed_runs"], 1)

    def test_existing_output_is_preserved_by_init_suite_and_simulate(self):
        with tempfile.TemporaryDirectory() as directory:
            config, workload = self.write_inputs(directory)
            out = Path(directory) / "existing"
            out.mkdir()
            sentinel = out / "keep.txt"
            sentinel.write_text("keep this")
            commands = (("init",), ("suite", "--workloads", "gemv"),
                        ("simulate", "--config", config, "--workload", workload))
            for arguments in commands:
                with self.subTest(command=arguments[0]):
                    status, output, error = self.call(*arguments, "--out", out)
                    self.assertEqual((status, output), (2, ""))
                    self.assertIn("already exists", error)
                    self.assertEqual(list(out.iterdir()), [sentinel])
                    self.assertEqual(sentinel.read_text(), "keep this")

    def test_invalid_request_model_and_workload_inputs_leave_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            config, workload = self.write_inputs(directory)
            requests = (("--requests", "0"), ("--requests", "1025"),
                        ("--arrival-interval-ns", "-1"), ("--max-inflight", "0"),
                        ("--requests", "1", "--max-inflight", "2"))
            for index, arguments in enumerate(requests):
                with self.subTest(arguments=arguments):
                    out = Path(directory) / f"invalid-request-{index}"
                    status, output, error = self.call("suite", "--workloads", "gemv", *arguments, "--out", out)
                    self.assertEqual((status, output), (2, ""))
                    self.assertTrue(error.startswith("error:"))
                    self.assertFalse(out.exists())
            for content in ([], {"schema_version": 1}, {"model": []}):
                write_json(config, content)
                out = Path(directory) / "invalid-model"
                status, output, error = self.call("simulate", "--config", config, "--workload", workload, "--out", out)
                self.assertEqual((status, output), (2, ""))
                self.assertTrue(error.startswith("error:"))
                self.assertFalse(out.exists())
            write_json(config, example_config())
            for content in ([], {"schema_version": 1}, {"schema_version": True, "graph": {}, "inputs": {}}):
                write_json(workload, content)
                out = Path(directory) / "invalid-workload"
                status, output, error = self.call("simulate", "--config", config, "--workload", workload, "--out", out)
                self.assertEqual((status, output), (2, ""))
                self.assertTrue(error.startswith("error:"))
                self.assertFalse(out.exists())

    def test_invalid_arrival_and_placement_json_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            config, workload = self.write_inputs(directory)
            bad = Path(directory) / "bad.json"
            for option, values in (("--arrivals", ([], {}, [True], [1.5], [3, 2])),
                                   ("--placement", ([], {}, {"linear": "hbm", "relu": []}))):
                for value in values:
                    with self.subTest(option=option, value=value):
                        write_json(bad, value)
                        out = Path(directory) / "invalid"
                        status, output, error = self.call("simulate", "--config", config, "--workload", workload,
                                                          option, bad, "--out", out)
                        self.assertEqual((status, output), (2, ""))
                        self.assertTrue(error.startswith("error:"))
                        self.assertFalse(out.exists())

    def test_report_escapes_hostile_configuration_and_device_labels(self):
        hostile = '</script><script>window.PWNED=1</script><img src=x onerror="PWNED=2">'
        config = example_config()
        config["name"] = "ROWS CONFIG_NAME PAYLOAD " + hostile
        for device in config["platform"]["devices"]:
            if device["name"] == "gpu":
                device["name"] = hostile
        for link in config["platform"]["links"]:
            link[:] = [hostile if name == "gpu" else name for name in link]
        config["model"]["devices"][hostile] = config["model"]["devices"].pop("gpu")
        for link in config["model"]["links"]:
            for endpoint in ("source", "destination"):
                if link[endpoint] == "gpu":
                    link[endpoint] = hostile
            link["resource"] = link["resource"].replace("gpu", hostile)
        with tempfile.TemporaryDirectory() as directory:
            config_file, workload = self.write_inputs(directory, config=config)
            out = Path(directory) / "hostile-labels"
            self.assert_success("simulate", "--config", config_file, "--workload", workload,
                                "--requests", "1", "--max-inflight", "1", "--out", out)
            page = (out / "report.html").read_text()
            self.assertNotIn(hostile, page)
            self.assertIn(html.escape(config["name"]), page)
            tags = Tags()
            tags.feed(page)
            self.assertEqual(tags.tags.count("script"), 2)
            self.assertNotIn("img", tags.tags)
            payload = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
            parsed = json.loads(payload)
            self.assertTrue(any(hostile in item["label"] for item in parsed))

    def test_malformed_replay_summary_is_a_clean_cli_error(self):
        suite = build_suite(example_config(), custom_workload=tiny_workload())
        with tempfile.TemporaryDirectory() as directory:
            for index, payload in enumerate(([], {}, {"cases": [], "options": {}},
                                             {"cases": [None], "options": {}},
                                             {"cases": [{"id": "case_000", "runs": None}], "options": {}})):
                with self.subTest(payload=payload):
                    out = Path(directory) / f"malformed-{index}"
                    write_bundle(suite, out)
                    write_json(out / "summary.json", payload)
                    manifest = json.loads((out / "manifest.json").read_text())
                    manifest["files"]["summary.json"] = hashlib.sha256((out / "summary.json").read_bytes()).hexdigest()
                    write_json(out / "manifest.json", manifest)
                    status, output, error = self.call("replay", out)
                    self.assertEqual((status, output), (2, ""))
                    self.assertTrue(error.startswith("error:"))

    def test_missing_engine_identity_or_invalid_replay_options_fail_cleanly(self):
        suite = build_suite(example_config(), custom_workload=tiny_workload())
        with tempfile.TemporaryDirectory() as directory:
            for field in ("engine_source_sha256", "arrivals_ns", "max_inflight", "initial_state"):
                with self.subTest(field=field):
                    out = Path(directory) / field
                    write_bundle(suite, out)
                    summary = json.loads((out / "summary.json").read_text())
                    if field == "engine_source_sha256":
                        del summary[field]
                    else:
                        summary["options"][field] = None
                    write_json(out / "summary.json", summary)
                    manifest = json.loads((out / "manifest.json").read_text())
                    manifest["files"]["summary.json"] = hashlib.sha256((out / "summary.json").read_bytes()).hexdigest()
                    write_json(out / "manifest.json", manifest)
                    status, output, error = self.call("replay", out)
                    self.assertEqual((status, output), (2, ""))
                    self.assertTrue(error.startswith("error:"))


if __name__ == "__main__":
    unittest.main()

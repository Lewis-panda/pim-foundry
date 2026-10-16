"""Integration boundaries for configurations, modeled experiments and replay."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pim_foundry.benchmark import compare, content_hash
from pim_foundry.config import example_config, platform_from_dict, read_json, validate_config
from pim_foundry.core import Graph, Node, Tensor
from pim_foundry.experiments import (
    baseline_placements, build_suite, chrome_trace, execute_case, explore_placements,
    replay_bundle, simulation_records, verify_bundle, write_bundle,
)
from pim_foundry.simulation import simulate
from pim_foundry.workloads import workload_identity


def tiny_workload():
    graph = Graph(
        (Tensor("x", (3,)), Tensor("w1", (3, 2), (1, -1, 2, 3, -2, 4)),
         Tensor("hidden", (2,)), Tensor("activated", (2,)),
         Tensor("w2", (2, 1), (2, -3)), Tensor("y", (1,))),
        (Node("first", "matvec", ("x", "w1"), "hidden"),
         Node("activation", "relu", ("hidden",), "activated"),
         Node("last", "matvec", ("activated", "w2"), "y")),
        ("x",), ("y",),
    )
    return graph, {"x": (1, -2, 3)}


def tiny_config():
    config = example_config(("hbm", "gpu"))
    for device in config["platform"]["devices"]:
        device.update(capacity_bytes=8192, tile_k=2, tile_n=2)
    config["model"]["devices"]["gpu"]["macs_per_ns"] = 64
    config["model"]["devices"]["hbm"]["macs_per_ns"] = 8
    return config


def tiny_suite(**options):
    return build_suite(tiny_config(), names=("gemv",), custom_workload=tiny_workload(), **options)


def edit_json(path, edit):
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data, allow_nan=False))


class ConfigurationTests(unittest.TestCase):
    def test_default_configuration_json_roundtrip_and_canonical_device_order(self):
        config = example_config()
        platform = validate_config(config)
        self.assertEqual([device.kind for device in platform.devices], ["hbm", "gpu"])
        self.assertEqual(config, example_config(("gpu", "hbm")))
        restored = json.loads(json.dumps(config, allow_nan=False))
        self.assertEqual(validate_config(restored), platform)
        self.assertEqual(platform_from_dict(json.loads(json.dumps(asdict(platform)))), platform)
        self.assertEqual(config["model"]["provenance"]["kind"], "synthetic")

    def test_configuration_unknown_fields_and_missing_fields_are_rejected(self):
        paths = ((), ("platform",), ("platform", "devices", 0), ("model",),
                 ("model", "provenance"), ("model", "devices", "hbm"), ("model", "links", 0))
        for path in paths:
            for mode in ("unknown", "missing"):
                with self.subTest(path=path, mode=mode):
                    config = tiny_config()
                    target = config
                    for key in path:
                        target = target[key]
                    if mode == "unknown":
                        target["unexpected"] = 1
                    else:
                        del target[next(iter(target))]
                    with self.assertRaises(ValueError):
                        validate_config(config)

    def test_strict_configuration_types_and_resources(self):
        changes = (
            (("schema_version",), True), (("schema_version",), 1.0), (("schema_version",), 2),
            (("name",), " "), (("name",), 1), (("platform", "devices"), []),
            (("platform", "devices"), ()), (("platform", "links"), ()),
            (("platform", "devices", 0, "kind"), "unknown"),
            (("platform", "devices", 0, "name"), "host"),
            (("platform", "devices", 0, "capacity_bytes"), True),
            (("platform", "devices", 0, "capacity_bytes"), 1.0),
            (("platform", "devices", 0, "tile_k"), 0),
            (("platform", "devices", 0, "units"), -1),
            (("model", "devices", "hbm", "macs_per_ns"), False),
            (("model", "devices", "hbm", "macs_per_ns"), 0),
            (("model", "devices", "hbm", "idle_power_w"), float("nan")),
            (("model", "links", 0, "bytes_per_ns"), 0),
            (("model", "links", 0, "latency_ns"), float("inf")),
        )
        for path, value in changes:
            with self.subTest(path=path, value=value):
                config = tiny_config()
                target = config
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_links_need_known_distinct_endpoints_and_matching_costs(self):
        for link in (["host", "missing"], ["hbm", "hbm"], ["host"],
                     ["host", "hbm", "gpu"], [True, "hbm"], ("host", "hbm")):
            with self.subTest(link=link):
                config = tiny_config()
                config["platform"]["links"][0] = link
                with self.assertRaises(ValueError):
                    validate_config(config)
        for target in ("platform", "model"):
            for mode in ("duplicate", "missing"):
                with self.subTest(target=target, mode=mode):
                    config = tiny_config()
                    links = config[target]["links"]
                    if mode == "duplicate":
                        links.append(deepcopy(links[0]))
                    else:
                        links.pop()
                    with self.assertRaises(ValueError):
                        validate_config(config)

    def test_strict_json_reader_rejects_duplicate_keys_nonfinite_and_oversized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for text in ('{"x":1,"x":2}', '{"x":{"a":1,"a":2}}', '{"x":NaN}',
                         '{"x":Infinity}', '{"x":-Infinity}'):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        read_json(path)
            with path.open("wb") as stream:
                stream.seek(32 * 1024 * 1024)
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "32 MiB"):
                read_json(path)


class ExperimentTests(unittest.TestCase):
    def test_execute_case_checks_outputs_and_records_input_identity(self):
        graph, inputs = tiny_workload()
        config = tiny_config()
        placement = {"first": "hbm", "activation": "gpu", "last": "gpu"}
        result = execute_case(graph, inputs, config, placement, arrivals_ns=(0, 10), max_inflight=2)
        self.assertEqual(result["functional"]["outputs"], {"y": [-15]})
        self.assertTrue(result["functional"]["validation"]["matches_scalar_oracle"])
        self.assertEqual(result["workload_sha256"], workload_identity(graph, inputs))
        self.assertEqual(result["config_sha256"], content_hash(config))
        self.assertEqual(result["model_sha256"], content_hash(config["model"]))
        self.assertEqual(result["simulation"]["evidence"], "analytical")
        self.assertFalse(result["simulation"]["calibrated"])
        self.assertEqual(len(result["simulation"]["latency_ns"]), 2)
        self.assertEqual(result["simulation"]["logical_transfer_bytes"],
                         2 * result["functional"]["plan"]["logical_transfer_bytes"])
        self.assertEqual(result, execute_case(graph, inputs, config, placement,
                                              arrivals_ns=(0, 10), max_inflight=2))
        with self.assertRaises(ValueError):
            execute_case(graph, {"x": (1, 2)}, config, placement)

    def test_baselines_retain_platform_and_avoid_duplicate_one_node_split(self):
        suite = tiny_suite()
        case = suite["cases"][0]
        self.assertEqual(len(case["runs"]), 3)
        for run in case["runs"]:
            self.assertEqual(run["result"]["functional"]["plan"]["platform"], suite["config"]["platform"])
        graph, _ = tiny_workload()
        single = Graph(graph.tensors[:3], graph.nodes[:1], ("x",), ("hidden",))
        placements = baseline_placements(single, validate_config(tiny_config()))
        self.assertEqual(len(placements), 2)
        self.assertEqual(len({tuple(mapping.values()) for _, mapping in placements}), 2)

    def test_exhaustive_search_matches_independent_brute_force_and_frontier(self):
        graph, _ = tiny_workload()
        config = tiny_config()
        platform = validate_config(config)
        search = explore_placements(graph, platform, config["model"], candidate_limit=8)
        brute = []
        for targets in product(("gpu", "hbm"), repeat=3):
            placement = {node.name: target for node, target in zip(graph.nodes, targets)}
            report = simulate(graph, platform, placement, config["model"])
            brute.append((report["makespan_ns"], report["energy"]["total_pj"], targets))
        best = min(brute)
        self.assertEqual(len(search["candidates"]), 8)
        self.assertEqual(search["rejected"], [])
        self.assertEqual((search["best"]["makespan_ns"], search["best"]["energy_pj"],
                          tuple(search["best"]["placement"][node.name] for node in graph.nodes)), best)
        self.assertFalse(search["hardware_optimum_claimed"])
        for item in search["pareto_frontier"]:
            self.assertFalse(any(time <= item["makespan_ns"] and energy <= item["energy_pj"]
                                 and (time < item["makespan_ns"] or energy < item["energy_pj"])
                                 for time, energy, _ in brute))

    def test_search_keeps_capacity_rejections_and_enforces_limit_before_simulating(self):
        graph, _ = tiny_workload()
        config = tiny_config()
        config["platform"]["devices"][0]["capacity_bytes"] = 32
        platform = validate_config(config)
        search = explore_placements(graph, platform, config["model"], candidate_limit=8)
        self.assertEqual(len(search["candidates"]) + len(search["rejected"]), 8)
        self.assertTrue(search["candidates"])
        self.assertTrue(search["rejected"])
        self.assertTrue(all("capacity" in item["reason"] for item in search["rejected"]))
        with patch("pim_foundry.experiments.simulate", side_effect=AssertionError("must not run")):
            with self.assertRaisesRegex(ValueError, "above candidate_limit"):
                explore_placements(graph, platform, config["model"], candidate_limit=7)
        for limit in (True, 0, 4097, 8.0):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                explore_placements(graph, platform, config["model"], candidate_limit=limit)

    def test_cold_and_warm_observations_cannot_pass_uncalibrated_comparison_gate(self):
        graph, inputs = tiny_workload()
        config = tiny_config()
        placement = {node.name: "hbm" for node in graph.nodes}
        records, results = {}, {}
        for state in ("cold", "warm"):
            result = execute_case(graph, inputs, config, placement, initial_state=state)
            results[state] = result
            records[state] = simulation_records(state, result, config, {"test.py": "0" * 64})
            for metric, record in records[state].items():
                with self.subTest(state=state, metric=metric):
                    self.assertEqual(record.contract.initial_state, state)
                    self.assertEqual(record.contract.measurement_scope, "device_graph")
                    self.assertEqual("model_load" in record.contract.included_costs, state == "cold")
                    comparison = compare(record, record)
                    self.assertFalse(comparison["metadata_compatible"])
                    self.assertIn("left:performance_model_calibration_not_declared", comparison["reasons"])
                    self.assertIsNone(comparison["observations"])
        self.assertLess(results["warm"]["simulation"]["logical_transfer_bytes"],
                        results["cold"]["simulation"]["logical_transfer_bytes"])
        self.assertGreater(results["warm"]["simulation"]["persistent_weight_bytes"]["hbm"], 0)
        comparison = compare(records["cold"]["latency_ns"], records["warm"]["latency_ns"])
        self.assertIn("contract_mismatch:initial_state", comparison["reasons"])

    def test_suite_is_deterministic_and_comparisons_do_not_certify_predictions(self):
        suite = tiny_suite(search=True, candidate_limit=8)
        self.assertEqual(suite, tiny_suite(search=True, candidate_limit=8))
        self.assertFalse(suite["hardware_performance_claimed"])
        self.assertFalse(suite["model_validated"])
        self.assertTrue(suite["cases"][0]["comparisons"])
        for comparison in suite["cases"][0]["comparisons"]:
            self.assertFalse(comparison["metadata_compatible"])
            self.assertFalse(comparison["performance_ranking_produced"])
        for names in ([], (), ("gemv", "gemv"), ("unknown",), ([],), (True,)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                build_suite(tiny_config(), names=names)

    def test_chrome_trace_uses_microseconds_for_nanosecond_model(self):
        suite = tiny_suite()
        sim = suite["cases"][0]["runs"][0]["result"]["simulation"]
        trace = chrome_trace(sim)
        self.assertEqual(len(trace["traceEvents"]), len(sim["timeline"]))
        for item, source in zip(trace["traceEvents"], sim["timeline"]):
            self.assertEqual(item["ts"], source["start_ns"] / 1000)
            self.assertEqual(item["dur"], (source["end_ns"] - source["start_ns"]) / 1000)
            self.assertEqual(item["ph"], "X")


class BundleTests(unittest.TestCase):
    def test_artifacts_are_complete_byte_deterministic_and_replay(self):
        suite = tiny_suite(search=True, candidate_limit=8)
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            write_bundle(suite, first)
            write_bundle(suite, second)
            manifest = read_json(first / "manifest.json")
            self.assertTrue(manifest["complete"])
            for expected in ("config.json", "summary.json", "report.html", "workloads/case_000.json",
                             "comparisons/case_000.json", "search/case_000.json",
                             "results/case_000/run_000.json", "traces/case_000/run_000.json",
                             "benchmarks/case_000/run_000/latency_ns.json", "engine/pim_foundry/core.py"):
                self.assertIn(expected, manifest["files"])
            for filename, digest in suite["engine_source_sha256"].items():
                self.assertEqual(hashlib.sha256((first / "engine" / "pim_foundry" / filename).read_bytes()).hexdigest(), digest)
            for name, digest in manifest["files"].items():
                data = (first / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
                self.assertEqual(data, (second / name).read_bytes())
            self.assertEqual((first / "manifest.json").read_bytes(), (second / "manifest.json").read_bytes())
            verified = verify_bundle(first)
            self.assertTrue(verified["integrity_verified"])
            self.assertFalse(verified["hardware_measurements_audited"])
            replayed = replay_bundle(first)
            self.assertTrue(replayed["results_reproduced"])
            self.assertTrue(replayed["engine_sources_match"])
            self.assertEqual(replayed["replayed_runs"], len(suite["cases"][0]["runs"]))

    def test_frozen_launcher_ignores_external_engine_and_keeps_bundle_unchanged(self):
        suite = tiny_suite()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "bundle"
            write_bundle(suite, out)
            external = Path(directory) / "external" / "pim_foundry"
            external.mkdir(parents=True)
            (external / "__init__.py").write_text(
                'raise RuntimeError("the external engine must not be imported")\n'
            )
            environment = dict(os.environ, PYTHONPATH=str(external.parent))
            environment.pop("PYTHONDONTWRITEBYTECODE", None)

            def snapshot():
                return {
                    str(path.relative_to(out)): path.read_bytes() if path.is_file() else None
                    for path in out.rglob("*")
                }

            before = snapshot()
            result = subprocess.run(
                [sys.executable, str(out / "reproduce.py")], cwd="/tmp", env=environment,
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            replayed = json.loads(result.stdout)
            self.assertTrue(replayed["results_reproduced"])
            self.assertTrue(replayed["engine_sources_match"])
            self.assertEqual(replayed["replayed_runs"], len(suite["cases"][0]["runs"]))
            self.assertEqual(snapshot(), before)
            self.assertTrue(verify_bundle(out)["integrity_verified"])

    def test_bundle_never_overwrites_existing_directory_or_file(self):
        suite = tiny_suite()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "occupied"
            out.mkdir()
            sentinel = out / "keep.txt"
            sentinel.write_text("preserved")
            with self.assertRaises(ValueError):
                write_bundle(suite, out)
            self.assertEqual(sentinel.read_text(), "preserved")
            with self.assertRaises(ValueError):
                write_bundle(suite, sentinel)

    def test_serialization_failure_does_not_create_partial_output(self):
        suite = tiny_suite()
        suite["options"]["nonfinite"] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "invalid"
            with self.assertRaises(ValueError):
                write_bundle(suite, out)
            self.assertFalse(out.exists())

    def test_unsafe_export_identifiers_are_rejected_before_creating_output(self):
        original = tiny_suite()
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("case", "run", "metric", "engine"):
                with self.subTest(mode=mode):
                    suite = deepcopy(original)
                    if mode == "case":
                        suite["cases"][0]["id"] = "../../outside"
                    elif mode == "run":
                        suite["cases"][0]["runs"][0]["id"] = "../../outside"
                    elif mode == "metric":
                        records = suite["cases"][0]["runs"][0]["records"]
                        records["../../outside"] = records.pop("latency_ns")
                    else:
                        suite["engine_source_sha256"]["../outside.py"] = "0" * 64
                    out = Path(directory) / mode
                    with self.assertRaises(ValueError):
                        write_bundle(suite, out)
                    self.assertFalse(out.exists())

    def test_export_rejects_engine_snapshot_digest_mismatch(self):
        suite = tiny_suite()
        suite["engine_source_sha256"]["core.py"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "stale"
            with self.assertRaisesRegex(ValueError, "engine sources changed"):
                write_bundle(suite, out)
            self.assertFalse(out.exists())

    def test_export_rejects_empty_or_duplicate_cases_and_runs_before_writing(self):
        original = tiny_suite()
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("no_cases", "no_runs", "duplicate_case", "duplicate_run"):
                with self.subTest(mode=mode):
                    suite = deepcopy(original)
                    case = suite["cases"][0]
                    if mode == "no_cases":
                        suite["cases"] = []
                    elif mode == "no_runs":
                        case["runs"] = []
                    elif mode == "duplicate_case":
                        suite["cases"].append(deepcopy(case))
                    else:
                        case["runs"].append(deepcopy(case["runs"][0]))
                    out = Path(directory) / mode
                    with self.assertRaises(ValueError):
                        write_bundle(suite, out)
                    self.assertFalse(out.exists())

    def test_modified_missing_or_unlisted_artifacts_fail_integrity(self):
        suite = tiny_suite()
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("modified", "missing", "unlisted", "incomplete"):
                with self.subTest(mode=mode):
                    out = Path(directory) / mode
                    write_bundle(suite, out)
                    if mode == "modified":
                        with (out / "config.json").open("a") as stream:
                            stream.write(" ")
                    elif mode == "missing":
                        (out / "config.json").unlink()
                    elif mode == "unlisted":
                        (out / "unlisted.json").write_text("{}")
                    else:
                        edit_json(out / "manifest.json", lambda value: value.update(complete=False))
                    with self.assertRaises(ValueError):
                        verify_bundle(out)

    def test_manifest_rejects_relative_absolute_and_symlink_directory_escape(self):
        suite = tiny_suite()
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "external"
            external.mkdir()
            secret = external / "fixture.json"
            secret.write_text("{}")
            digest = hashlib.sha256(secret.read_bytes()).hexdigest()
            for mode in ("relative", "absolute", "symlink"):
                with self.subTest(mode=mode):
                    out = Path(directory) / mode
                    write_bundle(suite, out)
                    if mode == "relative":
                        name = "../external/fixture.json"
                    elif mode == "absolute":
                        name = str(secret)
                    else:
                        (out / "linked").symlink_to(external, target_is_directory=True)
                        name = "linked/fixture.json"
                    edit_json(out / "manifest.json", lambda value: value["files"].update({name: digest}))
                    with self.assertRaises(ValueError):
                        verify_bundle(out)

    def test_replay_rejects_unsafe_summary_identifiers_even_with_updated_digest(self):
        suite = tiny_suite()
        with tempfile.TemporaryDirectory() as directory:
            for field in ("case", "run"):
                with self.subTest(field=field):
                    out = Path(directory) / field
                    write_bundle(suite, out)

                    def change(summary):
                        case = summary["cases"][0]
                        target = case if field == "case" else case["runs"][0]
                        target["id"] = "../../outside"

                    edit_json(out / "summary.json", change)
                    digest = hashlib.sha256((out / "summary.json").read_bytes()).hexdigest()
                    edit_json(out / "manifest.json", lambda value: value["files"].update({"summary.json": digest}))
                    with self.assertRaises(ValueError):
                        replay_bundle(out)

    def test_replay_requires_nonempty_complete_unique_case_and_run_lists(self):
        suite = tiny_suite()
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("no_cases", "no_runs", "omitted_run", "duplicate_case", "duplicate_run"):
                with self.subTest(mode=mode):
                    out = Path(directory) / mode
                    write_bundle(suite, out)

                    def change(summary):
                        case = summary["cases"][0]
                        if mode == "no_cases":
                            summary["cases"] = []
                        elif mode == "no_runs":
                            case["runs"] = []
                        elif mode == "omitted_run":
                            case["runs"].pop()
                        elif mode == "duplicate_case":
                            summary["cases"].append(deepcopy(case))
                        else:
                            case["runs"].append(deepcopy(case["runs"][0]))

                    edit_json(out / "summary.json", change)
                    digest = hashlib.sha256((out / "summary.json").read_bytes()).hexdigest()
                    edit_json(out / "manifest.json", lambda value: value["files"].update({"summary.json": digest}))
                    with self.assertRaises(ValueError):
                        replay_bundle(out)

    def test_replay_rejects_unreferenced_workload_and_result_files(self):
        suite = tiny_suite()
        with tempfile.TemporaryDirectory() as directory:
            for kind, original, extra in (
                ("workload", "workloads/case_000.json", "workloads/case_999.json"),
                ("result", "results/case_000/run_000.json", "results/case_000/run_999.json"),
            ):
                with self.subTest(kind=kind):
                    out = Path(directory) / kind
                    write_bundle(suite, out)
                    data = (out / original).read_bytes()
                    (out / extra).write_bytes(data)
                    digest = hashlib.sha256(data).hexdigest()
                    edit_json(out / "manifest.json", lambda value: value["files"].update({extra: digest}))
                    self.assertTrue(verify_bundle(out)["integrity_verified"])
                    with self.assertRaises(ValueError):
                        replay_bundle(out)


if __name__ == "__main__":
    unittest.main()

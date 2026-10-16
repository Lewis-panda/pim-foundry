"""Reproducible analytical experiments, exhaustive placement search and bundles.

Search ranks predictions inside the supplied model. These are not measured
speedups, and synthetic models cannot pass the calibrated benchmark gate.
"""

from dataclasses import asdict
from itertools import combinations, product
import hashlib
import json
import re
from pathlib import Path

from .benchmark import BenchmarkContract, BenchmarkResult, compare, content_hash
from .config import json_value, read_json, validate_config
from .research import execute
from .simulation import simulate
from .workloads import WORKLOAD_NAMES, make_workload, workload_identity, workload_to_dict


def engine_sources():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def baseline_placements(graph, platform):
    """Baselines keep the same system; placement names do not remove devices."""
    names = sorted(device.name for device in platform.devices)
    result = [(f"only {name}", {node.name: name for node in graph.nodes}) for name in names]
    if len(names) > 1:
        upstream = next((d.name for d in platform.devices if d.kind == "hbm"), names[0])
        downstream = next((d.name for kind in ("gpu", "npu", "sram") for d in platform.devices
                           if d.kind == kind and d.name != upstream), next(n for n in names if n != upstream))
        result.append((f"split {upstream} + {downstream}", {
            node.name: upstream if index == 0 else downstream for index, node in enumerate(graph.nodes)
        }))
    # A one-node workload has no heterogeneous split; avoid duplicate assignments.
    unique, seen = [], set()
    for label, placement in result:
        key = tuple(placement[node.name] for node in graph.nodes)
        if key not in seen:
            unique.append((label, placement))
            seen.add(key)
    return unique


def _metrics(simulation):
    return {"makespan_ns": simulation["makespan_ns"],
            "energy_pj": simulation["energy"]["total_pj"],
            "logical_transfer_bytes": simulation["logical_transfer_bytes"]}


def explore_placements(graph, platform, model, *, arrivals_ns=(0,), max_inflight=1,
                       initial_state="cold", candidate_limit=256):
    if type(candidate_limit) is not int or not 1 <= candidate_limit <= 4096:
        raise ValueError("candidate_limit must be in 1..4096")
    names = sorted(device.name for device in platform.devices)
    count = len(names) ** len(graph.nodes)
    if count > candidate_limit:
        raise ValueError(f"search has {count} placements, above candidate_limit={candidate_limit}; reduce devices/nodes or raise the explicit limit")
    candidates, rejected = [], []
    for targets in product(names, repeat=len(graph.nodes)):
        placement = dict(zip((node.name for node in graph.nodes), targets))
        try:
            prediction = simulate(graph, platform, placement, model, arrivals_ns=arrivals_ns,
                                  max_inflight=max_inflight, initial_state=initial_state)
            candidates.append({"placement": placement, **_metrics(prediction)})
        except ValueError as exc:
            rejected.append({"placement": placement, "reason": str(exc)})
    candidates.sort(key=lambda item: (item["makespan_ns"], item["energy_pj"],
                                     tuple(item["placement"][n.name] for n in graph.nodes)))
    frontier = [item for item in candidates if not any(
        other["makespan_ns"] <= item["makespan_ns"] and other["energy_pj"] <= item["energy_pj"]
        and (other["makespan_ns"] < item["makespan_ns"] or other["energy_pj"] < item["energy_pj"])
        for other in candidates)]
    return {"objective": "modeled_makespan_ns_then_energy_pj", "evidence": "analytical",
            "hardware_optimum_claimed": False, "model_validated": False,
            "candidates": candidates, "rejected": rejected, "pareto_frontier": frontier,
            "best": candidates[0] if candidates else None}


def execute_case(graph, inputs, config, placement, *, arrivals_ns=(0,), max_inflight=1, initial_state="cold"):
    platform = validate_config(config)
    functional = execute(graph, inputs, platform, placement)
    simulation = simulate(graph, platform, placement, config["model"], arrivals_ns=arrivals_ns,
                          max_inflight=max_inflight, initial_state=initial_state)
    return json_value({"schema_version": 1, "workload_sha256": workload_identity(graph, inputs),
                       "config_sha256": content_hash(config), "model_sha256": content_hash(config["model"]),
                       "functional": functional, "simulation": simulation})


def simulation_contract(result, metric):
    sim = result["simulation"]
    state = sim["initial_state"]
    costs = ("queueing", "input_transfer", "compute", "inter_device_transfer", "synchronization", "output_transfer")
    if state == "cold":
        costs += ("model_load",)
    protocol = {
        "id": "pim-foundry/analytical-device-graph/v1", "requests": len(sim["arrivals_ns"]),
        "max_inflight": sim["max_inflight"], "initial_state": state,
        "scope": "device_graph", "statistics": "latency median; aggregate throughput; total modeled energy",
        "execution": "serialized nodes per request; overlapping fixed request slots; conservative memory reservations",
        "excluded": ["preprocessing", "postprocessing", "compiler time", "host software overhead", "warm setup"],
    }
    base = dict(benchmark_id="pim-foundry/analytical-suite/v1", workload_sha256=result["workload_sha256"],
                request_stream_sha256=content_hash(sim["arrivals_ns"]),
                numeric_policy="signed_int64_storage_exact_integer_intermediates", quality_metric="exact_output_match",
                quality_direction="at_least", quality_threshold=1, scenario="fixed_arrival_trace",
                initial_state=state, measurement_scope="device_graph", included_costs=costs,
                measurement_protocol_sha256=content_hash(protocol), comparison_track="whole_system")
    statistic = {"latency_ns": "median", "throughput_per_s": "mean", "energy_j": "total"}[metric]
    return BenchmarkContract(**base, statistic=statistic)


def simulation_records(label, result, config, sources):
    sim = result["simulation"]
    records = {}
    observations = {"latency_ns": ("median", tuple(sim["latency_ns"])),
                    "throughput_per_s": ("mean", (sim["throughput_requests_per_second"],))}
    if sim["energy"]["total_pj"] > 0:
        observations["energy_j"] = ("total", (sim["energy"]["total_pj"] / 1e12,))
    for metric, (statistic, samples) in observations.items():
        records[metric] = BenchmarkResult(
            run_id=label, contract=simulation_contract(result, metric), evidence="analytical",
            component_evidence=tuple((name, "analytical") for name in
                                     ["host_dma", "interconnect", *("device:" + d["name"] for d in config["platform"]["devices"])]),
            metric=metric, samples=samples,
            quality_value=int(result["functional"]["validation"]["matches_scalar_oracle"]),
            system_manifest_sha256=content_hash(config), engine_revision_sha256=content_hash(sources),
            raw_artifact_sha256=content_hash(result),
        )
    return records


def build_suite(config, *, names=WORKLOAD_NAMES, scale=1, seed=0, arrivals_ns=(0,), max_inflight=1,
                initial_state="cold", search=False, candidate_limit=256, custom_workload=None, placement_override=None):
    platform = validate_config(config)
    if (type(names) is not tuple or not names or any(type(n) is not str or n not in WORKLOAD_NAMES for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("suite names must be unique registered workload names")
    sources = engine_sources()
    cases = []
    specifications = [(name, *make_workload(name, scale, seed)) for name in names] if custom_workload is None else [("custom", *custom_workload)]
    for index, (name, graph, inputs) in enumerate(specifications):
        placements = baseline_placements(graph, platform) if placement_override is None else [("custom placement", placement_override)]
        exploration = None
        if search:
            exploration = explore_placements(graph, platform, config["model"], arrivals_ns=arrivals_ns,
                                             max_inflight=max_inflight, initial_state=initial_state, candidate_limit=candidate_limit)
            if exploration["best"] is not None and all(placement != exploration["best"]["placement"] for _, placement in placements):
                placements.append(("model search choice", exploration["best"]["placement"]))
        runs, rejected = [], []
        for run_index, (label, placement) in enumerate(placements):
            try:
                result = execute_case(graph, inputs, config, placement, arrivals_ns=arrivals_ns,
                                      max_inflight=max_inflight, initial_state=initial_state)
                records = simulation_records(label, result, config, sources)
                runs.append({"id": f"run_{run_index:03d}", "label": label, "result": result,
                             "records": {metric: value.to_dict() for metric, value in records.items()}})
            except ValueError as exc:
                rejected.append({"label": label, "placement": placement, "reason": str(exc)})
        if not runs:
            raise ValueError(f"{name}: no legal baseline or searched placement: {rejected}")
        comparisons = []
        for left, right in combinations(runs, 2):
            for metric in sorted(set(left["records"]) & set(right["records"])):
                comparison = compare(BenchmarkResult.from_dict(left["records"][metric]), BenchmarkResult.from_dict(right["records"][metric]))
                comparisons.append({"metric": metric, **comparison})
        cases.append({"id": f"case_{index:03d}", "name": name, "workload": workload_to_dict(graph, inputs),
                      "workload_sha256": workload_identity(graph, inputs), "runs": runs,
                      "rejected": rejected, "search": exploration, "comparisons": comparisons})
    return {"schema_version": 1, "kind": "analytical_research_suite", "config": json_value(config),
            "engine_source_sha256": sources, "cases": cases,
            "hardware_performance_claimed": False, "model_validated": False,
            "options": {"scale": scale, "seed": seed, "arrivals_ns": list(arrivals_ns), "max_inflight": max_inflight,
                        "initial_state": initial_state, "search": search, "candidate_limit": candidate_limit}}


def chrome_trace(simulation):
    return {"displayTimeUnit": "ns", "traceEvents": [
        {"name": event["kind"], "cat": "analytical", "ph": "X", "pid": event["request"],
         "tid": "+".join(event["resources"]), "ts": event["start_ns"] / 1000,
         "dur": (event["end_ns"] - event["start_ns"]) / 1000,
         "args": {key: value for key, value in event.items() if key not in ("start_ns", "end_ns")}}
        for event in simulation["timeline"]]}


def write_bundle(suite, destination):
    from .reporting import render_report
    out = Path(destination)
    if out.exists():
        raise ValueError("output path already exists; choose a new experiment directory")
    # Materialize before creating a directory; schema/serialization errors leave no output.
    files = {}
    if type(suite.get("cases")) is not list or not suite["cases"]:
        raise ValueError("experiment must contain nonempty cases")

    def add(path, value):
        files[path] = json.dumps(value, indent=2, allow_nan=False) + "\n"

    def safe_id(value, prefix):
        if type(value) is not str or re.fullmatch(prefix + r"_[0-9]+", value) is None:
            raise ValueError(f"invalid {prefix} artifact identifier")

    for filename, expected in suite["engine_source_sha256"].items():
        if type(filename) is not str or re.fullmatch(r"[a-z_]+\.py", filename) is None:
            raise ValueError("invalid engine source filename")
        data = Path(__file__).with_name(filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("engine sources changed during the experiment; rerun before exporting")
        files[f"engine/pim_foundry/{filename}"] = data.decode()
    files["reproduce.py"] = (
        '"""Replay this bundle with its frozen engine, without writing bytecode."""\n'
        'from pathlib import Path\nimport sys\n'
        'sys.dont_write_bytecode = True\n'
        'root = Path(__file__).resolve().parent\n'
        'sys.path.insert(0, str(root / "engine"))\n'
        'from pim_foundry.__main__ import main\n'
        'raise SystemExit(main(["replay", str(root)]))\n'
    )
    add("config.json", suite["config"])
    summary = {key: value for key, value in suite.items() if key not in ("cases", "config")}
    summary["cases"] = []
    seen_cases = set()
    for case in suite["cases"]:
        case_id = case["id"]
        safe_id(case_id, "case")
        if case_id in seen_cases:
            raise ValueError("duplicate case artifact identifier")
        seen_cases.add(case_id)
        if type(case.get("runs")) is not list or not case["runs"]:
            raise ValueError("experiment cases must contain nonempty runs")
        add(f"workloads/{case_id}.json", case["workload"])
        add(f"comparisons/{case_id}.json", case["comparisons"])
        if case["search"] is not None:
            add(f"search/{case_id}.json", case["search"])
        brief = {key: value for key, value in case.items() if key not in ("runs", "workload", "search", "comparisons")}
        brief["runs"] = []
        seen_runs = set()
        for run in case["runs"]:
            safe_id(run["id"], "run")
            if run["id"] in seen_runs:
                raise ValueError("duplicate run artifact identifier")
            seen_runs.add(run["id"])
            stem = f"{case_id}/{run['id']}"
            add(f"results/{stem}.json", run["result"])
            add(f"traces/{stem}.json", chrome_trace(run["result"]["simulation"]))
            for metric, record in run["records"].items():
                if metric not in ("latency_ns", "throughput_per_s", "energy_j"):
                    raise ValueError("invalid benchmark artifact metric")
                add(f"benchmarks/{stem}/{metric}.json", record)
            brief["runs"].append({"id": run["id"], "label": run["label"],
                                  "placement": run["result"]["functional"]["plan"]["placement"],
                                  **_metrics(run["result"]["simulation"])})
        summary["cases"].append(brief)
    add("summary.json", summary)
    files["report.html"] = render_report(suite)
    manifest = {"schema_version": 1, "kind": "pim-foundry-experiment-bundle", "complete": True,
                "files": {path: hashlib.sha256(data.encode()).hexdigest() for path, data in sorted(files.items())}}
    out.mkdir(parents=True, exist_ok=False)
    for filename, data in files.items():
        path = out / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def verify_bundle(directory):
    root = Path(directory).resolve()
    manifest = read_json(root / "manifest.json")
    if (type(manifest) is not dict or set(manifest) != {"schema_version", "kind", "complete", "files"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["kind"] != "pim-foundry-experiment-bundle" or manifest["complete"] is not True
            or type(manifest["files"]) is not dict or not manifest["files"]):
        raise ValueError("invalid or incomplete experiment manifest")
    for name, expected in manifest["files"].items():
        path = root / name
        if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                or not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file()):
            raise ValueError("manifest paths must be regular files inside the bundle")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path != root / "manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("bundle contains files missing from its manifest")
    return {"integrity_verified": True, "file_count": len(manifest["files"]),
            "hardware_measurements_audited": False}


def replay_bundle(directory):
    from .workloads import workload_from_dict
    integrity = verify_bundle(directory)
    root = Path(directory)
    summary, config = read_json(root / "summary.json"), read_json(root / "config.json")
    if (type(summary) is not dict or type(summary.get("cases")) is not list or not summary["cases"]
            or type(summary.get("options")) is not dict or type(summary.get("engine_source_sha256")) is not dict
            or not summary["engine_source_sha256"] or type(summary.get("schema_version")) is not int
            or summary["schema_version"] != 1 or summary.get("kind") != "analytical_research_suite"):
        raise ValueError("replay summary must contain nonempty cases and options")
    options = summary["options"]
    if not {"arrivals_ns", "max_inflight", "initial_state"}.issubset(options):
        raise ValueError("replay options are incomplete")
    runs = 0
    workload_paths, result_paths = set(), set()
    for case in summary["cases"]:
        if type(case) is not dict or type(case.get("id")) is not str or re.fullmatch(r"case_[0-9]+", case["id"]) is None:
            raise ValueError("invalid case artifact identifier")
        if type(case.get("runs")) is not list or not case["runs"]:
            raise ValueError("replay cases must contain nonempty runs")
        workload_path = f"workloads/{case['id']}.json"
        if workload_path in workload_paths:
            raise ValueError("duplicate case artifact identifier")
        workload_paths.add(workload_path)
        graph, inputs = workload_from_dict(read_json(root / "workloads" / f"{case['id']}.json"))
        for run in case["runs"]:
            if type(run) is not dict or type(run.get("id")) is not str or re.fullmatch(r"run_[0-9]+", run["id"]) is None:
                raise ValueError("invalid run artifact identifier")
            if "placement" not in run:
                raise ValueError("replay run lacks placement")
            result_path = f"results/{case['id']}/{run['id']}.json"
            if result_path in result_paths:
                raise ValueError("duplicate run artifact identifier")
            result_paths.add(result_path)
            expected = read_json(root / "results" / case["id"] / f"{run['id']}.json")
            result = execute_case(graph, inputs, config, run["placement"], arrivals_ns=options["arrivals_ns"],
                                  max_inflight=options["max_inflight"], initial_state=options["initial_state"])
            if result != expected:
                raise ValueError(f"replay differs for {case['id']}/{run['id']}")
            runs += 1
    manifest = read_json(root / "manifest.json")
    if (workload_paths != {name for name in manifest["files"] if name.startswith("workloads/")}
            or result_paths != {name for name in manifest["files"] if name.startswith("results/")}):
        raise ValueError("replay summary must cover all workload and result artifacts")
    return {**integrity, "replayed_runs": runs, "results_reproduced": True,
            "engine_sources_match": engine_sources() == summary["engine_source_sha256"]}

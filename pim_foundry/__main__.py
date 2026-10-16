"""Small dependency-free command line entrypoint."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

from .backend import compile_trace, enumerate_candidates, verify_upstream
from .model import Gemv
from .trace import analyze_trace


def _read_record(path):
    from .benchmark import BenchmarkResult

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return BenchmarkResult.from_dict(json.loads(
        path.read_text(), object_pairs_hook=unique_keys, parse_constant=invalid_constant,
    ))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m pim_foundry",
        description="CIM/PIM heterogeneous analytical simulation, reproducible benchmarks and functional validation.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    research_demo = commands.add_parser("research-demo", help="run selected SRAM/HBM/GPU/NPU logical reference profiles")
    research_demo.add_argument("--out", type=Path, required=True, help="new experiment directory")
    research_demo.add_argument("--devices", default="sram,hbm", help="comma-separated unique sram,hbm,gpu,npu kinds")
    benchmark_check = commands.add_parser("benchmark-check", help="check declared benchmark comparability; does not certify results")
    benchmark_check.add_argument("left", type=Path)
    benchmark_check.add_argument("right", type=Path)
    doctor = commands.add_parser("doctor", help="check reference environment and optional hardware availability")
    doctor.add_argument("--hardware", action="store_true", help="query local NVIDIA driver availability")
    init = commands.add_parser("init", help="create editable synthetic platform and workload configurations")
    init.add_argument("--devices", default="hbm,gpu")
    init.add_argument("--out", type=Path, required=True)
    workload_command = commands.add_parser("workload", help="export a versioned deterministic workload")
    workload_command.add_argument("name", choices=("gemv", "gemm", "mlp", "residual"))
    workload_command.add_argument("--scale", type=int, default=1)
    workload_command.add_argument("--seed", type=int, default=0)
    workload_command.add_argument("--out", type=Path, required=True)
    for name in ("suite", "simulate"):
        command = commands.add_parser(name, help="run analytical experiments and save a self-contained report")
        command.add_argument("--config", type=Path, required=name == "simulate")
        command.add_argument("--devices", default="hbm,gpu", help="synthetic defaults when --config is omitted")
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--requests", type=int, default=4)
        command.add_argument("--arrival-interval-ns", type=int, default=0)
        command.add_argument("--arrivals", type=Path, help="JSON array of arrival times; overrides count and interval")
        command.add_argument("--max-inflight", type=int, default=2)
        command.add_argument("--state", choices=("cold", "warm"), default="cold")
        command.add_argument("--search", action="store_true", help="enumerate placements inside the supplied analytical model")
        command.add_argument("--candidate-limit", type=int, default=256)
        if name == "suite":
            command.add_argument("--workloads", default="gemv,gemm,mlp,residual")
            command.add_argument("--scale", type=int, default=1)
            command.add_argument("--seed", type=int, default=0)
        else:
            command.add_argument("--workload", type=Path, required=True)
            command.add_argument("--placement", type=Path, help="optional JSON mapping of node to device")
    for name in ("verify-bundle", "replay"):
        command = commands.add_parser(name, help="verify exported artifact integrity" if name == "verify-bundle" else "rerun saved workloads and check identical results")
        command.add_argument("directory", type=Path)
    validation = commands.add_parser("validate-model", help="audit external measurement bindings and report held-out model errors")
    validation.add_argument("dataset", type=Path)
    validation.add_argument("--out", type=Path, required=True)
    qualify = commands.add_parser("qualify-latency", help="bind an exact validated latency record to supplied held-out measurement evidence")
    qualify.add_argument("record", type=Path)
    qualify.add_argument("result", type=Path)
    qualify.add_argument("dataset", type=Path)
    qualify.add_argument("--out", type=Path, required=True)
    for name in ("enumerate", "compile"):
        command = commands.add_parser(name)
        command.add_argument("--batch", type=int, default=1)
        command.add_argument("--heads", type=int, default=16)
        command.add_argument("--input", type=int, default=64, dest="input_dim")
        command.add_argument("--output", type=int, default=256, dest="output_dim")
        if name == "compile":
            command.add_argument("--out", type=Path, required=True, help="new output directory")
            command.add_argument("--candidate", type=int, default=None)
    inspect = commands.add_parser("inspect", help="describe an existing trace without simulating it")
    inspect.add_argument("trace", type=Path)
    native = commands.add_parser("dcc-simulate", help="execute the pinned DCC simulator; observations remain unqualified for performance claims")
    native.add_argument("directory", type=Path, help="compiled DCC trace directory")
    native.add_argument("--executable", type=Path, required=True)
    native.add_argument("--library", type=Path, required=True)
    native.add_argument("--out", type=Path, required=True)
    native.add_argument("--timeout", type=float, default=60)
    native_init = commands.add_parser("native-init", help="export a typed mixed SRAM/HBM graph and explicit native/model platform")
    native_init.add_argument("--case", choices=("mixed", "hbm-gpu", "hbm-npu"), default="mixed")
    native_init.add_argument("--out", type=Path, required=True)
    native_init.add_argument("--project-root", type=Path, help="checkout containing external runtimes")
    native_demo = commands.add_parser("native-demo", help="run a typed graph across native SRAM/HBM models and GPU/NPU analytical models")
    native_demo.add_argument("--case", choices=("mixed", "hbm-gpu", "hbm-npu"), default="mixed")
    native_demo.add_argument("--out", type=Path, required=True)
    native_demo.add_argument("--project-root", type=Path, help="checkout containing external runtimes")
    native_demo.add_argument("--asan", action="store_true", help="use the instrumented patched HBM runtime")
    native_run = commands.add_parser("native-run", help="execute a custom typed graph with explicit backend placement")
    for flag in ("config", "graph", "inputs", "placement", "out"):
        native_run.add_argument("--" + flag, type=Path, required=True)
    native_verify = commands.add_parser("native-verify", help="verify all saved native experiment artifacts")
    native_verify.add_argument("directory", type=Path)
    native_compare = commands.add_parser("native-compare", help="compare conditional predictions under identical declared models")
    native_compare.add_argument("left", type=Path)
    native_compare.add_argument("right", type=Path)
    native_compare.add_argument("--metric", choices=("latency", "energy"), default="latency")
    for name in ("app-inspect", "app-run"):
        command = commands.add_parser(name, help="validate or execute a hardware-independent application contract")
        command.add_argument("--application", type=Path, required=True)
        command.add_argument("--experiment", type=Path, required=True)
        if name == "app-run":
            command.add_argument("--out", type=Path, required=True)
    app_verify = commands.add_parser("app-verify", help="verify application assets, execution evidence and engine snapshot")
    app_verify.add_argument("directory", type=Path)
    app_replay = commands.add_parser("app-replay", help="rerun an application and compare scoped observations")
    app_replay.add_argument("directory", type=Path)
    app_replay.add_argument("--out", type=Path, required=True)
    app_compare = commands.add_parser("app-compare", help="compare two verified application bundles under the same task and platform")
    app_compare.add_argument("left", type=Path)
    app_compare.add_argument("right", type=Path)
    app_compare.add_argument("--metric", default="latency")
    protocol_compare = commands.add_parser("compare-protocol", help="compare verified bundles under an explicit resource and evidence protocol")
    protocol_compare.add_argument("left", type=Path)
    protocol_compare.add_argument("right", type=Path)
    protocol_compare.add_argument("--protocol", type=Path, required=True)
    protocol_compare.add_argument("--evidence", type=Path, action="append", default=[])
    protocol_inspect = commands.add_parser("comparison-inspect", help="validate a comparison protocol and resource disclosures")
    protocol_inspect.add_argument("protocol", type=Path)
    tensor_run = commands.add_parser("tensor-run", help="execute an n-dimensional graph through shared device backends")
    for flag in ("graph", "inputs", "platform", "out"):
        tensor_run.add_argument("--" + flag, type=Path, required=True)
    tensor_run.add_argument("--placement", type=Path)
    tensor_schedule = commands.add_parser("tensor-schedule", help="schedule executed graph profiles with shared compute, links, DMA and capacity")
    tensor_schedule.add_argument("profiles", type=Path, nargs="+")
    tensor_schedule.add_argument("--platform", type=Path, required=True)
    tensor_schedule.add_argument("--arrivals", type=Path)
    tensor_schedule.add_argument("--max-inflight", type=int, default=1)
    tensor_schedule.add_argument("--state", choices=("cold", "warm"), default="cold")
    tensor_schedule.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        status = 0
        if args.command in ("compare-protocol", "comparison-inspect"):
            from . import comparison
            if args.command == "comparison-inspect":
                result = comparison.inspect(args.protocol)
            else:
                result = comparison.compare(args.left, args.right, args.protocol, evidence_paths=args.evidence)
                status = 0 if result["comparable"] else 1
        elif args.command in ("tensor-run", "tensor-schedule"):
            from . import tensor_platform, unified
            from .config import read_json
            if args.command == "tensor-run":
                executed = tensor_platform.run(read_json(args.graph), read_json(args.inputs), read_json(args.platform),
                                               placement=read_json(args.placement) if args.placement else None, output=args.out)
                result = {"output": str(args.out.resolve()), "outputs": {k: v.tolist() for k, v in executed["outputs"].items()},
                          "modeled_latency_ns": executed["platform_execution"]["modeled_latency_ns"], "hardware_execution": False}
            else:
                if args.out.exists():
                    raise ValueError("output already exists")
                profiles = [read_json(p) for p in args.profiles]
                result = tensor_platform.schedule([p.get("profile", p) for p in profiles], read_json(args.platform),
                                                  arrivals=read_json(args.arrivals) if args.arrivals else None,
                                                  max_inflight=args.max_inflight, state=args.state)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                unified.save(args.out, result)
        elif args.command.startswith("app-"):
            from . import application
            if args.command == "app-inspect":
                result = application.inspect(args.application, args.experiment)
                status = 0 if result["diagnostics"]["passed"] else 1
            elif args.command == "app-run":
                observation = application.run(args.application, args.experiment, args.out)
                result = {"output": str(args.out.resolve()), "status": observation["status"],
                          "execution_mode": observation["execution_mode"], "metrics": observation["metrics"],
                          "quality": observation["quality"], "report": str((args.out / "report.html").resolve())}
                status = 0 if observation["status"] == "completed" else 1
            elif args.command == "app-verify":
                result = application.verify(args.directory)
            elif args.command == "app-replay":
                result = application.replay(args.directory, args.out)
                status = 0 if result["passed"] else 1
            else:
                result = application.compare(args.left, args.right, args.metric)
                status = 0 if result["same_model_comparable"] else 1
        elif args.command in ("native-init", "native-demo", "native-run", "native-verify", "native-compare"):
            from . import typed, unified, unified_examples
            from .config import read_json
            if args.command == "native-verify":
                result = unified.verify(args.directory)
            elif args.command == "native-compare":
                result = unified.compare(read_json(args.left), read_json(args.right), metric=args.metric)
                status = 0 if result["same_model_comparable"] else 1
            else:
                if args.out.exists():
                    raise ValueError("output already exists")
                if args.command == "native-run":
                    graph = typed.graph_from_dict(read_json(args.graph))
                    inputs, config, placement = read_json(args.inputs), read_json(args.config), read_json(args.placement)
                else:
                    graph, inputs, placement = unified_examples.workload(args.case)
                    kinds = ("sram", "hbm", "gpu", "npu") if args.case == "mixed" else tuple(args.case.split("-"))
                    config = unified_examples.platform(kinds, root=args.project_root)
                if args.command == "native-init":
                    args.out.mkdir(parents=True)
                    for name, value in (("graph.json", typed.graph_to_dict(graph)), ("inputs.json", inputs),
                                        ("platform.json", config), ("placement.json", placement)):
                        unified.save(args.out / name, value)
                    result = {"output": str(args.out.resolve()), "case": args.case,
                              "numeric_policy": "explicit typed tensors and per-node arithmetic", "native_runtimes_required": True}
                else:
                    if getattr(args, "asan", False):
                        for device in config["devices"]:
                            if device["backend"] == "hbm-simd16":
                                device["options"]["variant"] = "asan"
                    experiment = unified.run(graph, inputs, config, placement, args.out)
                    result = {"output": str(args.out.resolve()), "report": str((args.out / "report.html").resolve()),
                              "quality": experiment["quality"], "modeled_latency_ns": experiment["modeled_latency_ns"],
                              "modeled_energy_pj": experiment["modeled_energy_pj"],
                              "hardware_calibrated": False, "nodes": len(experiment["nodes"])}
        elif args.command == "dcc-simulate":
            from .dcc_simulator import run_dcc_simulator
            result = run_dcc_simulator(args.directory, args.executable, args.library, args.out,
                                       timeout_seconds=args.timeout)
            status = 0 if result["status"] == "completed" else 1
        elif args.command in ("suite", "simulate"):
            from .config import example_config, read_json
            from .experiments import build_suite, write_bundle
            from .workloads import workload_from_dict
            if args.out.exists():
                raise ValueError("output path already exists; choose a new experiment directory")
            if not 1 <= args.requests <= 1024 or args.arrival_interval_ns < 0:
                raise ValueError("requests must be 1..1024 and arrival interval nonnegative")
            arrivals = read_json(args.arrivals) if args.arrivals else [i * args.arrival_interval_ns for i in range(args.requests)]
            config = read_json(args.config) if args.config else example_config(tuple(args.devices.split(",")))
            options = dict(arrivals_ns=arrivals, max_inflight=args.max_inflight, initial_state=args.state,
                           search=args.search, candidate_limit=args.candidate_limit)
            if args.command == "suite":
                options.update(names=tuple(args.workloads.split(",")), scale=args.scale, seed=args.seed)
            else:
                options.update(custom_workload=workload_from_dict(read_json(args.workload)),
                               placement_override=read_json(args.placement) if args.placement else None)
            suite = build_suite(config, **options)
            summary = write_bundle(suite, args.out)
            result = {"output": str(args.out.resolve()), "report": str((args.out / "report.html").resolve()),
                      "evidence": "analytical", "model_validated": False, "hardware_performance_claimed": False,
                      "cases": [{"name": case["name"], "runs": len(case["runs"]), "rejected": len(case["rejected"])}
                                for case in summary["cases"]]}
        elif args.command == "init":
            from .config import example_config
            from .workloads import WORKLOAD_NAMES, make_workload, workload_to_dict
            if args.out.exists():
                raise ValueError("output path already exists; choose a new configuration directory")
            config = example_config(tuple(args.devices.split(",")))
            workloads = {name: workload_to_dict(*make_workload(name)) for name in WORKLOAD_NAMES}
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out / "platform.json").write_text(json.dumps(config, indent=2) + "\n")
            for name, value in workloads.items():
                (args.out / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
            result = {"output": str(args.out.resolve()), "configuration": "platform.json", "workloads": list(workloads),
                      "cost_provenance": "synthetic; edit and validate against measurements before hardware claims"}
        elif args.command == "workload":
            from .workloads import make_workload, workload_to_dict
            value = workload_to_dict(*make_workload(args.name, args.scale, args.seed))
            with args.out.open("x") as file:
                file.write(json.dumps(value, indent=2) + "\n")
            result = {"output": str(args.out.resolve())}
        elif args.command in ("verify-bundle", "replay"):
            from .experiments import replay_bundle, verify_bundle
            result = (replay_bundle if args.command == "replay" else verify_bundle)(args.directory)
        elif args.command == "qualify-latency":
            from .calibration import qualify_latency
            if args.out.exists():
                raise ValueError("output file already exists")
            record = qualify_latency(args.record, args.result, args.dataset)
            with args.out.open("x") as file:
                file.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
            result = {"output": str(args.out.resolve()), "calibration_artifact_sha256": record["calibration_artifact_sha256"],
                      "independently_audited": False}
        elif args.command == "validate-model":
            from .calibration import evaluate_validation
            if args.out.exists():
                raise ValueError("output file already exists")
            result = evaluate_validation(args.dataset)
            with args.out.open("x") as file:
                file.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
            status = 0 if result["passed"] else 1
        elif args.command == "research-demo":
            from .benchmark import compare, reference_records
            from .research import platform_demo
            if args.out.exists():
                raise ValueError("output path already exists; choose a new experiment directory")
            experiment = platform_demo(tuple(kind.strip() for kind in args.devices.split(",")))
            records = reference_records(experiment)
            comparisons = [compare(records[left], records[right])
                           for left, right in combinations(records, 2)]
            result = {
                "kind": experiment["kind"], "workload_sha256": experiment["workload_sha256"],
                "output": str(args.out.resolve()),
                "runs": {name: {"outputs": run["outputs"], "validation": run["validation"],
                                "logical_transfer_bytes": run["plan"]["logical_transfer_bytes"],
                                "peak_live_bytes": run["plan"]["peak_live_bytes"]}
                         for name, run in experiment["runs"].items()},
                "placement_search_objective": experiment["placement_search"]["objective"],
                "benchmark_protocol": "pim-foundry/reference-linear/v1",
                "comparison_pairs": len(comparisons),
                "all_comparison_metadata_compatible": all(item["metadata_compatible"] for item in comparisons),
                "performance_ranking_produced": False,
                "independently_audited": False,
                "hardware_performance_claimed": False,
            }
            args.out.mkdir(parents=True)
            (args.out / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
            (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            (args.out / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
            (args.out / "benchmarks").mkdir()
            for name, record in records.items():
                (args.out / "benchmarks" / f"{name}.json").write_text(json.dumps(record.to_dict(), indent=2) + "\n")
        elif args.command == "benchmark-check":
            from .benchmark import compare
            result = compare(_read_record(args.left), _read_record(args.right))
            status = 0 if result["metadata_compatible"] else 1
        elif args.command == "doctor":
            import shutil
            import subprocess
            upstream = {"available": False, "reason": "pinned upstream files are not present in this package"}
            if (Path(__file__).resolve().parents[1] / "upstream/manifest.json").exists():
                upstream = verify_upstream()
            result = {
                "upstream": upstream, "python": sys.version.split()[0],
                "ready_for": "tensor applications, explicit heterogeneous scheduling, paper artifacts and reproducible comparisons",
                "simulator": "analytical resource model; optional native CIMFlow SRAM and patched DCC HBM timing integrations",
                "numerical_execution": "NumPy n-D tensors; native CIM ISA interpreters checked against scalar arithmetic; hardware execution not implied",
            }
            if args.hardware:
                executable = shutil.which("nvidia-smi")
                if executable is None:
                    result["nvidia_driver"] = {"available": False, "reason": "nvidia-smi not found"}
                else:
                    try:
                        probe = subprocess.run([executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                               capture_output=True, text=True, timeout=5)
                        result["nvidia_driver"] = {"available": probe.returncode == 0,
                                                   "diagnostic": (probe.stdout + probe.stderr).strip(),
                                                   "hardware_benchmark_executed": False}
                    except subprocess.TimeoutExpired:
                        result["nvidia_driver"] = {"available": False, "reason": "driver query timed out"}
        elif args.command == "inspect":
            result = analyze_trace(args.trace)
        else:
            workload = Gemv(args.batch, args.heads, args.input_dim, args.output_dim)
            if args.command == "enumerate":
                result = enumerate_candidates(workload)
            else:
                report = compile_trace(workload, args.out, args.candidate)
                result = {
                    "status": report["status"], "output": str(args.out.resolve()),
                    "selected_candidate": report["selected_candidate"],
                    "instruction_count": report["analysis"]["instruction_count"],
                    "trace_sha256": report["trace_sha256"], "validation": report["validation"],
                }
        print(json.dumps(result, indent=2))
        return status
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

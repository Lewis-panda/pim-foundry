#!/usr/bin/env python3
"""Bounded ISCA 2026 artifact integration study; never produces speedup claims.

Run from the project root. Every eligible tiling of all eight upstream
gemv_single shapes and four prespecified inference probes is executed twice
through Foundry and once directly. No timing-dependent candidate selection.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_foundry.backend import REVISION, SOURCE_HASHES, compile_trace, enumerate_candidates
from pim_foundry.dcc_simulator import CONFIG, file_sha256, parse_statistics, run_dcc_simulator
from pim_foundry.model import Gemv, inspect_tiling


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def study(checkout, executable, library, out):
    checkout, executable, library, out = [Path(p).resolve() for p in (checkout, executable, library, out)]
    if out.exists():
        raise ValueError("study output already exists")
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("DCC checkout revision mismatch")
    subprocess.run(["git", "-C", str(checkout), "diff", "--exit-code", "HEAD", "--"], check=True,
                   stdout=subprocess.DEVNULL, timeout=10)
    for name, expected in SOURCE_HASHES.items():
        if file_sha256(checkout / name) != expected:
            raise ValueError("full checkout and generator snapshot differ: " + name)
    # Match the exact public artifact subset, without importing its ML stack.
    test_source = (checkout / "src/test_configs.py").read_text()
    expected_definitions = (
        'gemv_single = generate_test_set([[1, 4], [64, 32], [128], [512, 4096]])',
        'gemv_inference = generate_test_set([[1], [8, 32], [128], [5120]])',
        'gemv_inference += generate_test_set([[1], [11, 42], [128], [6656]])',
        'gemv_inference += generate_test_set([[1], [8, 32], [5120], [384]])',
    )
    if not all(definition in test_source for definition in expected_definitions):
        raise ValueError("reviewed upstream test-set definitions changed")
    out.mkdir(parents=True)
    runtime = out / "runtime"
    runtime.mkdir()
    shutil.copy2(executable, runtime / "ramulator2")
    shutil.copy2(library, runtime / "libramulator.so")
    executable, library = runtime / "ramulator2", runtime / "libramulator.so"
    engine = out / "engine"
    engine.mkdir()
    shutil.copytree(ROOT / "pim_foundry", engine / "pim_foundry", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "upstream", engine / "upstream", ignore=shutil.ignore_patterns("__pycache__"))
    (engine / "scripts").mkdir()
    shutil.copy2(Path(__file__), engine / "scripts/evaluate_dcc.py")
    shutil.copy2(checkout / "src/test_configs.py", out / "upstream-test-configs.py")
    summary = {
        "schema_version": 1, "paper": "DCC, ISCA 2026",
        "paper_url": "https://arxiv.org/abs/2511.15503",
        "venue_url": "https://www.iscaconf.org/isca2026/program/",
        "source_revision": REVISION, "source_tracking_diff_empty": True,
        "upstream_test_configs_sha256": file_sha256(checkout / "src/test_configs.py"),
        "scope": "all 8 artifact gemv_single shapes + 4 prespecified gemv_inference probes; all guard-eligible candidates",
        "selection": "no timing-based sampling; repeat count fixed at two wrapper runs plus one direct run",
        "performance_comparison_eligible": False, "numerical_correctness_verified": False,
        "whole_paper_reproduced": False, "hardware_execution": False,
        "runtime": {"executable_sha256": file_sha256(executable), "library_sha256": file_sha256(library)},
        "workloads": [],
    }
    cases = [("gemv_single", Gemv(b, h, 128, n)) for b in (1, 4) for h in (64, 32) for n in (512, 4096)]
    cases += [("gemv_inference_probe", Gemv(*shape)) for shape in (
        (1, 8, 128, 5120), (1, 11, 128, 6656), (1, 42, 128, 6656), (1, 8, 5120, 384))]
    for group, workload in cases:
        name = f"b{workload.batch}-h{workload.nhead}-k{workload.dhead}-n{workload.seq_len}"
        directory = out / name
        directory.mkdir()
        candidates = enumerate_candidates(workload)
        save(directory / "candidates.json", candidates)
        entry = {
            "name": name, "artifact_test_set": group, "workload": asdict(workload),
            "candidate_count": candidates["candidate_count"], "eligible_count": candidates["eligible_count"],
            "rejection_reason_counts": dict(Counter(reason for c in candidates["candidates"] for reason in c["rejection_reasons"])),
            "runs": [],
        }
        if group == "gemv_single":
            baseline = [[16, 1, workload.batch * workload.nhead // 16], [1, 4, 32], [1, 16, workload.seq_len // 16]]
            entry["upstream_baseline"] = {"tiling": baseline, **inspect_tiling(workload, baseline)}
        for candidate in candidates["candidates"]:
            if not candidate["eligible_for_trace_experiment"]:
                continue
            case = directory / f"candidate-{candidate['id']}"
            case.mkdir()
            compiled = case / "compiled"
            report = compile_trace(workload, compiled, candidate["id"])
            # Independently call the original generator from the full checkout.
            native = case / "direct"
            native.mkdir()
            request = {"action": "compile", "upstream": str(checkout), "output": str(native), "tiling": candidate["tiling"]}
            generator = subprocess.run([sys.executable, "-I", str(ROOT / "pim_foundry/worker.py")],
                                       input=json.dumps(request), text=True, capture_output=True, timeout=60)
            (native / "generator.stdout.log").write_text(generator.stdout)
            (native / "generator.stderr.log").write_text(generator.stderr)
            generator.check_returncode()
            matching_trace = file_sha256(native / "kernel.trace") == report["trace_sha256"]
            # Same configuration bytes, with private relative paths and no shared recorder.
            shutil.copyfile(native / "kernel.yaml", native / "original-upstream.yaml")
            (native / "kernel.yaml").write_text(CONFIG)
            environment = dict(os.environ)
            for key in ("LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"):
                environment.pop(key, None)
            environment.update(LD_LIBRARY_PATH=str(library.parent), LC_ALL="C")
            command = [str(executable), "-f", "kernel.yaml"]
            with (native / "stdout.log").open("wb") as stdout, (native / "stderr.log").open("wb") as stderr:
                process = subprocess.run(command, cwd=native, env=environment, stdout=stdout, stderr=stderr, timeout=60)
            direct_statistics = parse_statistics((native / "stdout.log").read_text()) if process.returncode == 0 else None
            save(native / "execution.json", {"command": command, "cwd": str(native), "returncode": process.returncode,
                 "statistics": direct_statistics, "trace_matches_adapter": matching_trace,
                 "input_sha256": {name: file_sha256(native / name) for name in ("kernel.trace", "kernel.yaml")}})
            observed = [run_dcc_simulator(compiled, executable, library, case / f"wrapped-{i}") for i in (1, 2)]
            direct_matches = all(r["status"] == "completed" and r["statistics"] == direct_statistics for r in observed)
            raw_matches = all(file_sha256(native / "stdout.log") == file_sha256(case / f"wrapped-{i}/stdout.log")
                              and file_sha256(native / "stderr.log") == file_sha256(case / f"wrapped-{i}/stderr.log") for i in (1, 2))
            entry["runs"].append({
                "candidate_id": candidate["id"], "tiling": candidate["tiling"],
                "trace_commands": report["analysis"]["instruction_count"],
                "direct_generator_trace_matches": matching_trace,
                "statuses": [r["status"] for r in observed],
                "direct_returncode": process.returncode,
                "direct_statistics_match": direct_matches, "raw_logs_match": raw_matches,
                "observed_memory_system_cycles": direct_statistics["memory_system_cycles"] if direct_statistics else None,
                "performance_comparison_eligible": False,
            })
        summary["workloads"].append(entry)
        save(out / "summary.json", summary)
        print(name, "candidates", entry["candidate_count"], "eligible", entry["eligible_count"], flush=True)
    all_runs = [r for w in summary["workloads"] for r in w["runs"]]
    summary["totals"] = {
        "workloads": len(cases), "workloads_with_eligible_candidates": sum(bool(w["runs"]) for w in summary["workloads"]),
        "raw_candidates": sum(w["candidate_count"] for w in summary["workloads"]),
        "guard_eligible_candidates": len(all_runs), "native_process_executions": len(all_runs) * 3,
        "all_direct_traces_match": all(r["direct_generator_trace_matches"] for r in all_runs),
        "all_direct_statistics_match": all(r["direct_statistics_match"] for r in all_runs),
        "all_raw_logs_match": all(r["raw_logs_match"] for r in all_runs),
    }
    save(out / "summary.json", summary)
    # Integrity manifest includes engine snapshot, raw logs, original inputs, and binaries.
    manifest = {str(p.relative_to(out)): file_sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    save(out / "manifest.json", {"schema_version": 1, "files": manifest, "trust": "local integrity, not external attestation"})
    print(json.dumps(summary["totals"], indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    study(args.checkout, args.executable, args.library, args.out)

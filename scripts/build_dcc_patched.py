#!/usr/bin/env python3
"""Build the isolated, explicitly modeled DCC fork without installing packages.

Dependencies must already exist at --dependencies (see the pinned build audit).
PIM completion latency is an explicit *assumption*, not an inferred paper value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7ffb80e4f33e357177dc8ab148953c0484e1e322"
PREFIX = "src/Ramulator2/HBMPIM/"
PATCH = ROOT / "patches/dcc/0001-controller-lifetime-and-explicit-completion.patch"
MANIFEST = ROOT / "patches/dcc/source-manifest.json"
PIM_COMMANDS = (
    "MACAB", "MACSB", "MACPB", "WRGB", "MVSB", "MVGB", "SFM", "SETM", "SETH", "DUMMY",
    "LD_OP1", "LD_OP2", "LD_OP3", "ADD_OP1", "ADD_OP2", "ADD_OP3",
    "MUL_OP1", "MUL_OP2", "MUL_OP3", "MAC_OP1", "MAC_OP2", "MAC_OP3",
    "WB_ACC", "WB_RES", "ACC_RESET", "RELU",
)
DEPENDENCIES = {
    "yaml-cpp": "0579ae3d976091d7d664aa9d2527e0d0cff25763",
    "spdlog": "ad0e89cbfb4d0c1ce4d097e134eb7be67baebb36",
    "argparse": "997da9255618311d1fcb0135ce86022729d1f1cb",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_template(completion_cycles):
    if type(completion_cycles) is not int or not 1 <= completion_cycles <= 1000000:
        raise ValueError("completion cycles must be an integer in [1,1000000]")
    # Import only the stable configuration, not the unpatched execution policy.
    sys.path.insert(0, str(ROOT))
    from pim_foundry.dcc_simulator import CONFIG
    mapping = "    pim_completion_cycles:\n" + "".join(
        f"      {name}: {completion_cycles}\n" for name in PIM_COMMANDS)
    return CONFIG.replace("    Scheduler:\n", mapping + "    Scheduler:\n", 1)


def verify_source(source):
    expected = json.loads(MANIFEST.read_text())
    if sha256(PATCH) != expected["patch_sha256"]:
        raise ValueError("patch hash differs from source manifest")
    for entry in expected["files"]:
        path = source / entry["path"]
        if not path.is_file() or sha256(path) != entry["patched_sha256"]:
            raise ValueError("patched source differs: " + entry["path"])
    return expected


def prepare_source(upstream, source):
    if source.exists():
        return verify_source(source)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if revision != REVISION:
        raise ValueError("upstream checkout must be pinned to " + REVISION)
    expected = json.loads(MANIFEST.read_text())
    source.mkdir(parents=True, exist_ok=False)
    for entry in expected["files"]:
        path = source / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        data = subprocess.check_output(["git", "show", f"{REVISION}:{PREFIX}{entry['path']}"], cwd=upstream)
        if hashlib.sha256(data).hexdigest() != entry["upstream_sha256"]:
            raise ValueError("upstream file hash mismatch")
        path.write_bytes(data)
    subprocess.run(["git", "apply", "--no-index", str(PATCH)], cwd=source, check=True)
    return verify_source(source)


def run_logged(command, cwd, prefix, *, timeout=300, environment=None):
    started = time.monotonic()
    with prefix.with_suffix(".stdout.log").open("wb") as stdout, prefix.with_suffix(".stderr.log").open("wb") as stderr:
        try:
            result = subprocess.run(command, cwd=cwd, env=environment, stdout=stdout, stderr=stderr, timeout=timeout)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = None
    record = {"argv": command, "cwd": str(cwd), "returncode": returncode,
              "timeout_seconds": timeout, "wall_seconds": time.monotonic() - started,
              "stdout_sha256": sha256(prefix.with_suffix(".stdout.log")),
              "stderr_sha256": sha256(prefix.with_suffix(".stderr.log"))}
    prefix.with_suffix(".command.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def regression_cases(paper_trace):
    cases = {
        "empty": "", "read": "LD 0x0\n", "write": "ST 0x0\n",
        "barrier-only": "PIM_BARRIER 0x0\n",
        "read-barrier-read": "LD 0x0\nPIM_BARRIER 0x0\nLD 0x20\n",
        "write-barrier-read": "ST 0x0\nPIM_BARRIER 0x0\nLD 0x0\n",
        "mixed-row-column": "".join(f"{op} {addr}\n" for op, addr in
            [("LD", "0x0"), ("ST", "0x20"), ("LD", "0x10000"), ("ST", "0x10020")]*16),
        "pim-writeback-tail": "PIM_WB_ACC 0x0\n",
        "pim-barrier": "PIM_ACC_RESET 0x0\nPIM_LD_OP1 0x0\nPIM_MAC_OP1 0x20\nPIM_WB_ACC 0x40\nPIM_BARRIER 0x0\nLD 0x40\n",
    }
    if paper_trace is not None:
        cases["paper-b1-h32-k128-n512"] = paper_trace.read_text()
    return cases


def run_regressions(output, runtimes, config, paper_trace):
    records = []
    for name, trace in regression_cases(paper_trace).items():
        for variant, runtime in runtimes.items():
            for repeat in range(2):
                case = output / "regressions" / name / f"{variant}-{repeat + 1}"
                case.mkdir(parents=True, exist_ok=False)
                (case / "kernel.trace").write_text(trace)
                (case / "kernel.yaml").write_text(config)
                environment = dict(os.environ)
                for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"):
                    environment.pop(key, None)
                environment.update(LD_LIBRARY_PATH=str(Path(runtime["library"]).parent), LC_ALL="C",
                                   ASAN_OPTIONS="detect_leaks=0:halt_on_error=1",
                                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
                record = run_logged([runtime["executable"], "-f", "kernel.yaml"], case,
                                    case / "native", timeout=60, environment=environment)
                stdout = (case / "native.stdout.log").read_text()
                stderr = (case / "native.stderr.log").read_text()
                cycles = re.findall(r"^  memory_system_cycles: ([0-9]+)$", stdout, re.M)
                record.update(case=name, variant=variant, repeat=repeat + 1,
                              trace_sha256=sha256(case / "kernel.trace"),
                              config_sha256=sha256(case / "kernel.yaml"),
                              cycles=int(cycles[0]) if len(cycles) == 1 else None,
                              sanitizer_diagnostic=("ERROR: AddressSanitizer" in stderr or "runtime error:" in stderr))
                record["passed"] = record["returncode"] == 0 and record["cycles"] is not None and not record["sanitizer_diagnostic"]
                records.append(record)
    for name in regression_cases(paper_trace):
        matched = [record for record in records if record["case"] == name]
        cycles = {record["cycles"] for record in matched}
        for record in matched:
            record["deterministic_across_builds"] = len(cycles) == 1
            record["passed"] &= len(cycles) == 1
    summary = {"schema_version": 1, "cases": records,
               "passed": all(record["passed"] for record in records),
               "leak_detection": False, "arithmetic_checked": False,
               "assumed_timing_not_hardware_calibration": True}
    (output / "regressions.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=ROOT / "external/dcc-isca26")
    parser.add_argument("--source", type=Path, default=ROOT / "external/dcc-foundry-patched")
    parser.add_argument("--dependencies", type=Path, default=ROOT / "runs/paper-dcc-build/deps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pim-completion-cycles", type=int, required=True,
                        help="explicit experimental completion assumption; not a paper value")
    parser.add_argument("--paper-trace", type=Path)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    config = config_template(args.pim_completion_cycles)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = prepare_source(args.upstream.resolve(), args.source.resolve())
    dependency_records = []
    for name, pin in DEPENDENCIES.items():
        source = (args.dependencies / name).resolve()
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True)
        if actual != pin or dirty:
            raise ValueError("dependency differs from clean pinned revision: " + name)
        dependency_records.append({"name": name, "revision": pin, "source": str(source)})
    (args.output / "kernel-template.yaml").write_text(config)
    runtimes = {}
    for variant in ("release", "asan"):
        build = args.output / variant
        configure = ["cmake", "-S", str(args.source.resolve()), "-B", str(build), "-G", "Ninja",
                     "-DCMAKE_BUILD_TYPE=" + ("Release" if variant == "release" else "RelWithDebInfo"),
                     "-DFETCHCONTENT_FULLY_DISCONNECTED=ON", "-DYAML_CPP_BUILD_TESTS=OFF",
                     "-DSPDLOG_BUILD_EXAMPLE=OFF", "-DSPDLOG_BUILD_TESTS=OFF", "-DARGPARSE_BUILD_TESTS=OFF"]
        for dependency in dependency_records:
            configure.append(f"-DFETCHCONTENT_SOURCE_DIR_{dependency['name'].upper()}={dependency['source']}")
        if variant == "asan":
            configure += ["-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer",
                          "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined",
                          "-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address,undefined"]
        for name, command in (("configure", configure), ("build", ["cmake", "--build", str(build), "--target", "ramulator-exe", "--parallel", "4"])):
            result = run_logged(command, ROOT, args.output / f"{variant}-{name}", timeout=600)
            if result["returncode"] != 0:
                raise RuntimeError(f"{variant} {name} failed; see retained logs")
        runtimes[variant] = {"executable": str(build / "ramulator2"), "library": str(build / "libramulator.so"),
                             "executable_sha256": sha256(build / "ramulator2"),
                             "library_sha256": sha256(build / "libramulator.so")}
    runtime = {
        "schema_version": 1, "runtime_name": "dcc-foundry-explicit-completion-v1",
        "source_revision": REVISION, "source_directory": str(args.source.resolve()),
        "source_manifest_sha256": sha256(MANIFEST), "source_files": manifest["files"],
        "patch_path": str(PATCH), "patch_sha256": sha256(PATCH), "dependencies": dependency_records,
        "runtimes": runtimes, "config_template": config,
        "config_template_sha256": hashlib.sha256(config.encode()).hexdigest(),
        "model": {"provenance": "explicit_synthetic_completion_assumption", "tCK_ps": 769,
                  "pim_completion_cycles": {name: args.pim_completion_cycles for name in PIM_COMMANDS},
                  "read_completion": "source nCL+nBL", "write_completion": "source nCWL+nBL+nWR write-recovery endpoint",
                  "refresh_completion": "source nRFC or nRFCSB",
                  "barrier_scope": "addressed channel, all queued requests and modeled completions",
                  "completion_timing_does_not_change_issue_spacing": True,
                  "calibrated": False, "hardware_measured": False,
                  "unresolved": ["source missing WB/RESET/RELU issue constraints", "PIM data dependency hazards beyond explicit barriers", "numerical values not simulated"]},
        "fixes": ["invalidated request iterator", "empty scheduling result dereference",
                  "write/PIM final completion tracking", "barrier waits completion and maintenance queues"],
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "cmake": subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0],
        "build_argv": [sys.executable, *sys.argv],
        "build_script_sha256": sha256(Path(__file__)),
        "max_build_jobs": 4, "system_installs": False,
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    summary = run_regressions(args.output, runtimes, config, args.paper_trace)
    print(json.dumps({"runtime": str(args.output / "runtime.json"), "regressions_passed": summary["passed"],
                      "native_runs": len(summary["cases"])}))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

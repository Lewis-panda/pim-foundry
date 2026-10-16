"""Run the patched DCC timing projection for a Foundry banked instruction file."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess

from .banked import execute_functional
from .config import read_json
from .dcc_simulator import CONFIG, check_request_counts, file_sha256, parse_statistics
from .trace import analyze_trace


def execute(artifact: Path, runtime_descriptor: Path, output: Path, *, timeout=60, variant="release"):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("native timeout must be in (0,60]")
    artifact, output = Path(artifact).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("native output already exists")
    functional = execute_functional(artifact)
    descriptor_sha = file_sha256(Path(runtime_descriptor))
    descriptor = read_json(Path(runtime_descriptor))
    if file_sha256(Path(runtime_descriptor)) != descriptor_sha:
        raise ValueError("runtime descriptor changed while being read")
    if type(descriptor.get("schema_version")) is not int or descriptor.get("schema_version") != 1 or descriptor.get("runtime_name") != "dcc-foundry-explicit-completion-v1":
        raise ValueError("requires the repaired DCC runtime descriptor")
    if variant not in ("release", "asan"):
        raise ValueError("runtime variant must be release or asan")
    runtime = descriptor["runtimes"][variant]
    executable, library = Path(runtime["executable"]), Path(runtime["library"])
    for name, path in (("executable", executable), ("library", library)):
        if file_sha256(path) != runtime[name + "_sha256"]:
            raise ValueError("patched runtime hash mismatch: " + name)
    config = descriptor["config_template"]
    if hashlib.sha256(config.encode()).hexdigest() != descriptor["config_template_sha256"]:
        raise ValueError("patched runtime configuration hash mismatch")
    mapping = re.search(r"    pim_completion_cycles:\n((?:      [A-Z0-9_]+: [0-9]+\n)+)", config)
    if mapping is None or config[:mapping.start()] + config[mapping.end():] != CONFIG:
        raise ValueError("only the pinned HBM clock/geometry/configuration with explicit completion is supported")
    completion = {name: int(value) for name, value in re.findall(r"      ([A-Z0-9_]+): ([0-9]+)", mapping.group(1))}
    if (completion != descriptor["model"].get("pim_completion_cycles")
            or descriptor["model"].get("tCK_ps") != 769):
        raise ValueError("descriptor completion or clock does not match the pinned configuration")
    analysis = analyze_trace(artifact / "kernel.trace")
    output.mkdir(parents=True)
    shutil.copyfile(artifact / "kernel.trace", output / "kernel.trace")
    manifest = read_json(artifact / "manifest.json")
    if (file_sha256(output / "kernel.trace") != manifest["files"]["kernel.trace"]
            or file_sha256(artifact / "program.json") != manifest["files"]["program.json"]):
        raise ValueError("functional artifact changed before native execution")
    (output / "kernel.yaml").write_text(config)
    report = {"schema_version": 1, "status": "failed", "outputs": functional["outputs"],
              "functional_evidence": functional["functional_evidence"],
              "native_arithmetic": False, "hardware_calibrated": False,
              "performance_comparison_eligible": False,
              "model": descriptor["model"], "profile": functional["profile"],
              "runtime_descriptor_sha256": descriptor_sha,
              "runtime": runtime, "variant": variant,
              "command": [str(executable.resolve()), "-f", "kernel.yaml"],
              "timeout_seconds": timeout, "returncode": None,
              "packed_initialization_bytes": functional["packed_initialization_bytes"],
              "reserved_bytes": functional["reserved_bytes"],
              "timing_scope": "resident-input Foundry SIMD16 request projection, per-operation completion barriers; input/output fabric charged by graph runtime",
              "native_timing_ns": None, "native_energy_pj": None}
    report["projection_semantics"] = "source PIM requests can activate all 64 banks of the channel; logical SIMD banks do not assert bank-selective native hardware execution"
    bound_inputs = {str(path): file_sha256(path) for path in (
        output / "kernel.trace", output / "kernel.yaml", Path(runtime_descriptor), artifact / "program.json", artifact / "manifest.json")}
    if bound_inputs[str(Path(runtime_descriptor))] != descriptor_sha:
        raise ValueError("runtime descriptor changed before execution")
    environment = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"):
        environment.pop(key, None)
    environment.update(LD_LIBRARY_PATH=str(library.resolve().parent), LC_ALL="C")
    if variant == "asan":
        environment.update(ASAN_OPTIONS="detect_leaks=0:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(report["command"], cwd=output, env=environment,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                report["returncode"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                report.update(status="timeout", returncode=process.returncode)
        except OSError as exc:
            report["error"] = str(exc)
    if report["returncode"] == 0:
        try:
            stats = parse_statistics((output / "stdout.log").read_text())
            check_request_counts(stats, analysis)
            report.update(status="completed", statistics=stats,
                          native_timing_ns=stats["memory_system_cycles"] * descriptor["model"]["tCK_ps"] / 1000)
        except ValueError as exc:
            report.update(status="invalid_output", error=str(exc))
    try:
        unchanged = (all(file_sha256(path) == runtime[name + "_sha256"]
                         for name, path in (("executable", executable), ("library", library)))
                     and all(file_sha256(Path(path)) == expected for path, expected in bound_inputs.items()))
    except OSError:
        unchanged = False
    if not unchanged:
        report.update(status="invalid_output", error="native runtime or bound inputs changed during execution", native_timing_ns=None)
    report["bound_inputs"] = bound_inputs
    report["files"] = {name: file_sha256(output / name) for name in ("kernel.trace", "kernel.yaml", "stdout.log", "stderr.log")}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["status"] != "completed":
        raise ValueError("native HBM execution failed; retained report at " + str(output / "report.json"))
    return report

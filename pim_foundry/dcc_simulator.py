"""Execute the pinned DCC HBM simulator and retain unqualified observations.

This adapter deliberately cannot promote native cycles to benchmark latency:
the pinned controller has an invalidated-iterator defect, lacks a final
write/PIM completion drain, and does not compute tensor values.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from .backend import REVISION, SOURCE_HASHES
from .config import read_json
from .trace import analyze_trace


COUNTERS = (
    "total_num_read_requests", "total_num_write_requests",
    "total_num_pim_ldop_requests", "total_num_mac_requests",
    "total_num_other_requests", "total_num_pim_mul_requests",
    "total_num_pim_add_requests", "total_num_pim_mac_per_bank_requests",
    "total_num_pim_mac_all_bank_requests", "total_num_pim_mac_same_bank_requests",
    "total_num_pim_write_to_gemv_buffer_requests",
    "total_num_pim_move_to_gemv_buffer_requests", "total_num_pim_softmax_requests",
    "total_num_pim_set_head_requests", "total_num_pim_dummy_requests",
    "total_num_pim_move_to_softmax_buffer_requests", "total_num_pim_set_model_requests",
)

CONFIG = """Frontend:
  impl: PIMLoadStoreTrace
  path: "kernel.trace"
  clock_ratio: 1
  Translation:
    impl: NoTranslation
    max_addr: 2147483648
MemorySystem:
  impl: PIMDRAM
  clock_ratio: 1
  DRAM:
    impl: HBM3-PIM
    org:
      preset: HBM3_8Gb_2R
      channel: 16
    timing:
      preset: HBM3_5.2Gbps
  Controller:
    impl: HBM3-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: AllBankHBM3
  AddrMapper:
    impl: HBM3-PIM
"""

KNOWN_ISSUES = [
    "controller_iterator_lifetime",
    "final_write_and_pim_completion_tail_not_drained",
    "barrier_does_not_wait_for_pending_read_completions",
    "some_pim_opcodes_lack_explicit_timing_constraints",
    "tensor_arithmetic_not_simulated",
    "generator_operand_base_and_lane_semantics_unresolved",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_statistics(stdout: str) -> dict:
    """Parse only a complete, unique MemorySystem section of pinned output."""
    statistics = {}
    in_section = False
    sections = 0
    for line in stdout.splitlines():
        if line == "MemorySystem:":
            sections += 1
            in_section = True
            continue
        if line and not line.startswith(" "):
            in_section = False
        if not in_section or not line.strip() or line == "  impl: PIMDRAM":
            continue
        match = re.fullmatch(r"  ([a-z_]+): ([0-9]+)", line)
        if match is None:
            raise ValueError("malformed MemorySystem statistic")
        name, value = match.groups()
        if name in statistics:
            raise ValueError("duplicate simulator statistic: " + name)
        statistics[name] = int(value)
    if sections != 1 or set(statistics) != {"memory_system_cycles", *COUNTERS}:
        raise ValueError("missing, extra, or duplicate MemorySystem statistics")
    if statistics["memory_system_cycles"] <= 0:
        raise ValueError("memory_system_cycles must be positive")
    return statistics


def check_request_counts(statistics: dict, analysis: dict) -> dict:
    """Check accepted requests, not arithmetic or physical completion."""
    opcodes = analysis["opcode_counts"]
    expected = {name: 0 for name in COUNTERS}
    expected.update({
        "total_num_read_requests": opcodes["LD"],
        "total_num_write_requests": opcodes["ST"],
        "total_num_pim_ldop_requests": opcodes["PIM_LD_OP1"],
        "total_num_mac_requests": opcodes["PIM_MAC_OP1"],
        "total_num_other_requests": opcodes["PIM_ACC_RESET"] + opcodes["PIM_WB_ACC"],
    })
    mismatches = [name for name, value in expected.items() if statistics.get(name) != value]
    if mismatches:
        raise ValueError("accepted request counts differ from trace: " + ", ".join(mismatches))
    return expected


def _normalized_config(text: str) -> str:
    lines = []
    paths = 0
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("  path:"):
            paths += 1
            line = '  path: "kernel.trace"'
        lines.append(line.rstrip())
    if paths != 1:
        raise ValueError("configuration must have one trace path")
    return "\n".join(lines) + "\n"


def run_dcc_simulator(compiled_directory: Path, executable: Path, library: Path,
                      output: Path, *, timeout_seconds: float = 60) -> dict:
    """Save every attempt in a new directory; failures retain their raw logs.

    The caller supplies a trusted native executable/library. Their hashes are
    recorded; this is not a cryptographic attestation of their build provenance.
    Only the reviewed fixed hardware configuration is supported by this adapter.
    """
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 60):
        raise ValueError("timeout_seconds must be finite and in (0,60]")
    compiled_directory = Path(compiled_directory).resolve()
    executable, library, output = (Path(p).resolve() for p in (executable, library, output))
    if output.exists():
        raise ValueError("output path already exists; preserve previous results")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("simulator executable is missing or not executable")
    if not library.is_file() or library.name != "libramulator.so":
        raise ValueError("library must be an existing libramulator.so")
    trace = compiled_directory / "kernel.trace"
    original_report = read_json(compiled_directory / "report.json")
    upstream = original_report.get("upstream") if isinstance(original_report, dict) else None
    if (not isinstance(original_report, dict)
            or type(original_report.get("schema_version")) is not int
            or original_report.get("schema_version") != 1
            or not isinstance(upstream, dict)
            or original_report.get("backend") != "dcc-hbm-gemv-trace"
            or upstream.get("revision") != REVISION
            or upstream.get("sha256") != SOURCE_HASHES):
        raise ValueError("trace report must identify the reviewed DCC source")
    if trace.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("trace exceeds the adapter's 64 MiB input budget")
    analysis = analyze_trace(trace)
    if (file_sha256(trace) != original_report.get("trace_sha256")
            or analysis != original_report.get("analysis")):
        raise ValueError("trace hash or analysis differs from compilation report")
    original_config = (compiled_directory / "kernel.yaml").read_text()
    if _normalized_config(original_config) != CONFIG:
        raise ValueError("only the pinned canonical HBM configuration is supported")
    runtime = {
        "executable": str(executable), "executable_sha256": file_sha256(executable),
        "library": str(library), "library_sha256": file_sha256(library),
        "library_selection": "LD_LIBRARY_PATH set to the supplied library directory",
        "build_provenance_independently_attested": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(trace, output / "kernel.trace")
    shutil.copyfile(compiled_directory / "report.json", output / "compile-report.json")
    (output / "kernel.yaml").write_text(CONFIG)
    command = [str(executable), "-f", "kernel.yaml"]
    environment = dict(os.environ)
    for name in ("LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"):
        environment.pop(name, None)
    environment.update(LD_LIBRARY_PATH=str(library.parent), LC_ALL="C")
    report = {
        "schema_version": 1, "backend": "dcc-hbm-ramulator-native",
        "evidence": "unqualified_native_simulator_observation",
        "source_revision": REVISION, "runtime": runtime,
        "command": command, "working_directory": str(output),
        "timeout_seconds": timeout_seconds, "trace_sha256": file_sha256(output / "kernel.trace"),
        "original_config_sha256": file_sha256(compiled_directory / "kernel.yaml"),
        "config_sha256": file_sha256(output / "kernel.yaml"),
        "configuration_changes": ["trace path relocated", "comments and blank lines removed"],
        "status": "failed", "returncode": None, "statistics": None,
        "accepted_request_counts_verified": False,
        "numerical_correctness_verified": False, "hardware_execution": False,
        "performance_comparison_eligible": False,
        "timing_scope": "memory-system ticks until queues and read completions drain; final write/PIM completion tail excluded",
        "known_issues": list(KNOWN_ISSUES),
    }
    started = time.monotonic()
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(command, cwd=output, env=environment, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            try:
                report["returncode"] = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                report.update(status="timeout", returncode=process.returncode,
                              error="simulator exceeded the execution budget")
        except OSError as exc:
            report["error"] = str(exc)
    report["elapsed_wall_seconds"] = time.monotonic() - started
    if report["returncode"] == 0:
        try:
            raw = (output / "stdout.log").read_text()
            statistics = parse_statistics(raw)
            loaded = re.findall(r"Loaded ([0-9]+) lines\.", raw)
            if loaded != [str(analysis["instruction_count"])]:
                raise ValueError("simulator loaded-line count differs from trace")
            report["expected_request_counts"] = check_request_counts(statistics, analysis)
            report.update(status="completed", statistics=statistics,
                          accepted_request_counts_verified=True)
        except ValueError as exc:
            report.update(status="invalid_output", error=str(exc))
    try:
        unchanged = (file_sha256(executable) == runtime["executable_sha256"]
                     and file_sha256(library) == runtime["library_sha256"]
                     and file_sha256(output / "kernel.trace") == report["trace_sha256"]
                     and file_sha256(output / "kernel.yaml") == report["config_sha256"])
    except OSError:
        unchanged = False
    if not unchanged:
        report.update(status="invalid_output", error="runtime or inputs changed during execution",
                      accepted_request_counts_verified=False)
    report["artifacts"] = {name: file_sha256(output / name) if (output / name).is_file() else None
                           for name in ("kernel.trace", "kernel.yaml", "compile-report.json", "stdout.log", "stderr.log")}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report

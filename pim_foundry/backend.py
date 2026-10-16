"""Pinned DCC adapter; trace generation is explicitly simulation-only."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .model import Gemv, inspect_tiling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVISION = "7ffb80e4f33e357177dc8ab148953c0484e1e322"
SOURCE_HASHES = {
    "src/generate_configs.py": "98f85972bc9ca1bd0e73e424fc6f7f85fb2672faa72f37017736775f4ce003a6",
    "src/hbmpim_trace_gen/gen_trace_HBMPIM_GEMV.py": "d4d37c8e85ee8175c7685c60e5b36dc05f6153aa2e7608c4ce1b67d5bc44bdfa",
}


def verify_upstream(root: Path = PROJECT_ROOT) -> dict:
    root = Path(root)
    manifest = json.loads((root / "upstream/manifest.json").read_text())
    if manifest.get("revision") != REVISION:
        raise ValueError("upstream revision differs from the reviewed revision")
    for name, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((root / "upstream/DCC" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"upstream source hash mismatch: {name}")
    return {"repository": manifest["repository"], "revision": REVISION, "sha256": dict(SOURCE_HASHES)}


def _worker(action: str, output: Path, **kwargs) -> dict:
    verify_upstream()
    request = {"action": action, "upstream": str(PROJECT_ROOT / "upstream/DCC"),
               "output": str(output), **kwargs}
    worker_path = Path(__file__).with_name("worker.py")
    try:
        completed = subprocess.run(
            [sys.executable, "-I", str(worker_path)], input=json.dumps(request),
            text=True, capture_output=True, timeout=60, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("upstream worker exceeded its 60-second budget") from exc
    if completed.returncode:
        raise ValueError(f"upstream worker failed: {completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def enumerate_candidates(workload: Gemv) -> dict:
    provenance = verify_upstream()
    with tempfile.TemporaryDirectory(prefix="pim-foundry-enumerate-") as tmp:
        config = _worker("enumerate", Path(tmp), workload=asdict(workload))
    candidates = []
    seen = set()
    for tiling in config["tiling"]:
        key = json.dumps(tiling)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"id": len(candidates), "tiling": tiling,
                           **inspect_tiling(workload, tiling)})
    return {
        "schema_version": 1, "backend": "dcc-hbm-gemv-trace",
        "workload": asdict(workload), "upstream": provenance,
        "candidate_count": len(candidates),
        "eligible_count": sum(c["eligible_for_trace_experiment"] for c in candidates),
        "candidates": candidates,
        "selection_policy": "first eligible upstream candidate; no performance tuning",
    }


def compile_trace(workload: Gemv, output: Path, candidate_id: int | None = None) -> dict:
    from .trace import analyze_trace

    config = enumerate_candidates(workload)
    if candidate_id is not None:
        if type(candidate_id) is not int or not 0 <= candidate_id < len(config["candidates"]):
            raise ValueError("candidate ID is out of range")
        selected = config["candidates"][candidate_id]
        if not selected["eligible_for_trace_experiment"]:
            raise ValueError("candidate rejected: " + "; ".join(selected["rejection_reasons"]))
    else:
        selected = next((c for c in config["candidates"] if c["eligible_for_trace_experiment"]), None)
        if selected is None:
            raise ValueError("no candidate passes conservative guards; use enumerate to inspect reasons")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output path already exists; choose a new directory to preserve existing results")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pim-foundry-build-", dir=output.parent) as tmp:
        staging = Path(tmp)
        _worker("compile", staging, tiling=selected["tiling"])
        analysis = analyze_trace(staging / "kernel.trace")
        if analysis["instruction_count"] != selected["predicted_commands"]:
            raise ValueError("generated command count differs from the inspected preflight model")
        trace_hash = hashlib.sha256((staging / "kernel.trace").read_bytes()).hexdigest()
        # Quote YAML path scalars; remove the optional upstream recorder whose
        # hardcoded output path would collide across independent experiments.
        yaml_path = staging / "kernel.yaml"
        cleaned = []
        skip_plugin = False
        for line in yaml_path.read_text().splitlines():
            if line == "    plugins:":
                skip_plugin = True
                continue
            if skip_plugin:
                if not line.strip() or line.startswith("    "):
                    continue
                skip_plugin = False
            if line.startswith("  path:"):
                line = "  path: " + json.dumps(str(output / "kernel.trace"))
            cleaned.append(line)
        yaml_path.write_text("\n".join(cleaned) + "\n")
        report = {
            "schema_version": 1, "status": "trace_generated_not_numerically_verified",
            "backend": config["backend"], "workload": config["workload"],
            "upstream": config["upstream"], "selected_candidate": selected,
            "selection_policy": config["selection_policy"],
            "trace_sha256": trace_hash, "analysis": analysis,
            "validation": {"source_hashes": True, "conservative_shape_guards": True,
                           "trace_syntax_and_address_bounds": True,
                           "command_count_consistency": True,
                           "numerical_correctness": False,
                           "simulator_execution": False, "hardware_execution": False},
            "limitations": [
                "Trace contains commands and addresses, not tensor values or computed outputs.",
                "No latency, energy, speedup, or optimality claim; command counts are not timing.",
                "Upstream operand-base ordering remains unresolved; inspect docs/upstream-audit.md.",
                "Guards exclude known reduction and local-head hazards but do not prove hardware legality.",
                "Only one fixed DCC HBM GEMV trace backend; no general tensor frontend or driver.",
                "YAML retains upstream NoTranslation.max_addr=2 GiB versus a 16 GiB trace address space; simulator semantics unverified.",
            ],
            "yaml_adapter_changes": ["absolute trace path quoted", "shared optional recorder plugin removed"],
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (staging / "candidates.json").write_text(json.dumps(config, indent=2) + "\n")
        output.mkdir()
        for artifact in staging.iterdir():
            os.replace(artifact, output / artifact.name)
    return report

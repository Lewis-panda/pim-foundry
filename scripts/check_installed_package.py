#!/usr/bin/env python3
"""Smoke-test the installed wheel outside the source checkout (no path injection)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import pim_foundry
    from pim_foundry import application
    root = args.project.resolve()
    source = Path(pim_foundry.__file__).resolve()
    if not source.is_relative_to(args.installed.resolve()):
        raise ValueError("package smoke accidentally imported the source checkout")
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for name, manifest, experiment in (
        ("mixed", root / "runs/tensor-platform-v08-release/fixtures/application.json", root / "runs/tensor-platform-v08-release/sram-hbm-gpu.json"),
        ("onnx", root / "runs/onnx-tensor-v08/fixture/application.json", root / "runs/onnx-tensor-v08/fixture/experiment.json"),
        ("native_transformer", root / "runs/transformer-native-hbm-v08-release/application.json", root / "runs/transformer-native-hbm-v08-release/experiment.json")):
        result = application.run(manifest, experiment, args.out / name)
        verified = application.verify(args.out / name)
        replay = application.replay(args.out / name, args.out / (name + "-replay"))
        results.append({"case": name, "status": result["status"], "verified": verified["passed"], "replay": replay["passed"]})
    doctor = subprocess.run([sys.executable, "-m", "pim_foundry", "doctor"], check=True, capture_output=True, text=True, cwd="/tmp")
    report = {"passed": all(r["status"] == "completed" and r["verified"] and r["replay"] for r in results),
              "version": pim_foundry.__version__, "imported_from": str(source),
              "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
              "cases": results, "doctor": json.loads(doctor.stdout),
              "scope": "installed wheel, existing pinned optional dependencies, native runtimes and bound assets"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

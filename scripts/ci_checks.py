#!/usr/bin/env python3
"""Run every discovered test and report explicit optional-integration skips.

The core profile verifies the zero-dependency installation. The onnx profile
requires both numerical extras and the independent ONNX reference evaluator.
Neither profile downloads author code, model weights, or native simulators.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SKIP_REASONS = {
    "optional author DCC generator sources are not installed",
    "optional native DCC build is not present",
    "explicit optional native integration",
    "native simulator path not configured",
    "native HBM opt-in",
    "native SRAM opt-in",
}
DEPENDENCY_SKIP_REASONS = {
    "optional NumPy required",
    "optional NumPy dependency",
    "NumPy optional dependency unavailable",
    "ONNX/NumPy optional dependencies unavailable",
    "ONNX independent reference unavailable",
    "install optional ONNX dependencies to test model interoperability",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "onnx"), required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "onnx")}
    if args.profile == "onnx" and not all(dependencies.values()):
        parser.error("onnx profile requires installation with .[onnx]")
    if args.profile == "core" and any(dependencies.values()):
        parser.error("core profile requires an isolated environment without NumPy or ONNX")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    discovered = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    allowed_skips = NATIVE_SKIP_REASONS | (DEPENDENCY_SKIP_REASONS if args.profile == "core" else set())
    skips = [{"test": test.id(), "reason": reason} for test, reason in result.skipped]
    unexpected_skips = [row for row in skips if row["reason"] not in allowed_skips]
    complete = result.testsRun == discovered and discovered > 0
    passed = result.wasSuccessful() and complete and not unexpected_skips
    report = {
        "schema": "pim-foundry.ci-tests.v1",
        "profile": args.profile,
        "python": sys.version.split()[0],
        "optional_dependencies": dependencies,
        "discovered": discovered,
        "tests_run_including_skips": result.testsRun,
        "passed_tests": result.testsRun - len(result.skipped) - len(result.failures) - len(result.errors),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": skips,
        "unexpected_skips": unexpected_skips,
        "complete_discovery": complete,
        "passed": passed,
        "evidence_scope": "software regression tests; not hardware or full-paper validation",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

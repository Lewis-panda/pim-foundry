#!/usr/bin/env python3
"""Recheck a concrete release's sealed studies, tests and install evidence."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_foundry import application, __version__


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensor-study", type=Path, required=True)
    parser.add_argument("--transformer-study", type=Path, required=True)
    parser.add_argument("--transformer-platform-study", type=Path, required=True)
    parser.add_argument("--native-transformer-study", type=Path, required=True)
    parser.add_argument("--onnx-study", type=Path, required=True)
    parser.add_argument("--comparison-study", type=Path, required=True)
    parser.add_argument("--transformer-comparison", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--package-check", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    checks, bundles = {}, []
    log = args.test_log.read_text()
    count = re.search(r"Ran (\d+) tests", log)
    checks["full_regression_without_skips"] = bool(count and re.search(r"\nOK\s*$", log))
    package = read(args.package_check)
    checks["clean_wheel_install"] = package.get("passed") is True
    checks["wheel_matches_installed_validation"] = package.get("wheel_sha256") == sha(args.wheel)
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "pim_foundry").glob("*.py"))}
    with zipfile.ZipFile(args.wheel) as archive:
        wheel_hashes = {name: hashlib.sha256(archive.read(name)).hexdigest()
                        for name in archive.namelist() if name.startswith("pim_foundry/") and name.endswith(".py")}
    checks["wheel_exactly_matches_current_source"] = wheel_hashes == source_hashes
    for study in (args.tensor_study, args.transformer_study, args.transformer_platform_study,
                  args.native_transformer_study, args.onnx_study):
        found = sorted(study.glob("*/manifest.json"))
        if not found:
            raise ValueError("no sealed application bundles in " + str(study))
        for manifest in found:
            if read(manifest).get("schema") != "pim-foundry.application-bundle.v1":
                continue
            verified = application.verify(manifest.parent)
            result = read(manifest.parent / "result.json")
            expected = "quality_failed" if manifest.parent.name == "wrong-reference" else "completed"
            passed = verified["passed"] and result["status"] == expected
            checks["bundle:" + str(manifest.parent)] = passed
            bundles.append({"path": str(manifest.parent.resolve()), "status": result["status"],
                            "manifest_sha256": sha(manifest), "passed": passed,
                            "quality": {k: v["status"] for k, v in result["quality"].items()}})
    tensor = read(args.tensor_study / "summary.json")
    checks["four_platforms_and_wrong_reference"] = tensor.get("passed") is True and tensor.get("wrong_reference_rejected") is True
    for label, path in (("complete_distilgpt2", args.transformer_study),
                        ("distilgpt2_platform", args.transformer_platform_study),
                        ("complete_native_hbm_transformer", args.native_transformer_study)):
        summary = read(path / "summary.json")
        result = summary.get("application", summary.get("result"))
        checks[label] = bool(result and result["status"] == "completed" and all(q["status"] == "passed" for q in result["quality"].values()))
        checks[label + "_replay"] = summary["replay"]["passed"] is True
    native = read(args.native_transformer_study / "summary.json")
    checks["native_baseline_replay"] = native["baseline_replay"]["passed"] is True
    checks["native_runtime_bound"] = bool(native["result"]["execution_identity"].get("runtime_bindings"))
    comparison = read(args.comparison_study / "summary.json")
    checks["comparison_positive_and_negative_controls"] = comparison.get("passed") is True
    native_comparison = read(args.transformer_comparison / "report.json")
    checks["full_application_comparison_qualified"] = native_comparison.get("qualified") is True and native_comparison.get("comparable") is True
    checks["hardware_claim_withheld"] = native_comparison.get("hardware_performance_comparison_eligible") is False
    checks["onnx_replay"] = read(args.onnx_study / "summary.json")["replay"]["passed"] is True
    negative = ROOT / "runs/transformer-quality-negative-v08/run"
    checks["wrong_transformer_token_reference_rejected"] = application.verify(negative)["passed"] and read(negative / "result.json")["status"] == "quality_failed"
    evidence = {str(p.resolve()): sha(p) for p in [args.test_log, args.package_check, args.wheel,
                args.tensor_study / "summary.json", args.transformer_study / "summary.json",
                args.transformer_platform_study / "summary.json", args.native_transformer_study / "summary.json",
                args.onnx_study / "summary.json", args.comparison_study / "summary.json", args.transformer_comparison / "report.json"]}
    report = {"schema": "pim-foundry.release-acceptance.v1", "version": __version__,
              "passed": all(checks.values()), "tests": int(count.group(1)) if count else None,
              "checks": checks, "bundles": bundles, "evidence": evidence,
              "source_sha256": source_hashes,
              "scope": "software acceptance for declared operator, precision, backend and comparison contracts",
              "evidence_limits": ["GPU/NPU performance is analytical; physical device calibration is not established",
                                  "DistilGPT2 quality is a fixed 128-target WikiText test fixture, not the full test benchmark",
                                  "tiny GPT2 native integration is a functional fixture",
                                  "IANUS author artifact was not obtained; source audit and third-party smoke remain distinct",
                                  "no community certification is implied"]}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = "".join("<tr><td>" + html.escape(name) + "</td><td>" + ("通過" if passed else "失敗") + "</td></tr>" for name, passed in checks.items())
    links = "".join("<li><a href='" + html.escape(str(Path(item["path"]) / "report.html"), quote=True) + "'>" + html.escape(Path(item["path"]).parent.name+" / "+Path(item["path"]).name) + "</a> — " + item["status"] + "</li>" for item in bundles)
    page = """<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>PIM Foundry release acceptance</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#183047;background:#f5f7fa}table{border-collapse:collapse;width:100%;background:white}td{padding:9px 12px;border-bottom:1px solid #dde4eb;overflow-wrap:anywhere}a{color:#126191}code{overflow-wrap:anywhere}</style>"""
    page += "<h1>PIM Foundry " + __version__ + " 軟體驗收</h1><p>" + ("通過" if report["passed"] else "未通過") + "；" + str(report["tests"]) + " 項回歸測試，" + str(len(bundles)) + " 份封存 application bundle。刻意錯誤的 reference 應得到 quality_failed。</p>"
    page += "<p>完整 DistilGPT-2、原生 HBM 的完整小型 GPT-2、SRAM／HBM 混合張量、ONNX、排程及比較契約各有執行與重播證據。模擬成本保留其來源與限制。</p><table>"+rows+"</table><h2>可查核報告</h2><ul>"+links+"</ul>"
    page += "<h2>證據邊界</h2><ul>"+"".join("<li>"+html.escape(v)+"</li>" for v in report["evidence_limits"])+"</ul><p><a href='report.json'>完整檢查與來源雜湊</a></p></html>"
    (args.out / "report.html").write_text(page)
    print(json.dumps({"passed": report["passed"], "tests": report["tests"], "bundles": len(bundles), "report": str((args.out / "report.html").resolve())}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

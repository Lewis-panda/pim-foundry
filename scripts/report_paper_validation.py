#!/usr/bin/env python3
"""Collect verified application bundles without pooling incompatible metrics."""

import argparse
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_foundry import application, unified
from pim_foundry.config import read_json
from pim_foundry.dcc_simulator import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases, studies, seen = [], [], set()
    for study in args.study:
        study = study.resolve()
        if not study.is_dir():
            raise ValueError("study directory does not exist: " + str(study))
        study_record = {"path": str(study), "files": {}}
        for name in ("summary.json", "report.html"):
            path = study / name
            if path.is_file():
                study_record["files"][name] = {"path": str(path), "sha256": file_sha256(path)}
        if "summary.json" in study_record["files"]:
            study_summary = read_json(study / "summary.json")
            study_record["validation_status"] = study_summary.get("validation_status", study_summary.get("status", "see_study_checks"))
        studies.append(study_record)
        for path in sorted(study.rglob("application.json")):
            directory = path.parent
            if directory in seen or not (directory / "manifest.json").is_file():
                continue
            manifest = read_json(directory / "manifest.json")
            if manifest.get("schema") != "pim-foundry.application-bundle.v1":
                continue
            integrity = application.verify(directory)
            result = read_json(directory / "result.json")
            app = read_json(path)
            cases.append({"directory": str(directory), "name": str(directory.relative_to(study)),
                          "study": str(study), "application_id": result["application_id"],
                          "adapter": result["adapter"], "status": result["status"],
                          "execution_mode": result["execution_mode"], "scenario": app["scenario"],
                          "metrics": result["metrics"], "quality": result["quality"],
                          "task_sha256": result["task_sha256"], "execution_sha256": result["execution_sha256"],
                          "platform_sha256": result["platform_sha256"], "cost_contract_sha256": result["cost_contract_sha256"],
                          "manifest_sha256": file_sha256(directory / "manifest.json"), "integrity": integrity,
                          "quality_preserving_comparison_eligible": result["quality_preserving_comparison_eligible"],
                          "hardware_performance_comparison_eligible": result["hardware_performance_comparison_eligible"]})
            seen.add(directory)
    if not cases:
        raise ValueError("no application bundles found")
    summary = {"schema": "pim-foundry.paper-validation-index.v1", "studies": studies, "cases": cases,
               "verified_application_bundles": len(cases), "adapters": sorted({c["adapter"] for c in cases}),
               "quality_failed_bundles": sum(c["status"] == "quality_failed" for c in cases),
               "aggregate_speedup": None, "aggregate_speedup_reason": "different_tasks_platforms_and_metric_boundaries",
               "hardware_validation_claimed": False}
    unified.save(output / "summary.json", summary)
    esc = lambda value: html.escape(str(value))
    study_links = []
    for study in studies:
        target = Path(study["path"])
        links = []
        for name in ("report.html", "summary.json"):
            if name in study["files"]:
                links.append("<a href='" + esc(os.path.relpath(target / name, output)) + "'>" + esc(name) + "</a>")
        if links:
            study_links.append("<li>" + esc(target.name) + ": " + " · ".join(links)
                               + " — " + esc(study.get("validation_status", "see_study_checks")) + "</li>")
    rows = []
    for case in cases:
        report = os.path.relpath(Path(case["directory"]) / "report.html", output)
        metrics = "<br>".join(esc(name) + ": " + esc(metric["value"]) + " " + esc(metric["unit"])
                               + "<small>" + esc(metric["scope"]) + " · " + esc(metric["aggregation"]) + "</small>"
                               for name, metric in case["metrics"].items())
        rows.append("<tr><td><a href='" + esc(report) + "'>" + esc(case["name"]) + "</a><small>"
                    + esc(case["adapter"]) + "</small></td><td>" + metrics + "</td><td>"
                    + esc(case["quality"]["numerical"]["status"]) + "</td><td>"
                    + esc(case["quality"]["task"]["status"]) + "</td><td>" + esc(case["status"]) + "</td></tr>")
    (output / "report.html").write_text("""<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Paper application validation</title>
<style>body{font:16px/1.7 system-ui;max-width:1280px;margin:40px auto;padding:0 24px;color:#18303a;background:#f6f8fa}table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;border-bottom:1px solid #d9e0e5;text-align:left;vertical-align:top}small{display:block;font-size:12px;color:#52616c;max-width:520px}a{color:#126473}.overflow{overflow:auto}</style>
<h1>論文 application 驗證</h1><p>每列都先通過結果包完整性檢查，再呈現實際 metrics、計量範圍與品質觀測。不同 paper 的數字沒有合併成總分或總加速比。</p>
<p>結果包數量包含來源快照重跑，不代表同樣數量的不同工作負載。各研究的直接執行核對、反例與公平性檢查請見下列詳細報告。</p>
<p>completed 表示該次執行完成；檔案完整性不等於原生模型準確或可重現。研究報告若標記 reproducibility_failed，代表沒有通過精確重現驗收，不能因個別執行完成就當成整項研究驗證通過。</p>
<ul>""" + "".join(study_links) + """</ul>
<p>numerical 表示對所宣告 reference 的核對；task 表示獨立任務語意／品質的觀測。兩者可以分別通過與失敗。quality_failed 是應保留的反例，不能作為有效效能比較。</p>
<div class="overflow"><table><tr><th>實驗</th><th>指標與範圍</th><th>Numerical</th><th>Task</th><th>結果</th></tr>"""
        + "".join(rows) + "</table></div><p><a href='summary.json'>完整索引與來源 hash</a> · "
        + "<a href='../../docs/paper-validation-protocol.md'>驗證方法與後續候選</a></p>"
        + "<p>已驗證結果包：" + str(len(cases)) + "；其中品質失敗：" + str(summary["quality_failed_bundles"])
        + "。模擬器執行、任務品質與硬體量測保持不同證據範圍。</p></html>")
    print(json.dumps({"output": str(output), "bundles": len(cases), "quality_failed": summary["quality_failed_bundles"]}))


if __name__ == "__main__":
    main()

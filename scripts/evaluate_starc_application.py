#!/usr/bin/env python3
"""Run STARC/Full through application manifests and verify frozen replays."""

import argparse
import copy
import html
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_foundry import application, starc_adapter, unified
from pim_foundry.config import read_json
from pim_foundry.dcc_simulator import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "runs/starc-build/runtime-safe.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--direct-full", type=Path)
    parser.add_argument("--direct-starc", type=Path)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = starc_adapter.validate_runtime(args.runtime)
    source = Path(runtime["source_root"])
    inputs = output / "inputs"
    inputs.mkdir()
    manifest = {"schema": application.APP_SCHEMA, "id": "starc-mistral-artifact-model",
                "adapter": starc_adapter.ADAPTER,
                "assets": {"model_definition": application.asset(source / starc_adapter.MODEL_DEFINITION, "model", "python_source")},
                "parameters": {"model": "Mistral-7B"},
                "scenario": {"kind": "llm_prefill_decode_model", "batch_size": args.batch_size,
                             "prompt_tokens": args.prompt_tokens, "generated_tokens": args.generated_tokens,
                             "state": {"kind": "kv_cache", "representation": "shape_and_budget_only",
                                       "lifetime": "one_sequence", "reset": "before_each_run"},
                             "request_schedule": "fixed_batch", "scope": "artifact_declared_system_model"},
                "quality": {"kind": "not_evaluated", "reason": "CPU artifact simulation does not execute model weights or quality datasets"}}
    application_path = inputs / "application.json"
    unified.save(application_path, manifest)
    full_paper = copy.deepcopy(manifest)
    full_paper["id"] = "starc-readme-e4-full-profile"
    full_paper["scenario"].update(prompt_tokens=2048, generated_tokens=32000, batch_size=16)
    unified.save(inputs / "application-full-paper-profile.json", full_paper)
    summary = {"schema_version": 1, "paper": {"title": "STARC", "venue": "ASPLOS 2026",
               "artifact": "https://github.com/EPIC-RPI/STARC", "source_commit": runtime["source_commit"]},
               "executed_scenario": manifest["scenario"], "cases": [], "comparisons": {},
               "full_paper_profile": {"application": "inputs/application-full-paper-profile.json",
                                      "executed": manifest["scenario"] == full_paper["scenario"]},
               "task_quality_evaluated": False, "hardware_execution": False,
               "limitations": ["Author Mistral shape model declares GQA=4 but does not apply KV-head sharing.",
                               "Author-supplied KV budget is reused; no LongBench/RULER/PG-19 evaluation was rerun.",
                               "This is original-artifact system modeling, not a common-IR Transformer port."]}
    if not summary["full_paper_profile"]["executed"]:
        summary["limitations"].append("The selected decode window is not the complete 32000-token E4 example.")
    for method in ("full", "starc"):
        variant = {"runtime_descriptor": str(args.runtime.resolve()), "method": method, "timeout_seconds": args.timeout}
        if method == "starc":
            budget = source / "simulator_starc/kv_budget_Mistral_STARC.txt"
            variant["kv_budget_table"] = {"path": str(budget), "sha256": file_sha256(budget)}
        experiment = {"schema": application.EXPERIMENT_SCHEMA, "platform": dict(starc_adapter.DEFAULT_PLATFORM),
                      "mapping": {"strategy": "artifact_native"}, "variant": variant}
        experiment_path = inputs / (method + ".json")
        unified.save(experiment_path, experiment)
        result = application.run(application_path, experiment_path, output / method)
        verified = application.verify(output / method)
        process = subprocess.run([sys.executable, str(output / method / "reproduce.py"), str(output / (method + "-replay"))],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=args.timeout + 90, check=True)
        replay = json.loads(process.stdout)
        if replay["passed"] is not True:
            raise ValueError("frozen replay mismatch: " + method)
        application.verify(output / (method + "-replay"))
        native_evidence = {name: starc_adapter.verify_native_evidence(output / name / "execution/artifact")
                           for name in (method, method + "-replay")}
        direct = args.direct_full if method == "full" else args.direct_starc
        parity = {"tested": direct is not None}
        if direct:
            parity["checks"] = {name: file_sha256(output / method / "execution/artifact" / name) == file_sha256(direct / name)
                                for name in ("output.csv", "ramulator.out")}
            parity["passed"] = all(parity["checks"].values())
            parity["direct_artifacts"] = {name: {"path": str((direct / name).resolve()), "sha256": file_sha256(direct / name)}
                                          for name in ("output.csv", "ramulator.out")}
            if not parity["passed"]:
                raise ValueError("direct artifact differs from Foundry application: " + method)
        summary["cases"].append({"name": method, "metrics": result["metrics"], "quality": result["quality"],
                                 "task_sha256": result["task_sha256"], "execution_sha256": result["execution_sha256"],
                                 "integrity": verified, "frozen_replay": replay, "direct_parity": parity,
                                 "native_evidence": native_evidence,
                                 "native_calls": result["diagnostics"]["native_calls"]})
        print(method + ": completed, integrity checked, frozen replay matched", flush=True)
    for metric in ("decode_latency", "decode_energy", "prefill_latency"):
        summary["comparisons"][metric] = application.compare(output / "full", output / "starc", metric)
        if not summary["comparisons"][metric]["same_model_comparable"]:
            raise ValueError("same-platform variants were not comparable")
    full_result = read_json(output / "full/result.json")
    changed = copy.deepcopy(read_json(output / "starc/result.json"))
    changed["task_sha256"] = "0" * 64
    summary["different_task_rejected"] = not application.compare_results(full_result, changed, "decode_latency")["same_model_comparable"]
    summary["all_frozen_replays_passed"] = all(case["frozen_replay"]["passed"] for case in summary["cases"])
    summary["native_calls_including_replays"] = sum(case["native_calls"] * 2 for case in summary["cases"])
    unified.save(output / "summary.json", summary)
    rows = "".join("<tr><td><a href='" + case["name"] + "/report.html'>" + case["name"] + "</a></td>"
                   + "".join("<td>" + html.escape(str(case["metrics"][metric]["value"])) + "</td>"
                             for metric in ("prefill_latency", "decode_latency", "decode_energy"))
                   + "<td>" + str(case["frozen_replay"]["passed"]) + "</td></tr>" for case in summary["cases"])
    (output / "report.html").write_text("""<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><title>STARC application validation</title>
<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#18303a}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style>
<h1>STARC：同一個 application，兩個實驗變體</h1><p>Full 與 STARC 使用相同 application manifest、模型定義、請求形狀與平台。method 和 KV budget 放在 experiment；每次使用空白 cache。原始 artifact 的 GPU 分析模型與 PIM simulator 確實執行，沒有載入 LLM 權重或產生 token 答案。</p>
<table><tr><th>方法</th><th>Prefill batch ms</th><th>平均 decode batch-token ms</th><th>平均 decode batch-token nJ</th><th>快照重跑</th></tr>""" + rows
        + "</table><p>decode 的平均取自 generated_tokens − 1 個步驟；不是每個使用者 token 的平均，也不包含 prefill。模型名稱指作者的 shape model，其 GQA sharing 未實作。</p><pre>"
        + html.escape(json.dumps({"scenario": summary["executed_scenario"], "comparisons": summary["comparisons"],
                                  "full_paper_profile": summary["full_paper_profile"]}, indent=2))
        + "</pre><p><a href='summary.json'>完整驗收</a> · <a href='inputs/application.json'>Application</a> · <a href='inputs/full.json'>Full experiment</a> · <a href='inputs/starc.json'>STARC experiment</a></p></html>")
    print(json.dumps({"output": str(output), "native_calls_including_replays": summary["native_calls_including_replays"],
                      "all_frozen_replays_passed": summary["all_frozen_replays_passed"],
                      "decode_latency_comparison": summary["comparisons"]["decode_latency"]}))


if __name__ == "__main__":
    main()

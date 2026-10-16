#!/usr/bin/env python3
"""Audit real sealed runs through explicit resource and evidence contracts."""

import argparse
import copy
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pim_foundry import application, comparison


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def resources(bundle, output):
    result = json.loads((bundle / "result.json").read_text())
    platform = json.loads((bundle / "experiment.json").read_text())["platform"]
    amounts = {name: None for name in comparison.UNITS}
    amounts.update(cpu_count=0, gpu_count=0, npu_count=0, pim_count=0)
    if "devices" in platform:
        for device in platform["devices"]:
            key = {"cpu": "cpu_count", "gpu": "gpu_count", "npu": "npu_count", "sram": "pim_count", "hbm": "pim_count"}[device["kind"]]
            amounts[key] += 1
        amounts["capacity"] = sum(device["capacity_bytes"] for device in platform["devices"])
        convention = "Counts refer to configured device objects; host execution is excluded by the recorded cost contract."
    elif platform.get("kind") == "cimflow-digital-sram/v1":
        amounts["pim_count"] = platform["cores"]
        convention = "PIM count refers to configured CIMFlow cores, not chips; only this same-platform failure check uses this disclosure."
    else:
        raise ValueError("resource extraction needs an explicit platform mapping")
    record = {"schema": comparison.RESOURCE_SCHEMA, "platform_sha256": result["platform_sha256"],
              "resources": {name: {"value": amounts[name], "unit": unit,
                                   "basis": "undisclosed" if amounts[name] is None else "model_assumption",
                                   "description": "Unknown: no defensible whole-platform value in this artifact." if amounts[name] is None else convention}
                            for name, unit in comparison.UNITS.items()},
              "compute_convention": "undisclosed whole-platform compute; no peak-sum normalization",
              "cost_basis": "undisclosed; no price or TDP numbers invented",
              "sources": [comparison.file_ref(bundle / "experiment.json"), comparison.file_ref(bundle / "result.json")]}
    return save(output, record)


def run_case(name, left, right, output, *, track="same_platform", quality="numerical", budget=None,
             metric="latency", replay_paths=None):
    directory = output / name
    directory.mkdir()
    evidence_paths = []
    repeats = {}
    for side, bundle in (("left", left), ("right", right)):
        result = json.loads((bundle / "result.json").read_text())
        scope = result["metrics"][metric]["scope"]
        evidence_paths.append(save(directory / (side + "-correctness.json"), {
            "schema": comparison.EVIDENCE_SCHEMA, "kind": "correctness", "basis": "observed",
            "status": result["quality"]["numerical"]["status"] if result["quality"]["numerical"]["status"] in ("passed", "failed") else "not_evaluated",
            "result_sha256": [application.file_sha256(bundle / "result.json")], "scope": scope,
            "sources": [comparison.file_ref(bundle / "result.json")], "details": result["quality"]["numerical"]}))
        replay = (replay_paths or {}).get(side, bundle.with_name(bundle.name + "-replay"))
        runs = [bundle]
        if replay.is_dir():
            evidence_paths.append(save(directory / (side + "-replay.json"), comparison.replay_evidence(bundle, replay, scope)))
            runs.append(replay)
        repeats[side] = [{"path": str(run), "manifest_sha256": application.file_sha256(run / "manifest.json")} for run in runs]
    # Equal tasks may point at the same original result; distinct evidence files
    # can then have identical content. Pin each content digest only once.
    evidence_paths = list({application.file_sha256(path): path for path in evidence_paths}.values())
    protocol = {"schema": comparison.PROTOCOL_SCHEMA, "id": name, "track": track, "metric": metric,
                "quality_requirement": quality,
                "resources": {side: comparison.file_ref(resources(bundle, directory / (side + "-resources.json"))) for side, bundle in (("left", left), ("right", right))},
                "fixed_axes": [], "budget_constraints": budget or [], "cost_model_bindings": [],
                "evidence_sha256": [application.file_sha256(path) for path in evidence_paths], "repetitions": repeats,
                "statistics": {"method": "deterministic", "seed": 0, "resamples": 0, "confidence": .95, "sampling_unit": "independent_runs"}}
    path = save(directory / "protocol.json", protocol)
    result = comparison.compare(left, right, path, evidence_paths)
    save(directory / "report.json", result)
    return {"case": name, "track": track, "comparable": result["comparable"], "qualified": result["qualified"],
            "ratio": result["left_over_right_ratio"], "blocking_reasons": result["blocking_reasons"],
            "qualification_gaps": result["qualification_gaps"], "report": str(directory / "report.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--transformer-study", type=Path,
                        help="Audit complete Transformer run/replay versus baseline/baseline-replay, requiring task quality")
    args = parser.parse_args()
    output, runs = args.out.resolve(), args.runs.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.transformer_study:
        study = args.transformer_study.resolve()
        case = run_case("native_hbm_vs_gpu_fp16", study / "run", study / "baseline", output,
                        quality="task", metric="generation_modeled_latency_ns",
                        replay_paths={"left": study / "replay", "right": study / "baseline-replay"})
        summary = {"schema": "pim-foundry.comparison-validation.v1", "passed": case["qualified"], "cases": [case],
                   "hardware_claim": False,
                   "quality_scope": "next-token NLL preservation on the declared 16-token fixture; no general language-model quality claim",
                   "note": "Full pretrained Transformer inference, KV cache, explicit FP16 variant, same configured HBM/GPU platform, original and frozen replay."}
        save(output / "summary.json", summary)
        value = case["ratio"]
        (output / "report.html").write_text("<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><title>Transformer comparison</title>"
            "<style>body{font:18px/1.8 system-ui;max-width:950px;margin:50px auto;padding:20px}pre{white-space:pre-wrap}</style>"
            "<h1>完整 Transformer：相同平台與精度比較</h1><p>兩邊使用相同模型、輸入、FP16 節點、HBM/GPU 平台與成本範圍；比較 HBM 混合映射與全 GPU 映射。</p>"
            "<p>GPU 時間為分析模型，HBM 時間為原生模擬器輸出；所有功能值由實際張量／指令直譯執行。這裡沒有實機效能測量。</p>"
            "<p>指定 fixture 的 next-token NLL、數值正確性及精確重跑條件：<strong>" + str(case["qualified"]) + "</strong>。</p>"
            "<p>HBM 混合／全 GPU 的模型延遲比：<strong>" + html.escape(str(value)) + "</strong>。比值大於 1 表示此條件下 HBM 混合映射較慢。</p>"
            "<p>任務品質只涵蓋固定的 16 個 scored tokens，不能推論通用語言品質。每次圖呼叫都是 cold，KV 在呼叫間經 host 搬移，沒有省略這項成本。</p>"
            "<p><a href='native_hbm_vs_gpu_fp16/report.json'>完整比較與來源雜湊</a> · <a href='native_hbm_vs_gpu_fp16/protocol.json'>固定比較規範</a></p>"
            "<pre>" + html.escape(json.dumps(case, indent=2)) + "</pre></html>")
        print(json.dumps(summary, indent=2))
        return 0 if summary["passed"] else 1
    old = runs / "application-v06"
    gpu, npu = old / "onnx-hbm-gpu-native", old / "onnx-hbm-npu-native"
    cases = [run_case("same_platform", gpu, old / "onnx-hbm-gpu-baseline", output),
             run_case("whole_system_gpu_vs_npu", gpu, npu, output, track="whole_system"),
             run_case("equal_memory_budget", gpu, npu, output, track="equal_budget",
                      budget=[{"resource": "capacity", "relation": "equal", "value": 33554432, "relative_tolerance": 0}]),
             run_case("reject_over_budget", gpu, npu, output, track="equal_budget",
                      budget=[{"resource": "capacity", "relation": "at_most", "value": 16777216, "relative_tolerance": 0}]),
             run_case("reject_missing_task_quality", gpu, npu, output, track="whole_system", quality="task"),
             run_case("reject_changed_application", gpu, old / "typed-sram-hbm-gpu-native", output, track="whole_system"),
             run_case("reject_cimflow_known_reproduction_failure", runs / "cimflow-applications-v07-validated/resnet18-dp",
                      runs / "cimflow-applications-v07-validated/resnet18-baseline1", output)]
    checks = {"same_platform_qualified": cases[0]["qualified"], "whole_system_qualified": cases[1]["qualified"],
              "equal_budget_qualified": cases[2]["qualified"], "over_budget_blocked": not cases[3]["comparable"],
              "missing_task_quality_unqualified": not cases[4]["qualified"], "changed_task_blocked": not cases[5]["comparable"],
              "cimflow_failed_study_blocked": not cases[6]["comparable"]}
    summary = {"schema": "pim-foundry.comparison-validation.v1", "passed": all(checks.values()), "checks": checks, "cases": cases,
               "hardware_claim": False, "note": "Previously executed sealed artifacts are re-audited; this script launches no new native simulator or hardware run."}
    save(output / "summary.json", summary)
    rows = "".join("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in
                     (case["case"], case["track"], case["comparable"], case["qualified"], case["ratio"], "; ".join(case["blocking_reasons"] + case["qualification_gaps"]))) + "</tr>" for case in cases)
    (output / "report.html").write_text("<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><title>Comparison protocol validation</title>"
        "<style>body{font:16px/1.6 system-ui;margin:40px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:10px}</style>"
        "<h1>跨平台比較驗收</h1><p>已封存的原生執行結果重新查核：同平台、相同容量預算與整體系統比較。數字仍為模型結果。</p>"
        "<p>qualified 表示滿足指定 numerical 品質與重現條件；不表示模型有實機準確度或應用品質。未知功耗、成本與整體峰值保持未知。</p>"
        "<table><tr><th>Case</th><th>Track</th><th>Comparable</th><th>Qualified</th><th>Left/right</th><th>Reasons</th></tr>" + rows + "</table></html>")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

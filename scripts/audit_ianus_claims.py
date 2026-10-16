#!/usr/bin/env python3
"""Recompute IANUS paper arithmetic; never impersonate its private simulator."""

import argparse
from fractions import Fraction
import hashlib
import html
import json
import math
from pathlib import Path
import shutil


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("paper numeric inputs must be finite and positive")
    return Fraction(str(value))


def calculate(spec):
    if spec["schema"] != "pim-foundry.paper-arithmetic-spec.v1":
        raise ValueError("unsupported arithmetic specification")
    p = {name: positive(value) for name, value in spec["table1_table2_page8"].items()}
    core = p["pe_rows"] * p["pe_columns"] * p["macs_per_pe_cycle"] * p["flops_per_mac_assumption"] * p["frequency_ghz"] / 1000
    chips = p["channels"] / p["channels_per_chip"]
    arithmetic = []

    def check(name, computed, reported, unit, printed_half_step, formula):
        arithmetic.append({"name": name, "computed": float(computed), "paper_reported": float(reported),
                           "unit": unit, "formula": formula,
                           "printed_rounding_half_step": float(printed_half_step),
                           "status": "consistent_with_printed_rounding" if abs(computed - reported) <= printed_half_step else "arithmetic_mismatch",
                           "evidence": "paper_inputs_only_not_independent_measurements"})

    check("NPU core peak", core, p["reported_core_tflops"], "TFLOP/s", Fraction(1, 2), "128 * 64 * 4 MAC * 2 FLOP/MAC * 0.7 GHz / 1000")
    check("NPU total peak", core * p["cores"], p["reported_total_tflops"], "TFLOP/s", Fraction(1, 2), "4 * derived core peak")
    check("External bandwidth", p["pin_gbps"] * p["channel_pins"] * p["channels"] / p["bits_per_byte"], p["reported_external_gbps"], "GB/s", Fraction(0), "16 Gb/s/pin * 16 pins/channel * 8 channels / 8 bits/byte")
    check("PIM total peak", p["channels"] * p["banks_per_channel"] * p["gflops_per_pu"] / 1000, chips * p["reported_pim_chip_tflops"], "TFLOP/s", chips * Fraction(1, 2), "8 channels * 16 banks/channel * 32 GFLOP/s / 1000; paper chip peak is rounded")
    check("Internal bandwidth", chips * p["internal_gbps_per_chip"], p["reported_internal_gbps"], "GB/s", Fraction(0), "(8 channels / 2 channels/chip) * 1024 GB/s/chip")
    a = spec["attention_example_section5_3_page8"]
    check("Attention row utilization", positive(a["used_bf16_elements"]) / positive(a["row_bf16_elements"]) * 100, positive(a["reported_percent"]), "%", Fraction(0), "64 / 1024 * 100")

    f9 = spec["figure9_page10"]
    if not (len(f9["dfx"]) == len(f9["npu_mem"]) == len(f9["ianus"]) == len(f9["input_output_tokens"]) == 9):
        raise ValueError("Figure 9 requires nine aligned printed observations")
    dfx, npu, ianus = ([positive(x) for x in f9[key]] for key in ("dfx", "npu_mem", "ianus"))
    ratios = [x / y for x, y in zip(dfx, ianus)]
    ratio_of_sums = sum(dfx) / sum(ianus)
    mean_of_ratios = sum(ratios) / len(ratios)
    geometric_mean = math.exp(sum(math.log(float(x)) for x in ratios) / len(ratios))
    slowdown = (sum(npu) / sum(dfx) - 1) * 100
    check("Figure 9 ratio of total displayed latencies", ratio_of_sums, positive(f9["reported_average_dfx_speedup"]), "ratio", Fraction(1, 20), "sum(DFX displayed ms) / sum(IANUS displayed ms)")
    check("Figure 9 NPU slowdown", slowdown, positive(f9["reported_npu_slowdown_percent"]), "%", Fraction(1, 2), "(sum(NPU-MEM displayed ms) / sum(DFX displayed ms) - 1) * 100")
    aggregation = {"ratio_of_sums": float(ratio_of_sums), "arithmetic_mean_of_case_speedups": float(mean_of_ratios),
                   "geometric_mean_of_case_speedups": geometric_mean,
                   "interpretation": "3.2x is numerically consistent with ratio of sums of rounded figure labels. This infers an aggregation, not author code or raw-data verification."}

    f14 = spec["figure14_page11"]
    if not (len(f14["models"]) == len(f14["reported_throughput_ratios"]) == len(f14["reported_utilization_ratios"]) == 4):
        raise ValueError("Figure 14 requires four aligned model observations")
    if type(f14["reported_decimal_places"]) is not int or not 0 <= f14["reported_decimal_places"] <= 6:
        raise ValueError("invalid printed precision")
    half = Fraction(1, 2 * 10 ** f14["reported_decimal_places"])
    peak_ratio = p["gpu_peak_tflops"] / p["reported_total_tflops"]
    utilization = []
    for model, throughput, observed in zip(f14["models"], f14["reported_throughput_ratios"], f14["reported_utilization_ratios"]):
        throughput, observed = positive(throughput), positive(observed)
        if throughput <= half or observed <= half:
            raise ValueError("rounding interval must stay positive")
        predicted = [(throughput - half) * peak_ratio, (throughput + half) * peak_ratio]
        reported = [observed - half, observed + half]
        overlap = max(predicted[0], reported[0]) <= min(predicted[1], reported[1])
        implied_gpu_peak = [p["reported_total_tflops"] * reported[0] / (throughput + half),
                            p["reported_total_tflops"] * reported[1] / (throughput - half)]
        utilization.append({"model": model, "reported_throughput_ratio": float(throughput),
                            "reported_utilization_ratio": float(observed),
                            "expected_utilization_ratio_using_table2": float(throughput * peak_ratio),
                            "expected_interval_allowing_print_rounding": [float(x) for x in predicted],
                            "reported_interval_allowing_print_rounding": [float(x) for x in reported],
                            "implied_gpu_peak_interval_tflops": [float(x) for x in implied_gpu_peak],
                            "declared_255_tflops_in_implied_interval": implied_gpu_peak[0] <= p["gpu_peak_tflops"] <= implied_gpu_peak[1],
                            "status": "consistent_under_declared_denominator" if overlap else "denominator_inconsistency_under_stated_assumptions"})
    f13 = spec["figure13_page11"]
    confounder = {"total_capacity_equal": positive(f13["unified_pim_gb"]) == positive(f13["partitioned_pim_gb"]) + positive(f13["partitioned_dram_gb"]),
                  "pim_capacity_and_paper_stated_compute_ratio": float(positive(f13["unified_pim_gb"]) / positive(f13["partitioned_pim_gb"])),
                  "scheduling_only_causal_claim_eligible": False,
                  "reason": "Figure 13 changes PIM resources together with memory organization. Same total capacity does not isolate scheduling."}
    return {"schema": "pim-foundry.paper-arithmetic-audit.v1", "paper": spec["paper"],
            "source_revision": spec["source_revision"], "execution_mode": "paper_arithmetic_only",
            "original_ianus_artifact_executed": False, "independent_hardware_measurement": False,
            "paper_speedup_independently_reproduced": False, "hardware_performance_comparison_eligible": False,
            "arithmetic_checks": arithmetic, "figure9_aggregation": aggregation,
            "figure14_assumption": f14["assumption"], "figure14_denominator_checks": utilization,
            "figure13_confounder": confounder,
            "scope": "Transcribed paper parameters and rounded labels; no model weights, timing simulation, raw timing records or GPU benchmark executed.",
            "conclusion": "Some paper arithmetic is consistent; Figure 14 cannot be reconciled with Table 2 under the stated denominator/aggregation assumptions. This does not establish data fabrication."}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=root / "configs/papers/ianus-arxiv-v1.json")
    parser.add_argument("--paper", type=Path, default=root / "runs/ianus-validation/ianus-paper.pdf")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if sha(args.paper) != spec["source_sha256"]:
        raise ValueError("paper bytes differ from the audited version")
    result = calculate(spec)
    result["provenance"] = {"paper_sha256": sha(args.paper), "spec_sha256": sha(args.spec), "script_sha256": sha(__file__)}
    output = args.out.resolve(); output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.paper, output / "source.pdf")
    shutil.copyfile(args.spec, output / "specification.json")
    shutil.copyfile(__file__, output / "audit.py")
    save(output / "result.json", result)
    esc = lambda value: html.escape(str(value))
    rows = "".join("<tr>" + "".join("<td>" + esc(value) + "</td>" for value in
                   (row["name"], row["computed"], row["paper_reported"], row["unit"], row["status"])) + "</tr>" for row in result["arithmetic_checks"])
    discrepancies = "".join("<tr>" + "".join("<td>" + esc(value) + "</td>" for value in
                            (row["model"], row["reported_throughput_ratio"], round(row["expected_utilization_ratio_using_table2"], 4), row["reported_utilization_ratio"], row["status"])) + "</tr>" for row in result["figure14_denominator_checks"])
    (output / "report.html").write_text("""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IANUS 數字與證據稽核</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#1c303b;background:#f7f9fb}table{border-collapse:collapse;background:white}td,th{padding:10px;border:1px solid #ddd;overflow-wrap:anywhere}pre{white-space:pre-wrap}a{color:#126473}.scroll{overflow:auto}</style>
<h1>IANUS：論文算術核對</h1><p>本頁只重算已發表的參數與圖表數字。沒有執行 IANUS 作者模擬器，沒有量測硬體，也沒有把圖表重算稱為效能重現。</p>
<p><a href="source.pdf">固定版本原文</a> · <a href="specification.json">人工轉錄數據與來源位置</a> · <a href="result.json">完整公式、區間與判定</a> · <a href="manifest.json">檔案雜湊</a></p>
<h2>規格與圖表內部算術</h2><div class="scroll"><table><tr><th>核對</th><th>重算值</th><th>論文值</th><th>單位</th><th>判定</th></tr>""" + rows + """</table></div>
<p>Fig. 9 的 3.2× 與總延遲比相容；逐案例加速比的算術平均、幾何平均會不同。完整數字保存在 result.json。這只能核對聚合算術，不能驗證原始測量。</p>
<h2>Fig. 14：利用率分母需要釐清</h2><p>若利用率＝實際 FLOPS／固定峰值，且採相同 workload 與平均權重，利用率比應等於吞吐比 × 255/184。即使容許圖中一位小數的四捨五入，四組區間仍無重疊。</p>
<div class="scroll"><table><tr><th>模型</th><th>圖中吞吐比</th><th>依 Table 2 預期利用率比</th><th>圖中利用率比</th><th>判定</th></tr>""" + discrepancies + """</table></div>
<p>這表示公開數字的分母或平均定義需要說明；沒有據此推斷資料造假。</p>
<h2>Fig. 13：容量相同，PIM 資源不同</h2><p>統一系統用 8 GB PIM；分離系統用 4 GB PIM＋4 GB DRAM。論文明示前者 PIM 算力加倍，這組比較不能隔離排程本身的貢獻。</p>
<p>完整研究審查與公開程式搜尋紀錄：<a href="../../docs/ianus-paper-audit.md">論文稽核</a> · <a href="../../docs/ianus-artifact-discovery.md">Artifact 搜尋</a>。</p>
<h2>離線重算</h2><pre>python3 audit.py --spec specification.json --paper source.pdf --out NEW_OUTPUT_DIRECTORY</pre></html>""")
    save(output / "manifest.json", {"schema": "pim-foundry.paper-audit-files.v1", "files": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}})
    print(json.dumps({"output": str(output), "mode": result["execution_mode"],
                      "rounding_consistent_checks": sum(row["status"] == "consistent_with_printed_rounding" for row in result["arithmetic_checks"]),
                      "conditional_denominator_inconsistencies": sum(row["status"].startswith("denominator_inconsistency") for row in result["figure14_denominator_checks"])}))


if __name__ == "__main__":
    main()

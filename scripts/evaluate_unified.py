#!/usr/bin/env python3
"""Execute native mixed graphs, fixed-platform baselines, and frozen replay."""

import argparse
from copy import deepcopy
import html
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_foundry import unified, unified_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        raise ValueError("output must be new")
    output.mkdir(parents=True)
    summary = {"schema_version": 1, "version": "0.5.0", "cases": [], "comparisons": [],
               "native_numerical_claimed": False, "hardware_calibrated": False}
    for name in ("mixed", "hbm-gpu", "hbm-npu"):
        graph, inputs, placement = unified_examples.workload(name)
        kinds = ("sram", "hbm", "gpu", "npu") if name == "mixed" else tuple(name.split("-"))
        config = unified_examples.platform(kinds, root=ROOT)
        native = unified.run(graph, inputs, config, placement, output / name)
        accelerator = "npu" if name == "hbm-npu" else "gpu"
        baseline = unified.run(graph, inputs, config, {node.name: accelerator for node in graph.nodes}, output / (name + "-baseline"))
        for suffix, result in (("", native), ("-baseline", baseline)):
            directory = output / (name + suffix)
            check = unified.verify(directory)
            replay_dir = output / (name + suffix + "-replayed")
            process = subprocess.run([sys.executable, str(directory / "reproduce.py"), str(replay_dir)],
                                     check=True, text=True, capture_output=True, timeout=60)
            replay = json.loads(process.stdout)
            if not replay["passed"]:
                raise ValueError("frozen replay differs")
            summary["cases"].append({"name": name + suffix, "latency_ns": result["modeled_latency_ns"],
                                      "energy_pj": result["modeled_energy_pj"], "quality": result["quality"],
                                      "integrity": check, "frozen_replay": replay,
                                      "nodes": [{"backend": n["backend"], "instruction_lowering_verified": n["instruction_lowering_verified"]} for n in result["nodes"]]})
        summary["comparisons"].append({"case": name, "latency": unified.compare(baseline, native),
                                       "energy": unified.compare(baseline, native, metric="energy")})
        print(name, "verified and replayed", flush=True)
        if name == "mixed":
            instrumented_config = deepcopy(config)
            for device in instrumented_config["devices"]:
                if device["backend"] == "hbm-simd16":
                    device["options"]["variant"] = "asan"
            instrumented = unified.run(graph, inputs, instrumented_config, placement, output / "mixed-asan")
            summary["asan_same_outputs_and_modeled_time"] = (instrumented["outputs"] == native["outputs"]
                                                           and instrumented["modeled_latency_ns"] == native["modeled_latency_ns"])
            if not summary["asan_same_outputs_and_modeled_time"]:
                raise ValueError("instrumented runtime differs")
            # Separate sensitivity experiment; the comparison gate must identify
            # the changed fabric assumption instead of reporting it as speedup.
            changed_config = deepcopy(config)
            for link in changed_config["links"]:
                link["bytes_per_ns"] *= 2
            changed = unified.run(graph, inputs, changed_config, placement, output / "mixed-double-fabric-bandwidth")
            rejection = unified.compare(native, changed)
            if rejection["same_model_comparable"]:
                raise ValueError("different platform assumptions were accepted")
            summary["changed_model_comparison_rejected"] = rejection
    summary["all_numeric_checks_passed"] = all(case["quality"]["passed"] for case in summary["cases"])
    summary["all_frozen_replays_passed"] = all(case["frozen_replay"]["passed"] for case in summary["cases"])
    unified.save(output / "summary.json", summary)
    rows = "".join(f"<tr><td><a href='{case['name']}/report.html'>{html.escape(case['name'])}</a></td>"
                   f"<td>{case['latency_ns']:.3f}</td><td>{case['energy_pj']:.3f}</td>"
                   f"<td>{case['quality']['passed']}</td><td>{case['frozen_replay']['passed']}</td></tr>" for case in summary["cases"])
    (output / "report.html").write_text("""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>PIM Foundry v0.5</title>
<style>body{font:16px/1.7 system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#17232d}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}a{color:#126b77}</style>
<h1>PIM Foundry v0.5：共同計算圖實驗</h1><p>SRAM 與 HBM 指令解譯器產生中間結果，原生模擬器提供時序；GPU/NPU 與搬移使用明示分析模型。所有數值與固定版本重跑均通過。以下數字是指定模型下的預測，並非硬體實測或論文加速比。</p>
<table><tr><th>實驗</th><th>模型時間 ns</th><th>模型能量 pJ</th><th>數值相符</th><th>快照重跑</th></tr>""" + rows + "</table><p>baseline 保留同一平台的所有裝置與閒置成本，將相同圖配置到 GPU/NPU 分析模型。不同頻寬設定的實驗被比較規則拒絕。</p><p><a href='summary.json'>完整驗收資料</a> · <a href='mixed-asan/report.html'>記憶體檢查版本</a> · <a href='mixed-double-fabric-bandwidth/report.html'>頻寬敏感度</a></p></html>")
    print(json.dumps({"output": str(output), "cases": len(summary["cases"]),
                      "all_numeric_checks_passed": summary["all_numeric_checks_passed"],
                      "all_frozen_replays_passed": summary["all_frozen_replays_passed"],
                      "asan_same_outputs_and_modeled_time": summary["asan_same_outputs_and_modeled_time"]}))


if __name__ == "__main__":
    main()

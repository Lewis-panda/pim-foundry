#!/usr/bin/env python3
"""Exercise immutable application manifests against several deployment bindings."""

import argparse
from copy import deepcopy
import html
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_foundry import application, typed, unified, unified_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--onnx-assets", type=Path, default=ROOT / "runs/onnx-frontend-v06/fp16-mlp")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = out / "specifications"
    spec.mkdir()
    onnx_assets = args.onnx_assets.resolve()
    onnx_app = {"schema": application.APP_SCHEMA, "id": "onnx-two-layer-mlp", "adapter": "onnx-graph/v1",
                "assets": {"model": application.asset(onnx_assets / "model.onnx", "model", "onnx"),
                           "inputs": application.asset(onnx_assets / "inputs.json", "inputs")},
                "parameters": {"input_shapes": json.loads((onnx_assets / "input-shapes.json").read_text())}, "scenario": application.STATELESS,
                "quality": {"kind": "numerical_equivalence", "reference": "onnx.reference.ReferenceEvaluator", "atol": 0.001, "rtol": 0.001}}
    unified.save(spec / "onnx-application.json", onnx_app)
    graph, inputs, _ = unified_examples.workload("mixed")
    # Semantic operation names carry no placement instructions.
    names = {"sram_linear": "integer_projection", "hbm_linear": "floating_projection"}
    graph = typed.Graph(graph.tensors, tuple(typed.Node(names.get(n.name, n.name), n.op, n.inputs, n.output, n.arithmetic) for n in graph.nodes), graph.inputs, graph.outputs)
    unified.save(spec / "typed-graph.json", typed.graph_to_dict(graph))
    unified.save(spec / "typed-inputs.json", inputs)
    typed_app = {"schema": application.APP_SCHEMA, "id": "explicit-mixed-precision-mlp", "adapter": "typed-graph/v1",
                 "assets": {"graph": application.asset(spec / "typed-graph.json", "model", "pim-foundry.typed-graph.v1"),
                            "inputs": application.asset(spec / "typed-inputs.json", "inputs")},
                 "parameters": {}, "scenario": application.STATELESS,
                 "quality": {"kind": "numerical_equivalence", "reference": "typed_scalar/v1", "atol": 0, "rtol": 0}}
    unified.save(spec / "typed-application.json", typed_app)
    results, summary = {}, {"version": "0.6.0", "applications": [], "comparisons": [], "paper_benchmark": False}
    for app_name, platform_name, kinds in (("onnx", "hbm-gpu", ("hbm", "gpu")), ("onnx", "hbm-npu", ("hbm", "npu")),
                                          ("typed", "sram-hbm-gpu", ("sram", "hbm", "gpu"))):
        config = unified_examples.platform(kinds, root=ROOT)
        accelerator = "npu" if "npu" in kinds else "gpu"
        for label, preference in (("native", list(kinds)), ("baseline", [accelerator])):
            name = app_name + "-" + platform_name + "-" + label
            experiment = {"schema": application.EXPERIMENT_SCHEMA, "platform": config,
                          "mapping": {"strategy": "first_feasible", "device_preference": preference, "max_candidates": 4096}, "variant": {}}
            experiment_path = spec / (name + ".json")
            unified.save(experiment_path, experiment)
            result = application.run(spec / (app_name + "-application.json"), experiment_path, out / name)
            if result["status"] != "completed" or result["quality"]["numerical"]["status"] != "passed":
                raise ValueError("application numerical quality did not pass")
            check = application.verify(out / name)
            replay_out = out / (name + "-replay")
            process = subprocess.run([sys.executable, str(out / name / "reproduce.py"), str(replay_out)],
                                     text=True, capture_output=True, check=True, timeout=90)
            replay = json.loads(process.stdout)
            if not replay["passed"]:
                raise ValueError("frozen application replay differs: " + name)
            results[name] = result
            summary["applications"].append({"name": name, "task_sha256": result["task_sha256"],
                                            "metrics": result["metrics"], "quality": result["quality"],
                                            "placement": result["diagnostics"]["placement"], "integrity": check, "replay": replay})
            print(name + " passed and replayed", flush=True)
        prefix = app_name + "-" + platform_name
        summary["comparisons"].append({"case": prefix, **application.compare(out / (prefix + "-baseline"), out / (prefix + "-native"))})
    onnx_results = [r for n, r in results.items() if n.startswith("onnx")]
    summary["same_onnx_task_across_gpu_npu"] = len({r["task_sha256"] for r in onnx_results}) == 1
    summary["same_onnx_outputs_across_gpu_npu"] = all(r["outputs"] == onnx_results[0]["outputs"] for r in onnx_results)
    changed_platform = application.compare_results(results["onnx-hbm-gpu-native"], results["onnx-hbm-npu-native"])
    if changed_platform["same_model_comparable"]:
        raise ValueError("different-platform comparison should require a separate fairness contract")
    summary["changed_platform_comparison_rejected"] = changed_platform
    if not summary["same_onnx_task_across_gpu_npu"] or not summary["same_onnx_outputs_across_gpu_npu"]:
        raise ValueError("application semantics changed across deployments")
    unified.save(out / "summary.json", summary)
    rows = "".join("<tr>" + f"<td><a href='{r['name']}/report.html'>{html.escape(r['name'])}</a></td>"
                   + f"<td>{r['metrics']['latency']['value']:.3f}</td><td>{html.escape(str(r['placement']))}</td><td>passed</td></tr>" for r in summary["applications"])
    (out / "report.html").write_text("<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><title>Application integration</title>"
        "<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:0 24px}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style>"
        "<h1>同一份 application，切換部署</h1><p>應用保存模型、輸入、精度與品質門檻；平台與配置策略由 experiment 決定。"
        "ONNX 直接讀取原始模型並與 ONNX ReferenceEvaluator 核對。以下模型是介面驗證用 fixture，沒有冒充論文工作負載。</p>"
        "<table><tr><th>Application / deployment</th><th>模型 ns</th><th>自動配置</th><th>數值 / 快照重跑</th></tr>" + rows + "</table>"
        "<p>所有時間為指定成本模型的預測。相同應用可換 GPU/NPU；不同平台不會自動取得公平效能比較資格。</p>"
        "<p><a href='summary.json'>完整結果</a> · <a href='../starc-application-v06/report.html'>STARC 原始論文 application 實驗</a></p></html>")
    print(json.dumps({"output": str(out), "cases": len(summary["applications"]), "replays": len(summary["applications"]),
                      "same_onnx_task_across_gpu_npu": summary["same_onnx_task_across_gpu_npu"]}))


if __name__ == "__main__":
    main()

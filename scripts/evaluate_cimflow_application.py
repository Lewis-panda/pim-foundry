#!/usr/bin/env python3
"""Run original CIMFlow backbone mapping methods through application contracts."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_foundry import application, cimflow_application as cim


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_descriptor(path):
    source = ROOT / "external/cimflow-compiler"
    prebuilt = ROOT / "external/cimflow-prebuilt"
    binary_root = prebuilt / "app/cimflow/modules/CIMFlow-Compiler/build"
    simulator = ROOT / "external/cimflow-simulator"
    python = source / ".venv/bin/python"
    java = Path(subprocess.check_output(["which", "java"], text=True).strip()).resolve()
    sources = subprocess.check_output(["git", "-C", str(source), "ls-files"], text=True).splitlines()
    sources = {name: sha(source / name) for name in sources if (name.startswith("src/") or name == "CIM.g") and (source / name).is_file()}
    libraries = [prebuilt / "runtime-libs/usr/lib/x86_64-linux-gnu" / name for name in ("libglog.so.3", "libgflags.so.2.2")]
    runtimes = [python, binary_root / "bin/main", simulator / "build/cim-simulator", java, *libraries]
    configurations = {str(b): str(simulator / f"config/example/config-mg8-b{b}.json") for b in (8, 16)}
    config_files = [Path(p) for p in configurations.values()] + [simulator / f"config/network/flit{b}dim8.json" for b in (8, 16)]
    packages = json.loads(subprocess.check_output([str(python), "-I", "-c", "import importlib.metadata,json;print(json.dumps(sorted(d.metadata['Name']+'=='+d.version for d in importlib.metadata.distributions())))"], text=True))
    descriptor = {"schema": "pim-foundry.cimflow-runtime.v1", "compiler_commit": cim.COMPILER_COMMIT,
                  "simulator_commit": cim.SIMULATOR_COMMIT, "source_root": str(source), "source_files": sources,
                  "python": str(python), "compiler_binary": str(binary_root / "bin/main"),
                  "simulator_binary": str(simulator / "build/cim-simulator"), "java": str(java),
                  "runtime_files": {str(p): sha(p) for p in runtimes},
                  "antlr_files": {str(p): sha(p) for p in sorted((binary_root / "antlr").glob("*.class"))},
                  "libraries": [str(p) for p in libraries], "configurations": configurations,
                  "config_files": {str(p): sha(p) for p in config_files}, "python_packages": packages,
                  "provenance": {"compiler": "author_prebuilt_public_OCI_layer", "compiler_rebuilt_from_source": False,
                                 "image_config_digest": "sha256:1d7b4f700dae01bbea3c4e573da93c3ee677498a7384d051e838692837dd6c0a",
                                 "compiler_layer_digest": "sha256:0e935a68451b845ce5d210e4c1af3f5a0ee995e7a21e8064937925a98ea3aa42",
                                 "simulator": "local_build_of_pinned_original_source", "source_patches": [],
                                 "paper": "https://doi.org/10.1109/DAC63849.2025.11133270",
                                 "release_models": "https://github.com/BUAA-CI-LAB/CIMFlow-Compiler/releases/tag/v0.1.0"}}
    save(path, descriptor)
    cim.validate_runtime(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(cim.MODELS), default=["resnet18"])
    parser.add_argument("--methods", nargs="+", choices=["dp", "baseline1", "baseline2"], default=["dp", "baseline1"])
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--existing", action="store_true", help="assess completed bundles and run only missing replays")
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=args.existing)
    descriptor = out / "runtime.json"
    if args.existing:
        cim.validate_runtime(descriptor)
    else:
        build_descriptor(descriptor)
    source_assets = ROOT / "runs/cimflow-application-validation/models"
    results = []
    for model in args.models:
        manifest = {"schema": application.APP_SCHEMA, "id": "cimflow-" + model + "-backbone",
                    "adapter": cim.ADAPTER, "assets": {"model": application.asset(source_assets / (model + ".onnx"), "model", "onnx")},
                    "parameters": {"model": model},
                    "scenario": {"kind": "dnn_network_shape_model", "batch_size": 8, "input_shape": [1, 3, 32, 32],
                                 "state": "cold_independent_batch", "scope": cim.SCOPE},
                    "quality": {"kind": "not_evaluated", "reason": "original compiler omits classifier/pooling and native execution models costs"}}
        app_path = out / (model + "-application.json"); save(app_path, manifest)
        for method in args.methods:
            experiment = {"schema": application.EXPERIMENT_SCHEMA,
                          "platform": {"kind": "cimflow-digital-sram/v1", "cores": 64, "macro_group_size": 8,
                                       "macro_groups": 16, "noc_bytes_per_flit": 8},
                          "mapping": {"strategy": "artifact_native"},
                          "variant": {"runtime_descriptor": str(descriptor), "method": method, "timeout_seconds": 900}}
            exp_path = out / (model + "-" + method + "-experiment.json"); save(exp_path, experiment)
            target = out / (model + "-" + method)
            result = json.loads((target / "result.json").read_text()) if args.existing and (target / "result.json").exists() else application.run(app_path, exp_path, target)
            verified = application.verify(target)
            direct = ROOT / f"runs/cimflow-application-validation/direct-{model}-{method}-b8"
            independent = None
            if (direct / "native-report.json").exists():
                observed = json.loads((target / "execution/artifact/native-report.json").read_text())
                expected = json.loads((direct / "native-report.json").read_text())
                observed.pop("exec_time_", None); expected.pop("exec_time_", None)
                independent = {"native_report_equal_excluding_host_wall_time": observed == expected,
                               "isa_byte_equal": sha(next((target / "execution/artifact").glob("isa_*.json"))) == sha(next(direct.glob("isa_*.json")))}
            replay_path = out / (target.name + "-replay")
            if args.replay and args.existing and (replay_path / "result.json").exists():
                application.verify(replay_path)
                after = json.loads((replay_path / "result.json").read_text())
                fields = ("status", "task_sha256", "execution_sha256", "platform_sha256", "cost_contract_sha256", "metrics", "quality", "execution_identity")
                checks = {name: result[name] == after[name] for name in fields}
                replay = {"passed": all(checks.values()), "checks": checks, "output": str(replay_path)}
            else:
                replay = application.replay(target, replay_path) if args.replay else None
            if result["quality_preserving_comparison_eligible"] or result["hardware_performance_comparison_eligible"]:
                raise AssertionError("timing-only backbone experiment was promoted to unsupported evidence")
            failures = []
            if independent is None:
                failures.append("independent_direct_execution_missing")
            elif not all(independent.values()):
                failures.append("reproducibility_failed:independent_direct_result_mismatch")
            if replay is not None and not replay["passed"]:
                failures.append("reproducibility_failed:frozen_replay_mismatch")
            results.append({"model": model, "method": method, "metrics": result["metrics"],
                            "quality": result["quality"], "bundle": str(target), "verification": verified,
                            "validation_status": "reproducibility_failed" if failures else "passed",
                            "failures": failures, "independent_direct_comparison": independent, "replay": replay})
            save(out / "summary.json", {"scope": cim.SCOPE, "warnings": cim.WARNINGS, "results": results})
            print(model, method, result["metrics"], flush=True)
    comparisons = []
    for model in args.models:
        if "dp" in args.methods and "baseline1" in args.methods:
            cohort = [x for x in results if x["model"] == model and x["method"] in ("dp", "baseline1")]
            if any(x["validation_status"] != "passed" for x in cohort):
                comparisons.append({"model": model, "same_model_comparable": False,
                                    "reasons": ["original_artifact_reproducibility_failed"], "left_over_right_ratio": None,
                                    "quality_preserving_comparison_eligible": False, "hardware_performance_comparison_eligible": False})
            else:
                comparisons.append(application.compare(out / (model + "-baseline1"), out / (model + "-dp"), metric="latency"))
    passed = all(x["validation_status"] == "passed" for x in results)
    summary = {"status": "passed" if passed else "reproducibility_failed",
               "validation_status": "passed" if passed else "reproducibility_failed", "scope": cim.SCOPE,
               "warnings": cim.WARNINGS, "results": results, "comparisons": comparisons}
    save(out / "summary.json", summary)
    rows = "".join("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in
                   (x["model"], x["method"], x["metrics"]["latency"]["value"], x["metrics"]["energy"]["value"],
                    x["validation_status"], x["independent_direct_comparison"], x["replay"])) + "</tr>" for x in results)
    (out / "report.html").write_text("<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><title>CIMFlow application validation</title>"
        "<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}td{overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>"
        "<h1>CIMFlow 原始 CNN backbone 驗證</h1><p>整體狀態：" + summary["status"] + "</p>"
        "<p>原始模型：32×32 輸入，batch 8，64 核心。FC 與 pooling 未完整編譯；此處無分類準確率或實測硬體速度主張。</p>"
        "<table><tr><th>模型</th><th>方法</th><th>模擬延遲 ms</th><th>模擬能量 pJ</th><th>驗證</th><th>獨立執行比對</th><th>凍結重跑</th></tr>" + rows + "</table>"
        "<p>ISA 完全相同而 timing 不同時，保留失敗並撤回 ratio qualification；不放寬容差。</p><pre>" + html.escape(json.dumps(comparisons, indent=2)) + "</pre></html>")
    print(json.dumps({"completed": len(results), "summary": str(out / "summary.json")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

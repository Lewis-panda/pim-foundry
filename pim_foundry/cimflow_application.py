"""Pinned CIMFlow CNN-backbone compilation and native timing adapter.

The published artifact accepts a network, but its released CG generator omits
the classifier and pooling. This adapter preserves that scope explicitly.
"""

import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

from .config import read_json
from .starc_adapter import _environment_identity, _fields, _hash, _integer, _save, _sha

ADAPTER = "cimflow-artifact/v1"
COMPILER_COMMIT = "67f3da074a6dec76b26046001ebe7c380a2bf9a4"
SIMULATOR_COMMIT = "531abcb869b1b3af1f95cf923223f09e08725d96"
MODELS = {
    "resnet18": "53eff7156fa2898391c8794edf1fa7d139334ef96d63bcc884ab74eb99233b86",
    "mobilenetv2": "b927a4a5dfc36c95f0858d71c0dd633bb3024c94c4c7cad1a236de9659526285",
}
SCOPE = "upstream_compiled_cnn_backbone_fixed_batch"
WARNINGS = [
    "Released CG generator emits convolution, residual Add and communication; FC/Gemm is outside its compiled backbone.",
    "Pooling generation is commented out in the released compiler; whole-network ONNX semantics are not preserved.",
    "No input images or output tensors are executed; timing and energy cannot establish task accuracy.",
    "The supplied ONNX graphs use 1x3x32x32 inputs and 100 outputs, not an ImageNet 224x224 workload.",
]


def prepare(manifest, assets):
    if manifest.get("adapter") != ADAPTER:
        raise ValueError("CIMFlow adapter name mismatch")
    _fields(manifest.get("parameters"), {"model"}, "CIMFlow parameters")
    model = manifest["parameters"]["model"]
    if model not in MODELS or set(assets) != {"model"}:
        raise ValueError("CIMFlow v1 supports the two audited author model releases")
    spec = manifest["assets"]["model"]
    if (spec.get("role") != "model" or spec.get("format") != "onnx" or spec.get("sha256") != MODELS[model]
            or _sha(assets["model"]) != MODELS[model]):
        raise ValueError("CIMFlow model must bind the original released ONNX asset")
    scenario = manifest.get("scenario")
    _fields(scenario, {"kind", "batch_size", "input_shape", "state", "scope"}, "CIMFlow scenario")
    if (scenario["kind"] != "dnn_network_shape_model" or scenario["input_shape"] != [1, 3, 32, 32]
            or scenario["state"] != "cold_independent_batch" or scenario["scope"] != SCOPE):
        raise ValueError("CIMFlow requires its explicit 32x32 CNN-backbone cost-model scope")
    _integer(scenario["batch_size"], 1, 64, "batch_size")
    quality = manifest.get("quality")
    _fields(quality, {"kind", "reason"}, "CIMFlow quality")
    if quality["kind"] != "not_evaluated" or not isinstance(quality["reason"], str) or not quality["reason"]:
        raise ValueError("CIMFlow timing adapter does not evaluate numerical or task quality")
    return {"adapter": ADAPTER, "model": model, "model_path": str(assets["model"]),
            "model_sha256": MODELS[model], "scenario": dict(scenario)}


def validate_runtime(path):
    runtime = read_json(Path(path))
    required = {"schema", "compiler_commit", "simulator_commit", "source_root", "source_files",
                "python", "compiler_binary", "simulator_binary", "java", "runtime_files",
                "antlr_files", "libraries", "config_files", "configurations", "python_packages", "provenance"}
    _fields(runtime, required, "CIMFlow runtime")
    if (runtime["schema"] != "pim-foundry.cimflow-runtime.v1" or runtime["compiler_commit"] != COMPILER_COMMIT
            or runtime["simulator_commit"] != SIMULATOR_COMMIT):
        raise ValueError("unsupported CIMFlow source revision")
    source = Path(runtime["source_root"]).resolve()
    minimum = {"CIM.g", "src/cim_compiler/cli/main.py", "src/cim_compiler/cli/commands/network.py",
               "src/cim_compiler/cg/preprocess/model_preprocess.py", "src/cim_compiler/cg/utils/instructions_gen.py"}
    if not isinstance(runtime["source_files"], dict) or not minimum <= set(runtime["source_files"]):
        raise ValueError("incomplete compiler source bindings")
    for name, expected in runtime["source_files"].items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or not (source / relative).resolve().is_relative_to(source)
                or not _hash(expected) or _sha(source / relative) != expected):
            raise ValueError("compiler source changed: " + name)
    for group in ("runtime_files", "antlr_files", "config_files"):
        if not isinstance(runtime[group], dict) or not runtime[group]:
            raise ValueError("missing runtime bindings: " + group)
        for name, expected in runtime[group].items():
            if not Path(name).is_absolute() or not _hash(expected) or _sha(name) != expected:
                raise ValueError("CIMFlow runtime file changed: " + name)
    for key in ("python", "compiler_binary", "simulator_binary", "java"):
        if runtime[key] not in runtime["runtime_files"]:
            raise ValueError("unbound executable: " + key)
    if not isinstance(runtime["libraries"], list) or not runtime["libraries"]:
        raise ValueError("libraries must be explicitly bound")
    for name in runtime["libraries"]:
        if name not in runtime["runtime_files"]:
            raise ValueError("unbound shared library")
    if set(runtime["configurations"]) != {"8", "16"}:
        raise ValueError("runtime needs both audited NoC configurations")
    for bandwidth, name in runtime["configurations"].items():
        if name not in runtime["config_files"]:
            raise ValueError("unbound architecture configuration")
        config = read_json(Path(name))
        chip = config["chip_config"]
        cim = chip["core_config"]["cim_unit_config"]
        if (chip["core_cnt"] != 64 or chip["network_config"]["bus_width_byte"] != int(bandwidth)
                or cim["macro_group_size"] != 8 or cim["macro_total_cnt"] != 128):
            raise ValueError("runtime configuration differs from requested topology")
        network = (Path(name).parent / chip["network_config"]["network_config_file_path"]).resolve()
        if str(network) not in runtime["config_files"]:
            raise ValueError("unbound network cost table")
    return runtime


def _binding(experiment):
    if experiment.get("mapping") != {"strategy": "artifact_native"}:
        raise ValueError("CIMFlow controls its own mapping; use artifact_native")
    platform = experiment.get("platform")
    _fields(platform, {"kind", "cores", "macro_group_size", "macro_groups", "noc_bytes_per_flit"}, "CIMFlow platform")
    if (platform["kind"] != "cimflow-digital-sram/v1" or type(platform["cores"]) is not int or platform["cores"] != 64
            or type(platform["macro_group_size"]) is not int or platform["macro_group_size"] != 8
            or type(platform["macro_groups"]) is not int or platform["macro_groups"] != 16
            or type(platform["noc_bytes_per_flit"]) is not int or platform["noc_bytes_per_flit"] not in (8, 16)):
        raise ValueError("CIMFlow v1 requires audited 64-core T8 K16 B8/B16 platform")
    variant = experiment.get("variant")
    _fields(variant, {"runtime_descriptor", "method", "timeout_seconds"}, "CIMFlow variant")
    if variant["method"] not in ("dp", "baseline1", "baseline2"):
        raise ValueError("unsupported original compiler mapping method")
    _integer(variant["timeout_seconds"], 1, 3600, "timeout_seconds")
    return platform, variant, validate_runtime(variant["runtime_descriptor"])


def inspect(prepared, experiment):
    try:
        if prepared.get("adapter") != ADAPTER:
            raise ValueError("invalid prepared application")
        platform, _, runtime = _binding(experiment)
        return {"passed": True, "execution_mode": "original_artifact_cnn_backbone_model", "platform": platform,
                "lowered_to_common_ir": False, "whole_model_semantics_preserved": False,
                "warnings": WARNINGS, "runtime_provenance": runtime["provenance"]}
    except (ValueError, KeyError, TypeError, OSError) as error:
        return {"passed": False, "reasons": [str(error)]}


def parse_report(path, expected_op_count):
    report = read_json(Path(path))
    if type(expected_op_count) is not int or expected_op_count <= 0 or report.get("op_count_") != expected_op_count:
        raise ValueError("native reported operation count differs from compiled metadata")
    metrics = {}
    for name, key, unit in (("latency", "latency_", "ms"), ("energy", "total_energy_", "pJ")):
        value = report.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("native metric must be finite and positive: " + key)
        metrics[name] = {"value": value, "unit": unit, "scope": SCOPE, "aggregation": "one_fixed_batch",
                         "origin": "upstream_systemc_report:" + key}
    return metrics


def _process(command, directory, environment, output, timeout):
    started = time.monotonic()
    with output.with_suffix(".stdout").open("wb") as stdout, output.with_suffix(".stderr").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=directory, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    _save(output.with_suffix(".json"), {"command": command, "returncode": process.returncode, "timed_out": timed_out,
                                       "host_wall_seconds_not_device_latency": time.monotonic() - started})
    if timed_out or process.returncode:
        raise ValueError("CIMFlow process failed; see " + str(output))


def execute(prepared, experiment, output):
    if prepared.get("adapter") != ADAPTER:
        raise ValueError("invalid prepared application")
    platform, variant, runtime = _binding(experiment)
    descriptor_sha = _sha(variant["runtime_descriptor"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _save(output / "runtime.json", runtime)
    _save(output / "binding.json", experiment)
    batch, model, bandwidth = prepared["scenario"]["batch_size"], prepared["model"], platform["noc_bytes_per_flit"]
    config_path = runtime["configurations"][str(bandwidth)]
    method = variant["method"]
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"):
        environment.pop(key, None)
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    python_identity = _environment_identity(runtime, environment)
    with tempfile.TemporaryDirectory(prefix="pim-foundry-cimflow-") as temporary:
        work = Path(temporary)
        source = work / "compiler"
        for name in runtime["source_files"]:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(runtime["source_root"]) / name, target)
        for name in runtime["antlr_files"]:
            target = source / "build/antlr" / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
        libraries = work / "lib"
        libraries.mkdir()
        for name in runtime["libraries"]:
            shutil.copyfile(name, libraries / Path(name).name)
        binaries = work / "bin"
        binaries.mkdir()
        launcher = binaries / "cim-compiler"
        launcher.write_text("#!" + runtime["python"] + "\nfrom cim_compiler.cli.main import main\nmain()\n")
        launcher.chmod(0o755)
        java = binaries / "java"
        java.symlink_to(runtime["java"])
        environment.update(PYTHONPATH=str(source / "src"), LD_LIBRARY_PATH=str(libraries),
                           CIM_MLIR_COMPILER_BIN=runtime["compiler_binary"],
                           PATH=str(binaries) + os.pathsep + str(Path(runtime["python"]).parent) + os.pathsep + environment.get("PATH", ""))
        artifact = output / "artifact"
        artifact.mkdir()
        model_path = work / (model + ".onnx")
        shutil.copyfile(prepared["model_path"], model_path)
        if _sha(model_path) != prepared["model_sha256"]:
            raise ValueError("model asset changed before compilation")
        command = [runtime["python"], "-m", "cim_compiler.cli.main", "network", "-m", str(model_path),
                   "-o", str(artifact), "-c", config_path, "-T", "8", "-K", "16", "-B", str(bandwidth),
                   "-C", "64", "--batch-size", str(batch), "--strategy", method, "--keep-cg-ir"]
        _process(command, work, environment, output / "compile", variant["timeout_seconds"])
        suffix = f"{model}_{method}_T8_K16_B{bandwidth}_C64_batch{batch}.json"
        isa_path, cg_path = artifact / ("isa_instructions_" + suffix), artifact / ("instructions_" + suffix)
        isa, cg = read_json(isa_path), read_json(cg_path)
        if set(isa) != {str(i) for i in range(64)} | {"metadata"} or isa["metadata"] != cg["metadata"]:
            raise ValueError("compiled ISA must bind all 64 cores and original CG metadata")
        op_counts = {}
        for core, value in cg.items():
            if core == "metadata":
                continue
            for stage in value["stages"].values():
                for instruction in stage["instructions"]:
                    name = instruction["op"]
                    op_counts[name] = op_counts.get(name, 0) + 1
        if not op_counts.get("conv") or not op_counts.get("add") or not op_counts.get("send"):
            raise ValueError("compiled model lacks expected convolution/residual/communication operations")
        command = [runtime["simulator_binary"], config_path, str(isa_path), "-o", str(artifact / "native-report.json"), "--format", "json"]
        _process(command, work, environment, output / "simulate", variant["timeout_seconds"])
        simplified = source / "src/cim_compiler/cg/data/simplified_model_files" / (model + "-simplified.onnx")
        shutil.copyfile(simplified, artifact / "preprocessed.onnx")
    if _sha(variant["runtime_descriptor"]) != descriptor_sha or validate_runtime(variant["runtime_descriptor"]) != runtime:
        raise ValueError("CIMFlow runtime changed during execution")
    if _environment_identity(runtime, environment) != python_identity:
        raise ValueError("Python environment changed during execution")
    metrics = parse_report(artifact / "native-report.json", isa["metadata"]["op_count"])
    identity = {"compiler_commit": COMPILER_COMMIT, "simulator_commit": SIMULATOR_COMMIT,
                "source_files": runtime["source_files"], "runtime_files": runtime["runtime_files"],
                "antlr_files": runtime["antlr_files"], "python_environment": python_identity,
                "provenance": runtime["provenance"]}
    cost = {"name": "cimflow-original-backbone/v1", "scope": SCOPE, "runtime": identity,
            "configurations": runtime["config_files"], "whole_model_semantics_preserved": False,
            "host_preprocessing_and_input_loading_included": False, "hardware_calibrated_by_this_run": False}
    result = {"execution_mode": "original_artifact_cnn_backbone_model", "metrics": metrics,
              "execution_identity": {"runtime": identity, "method": method, "isa_sha256": _sha(isa_path),
                                     "preprocessed_model_sha256": _sha(artifact / "preprocessed.onnx")},
              "platform_identity": {**platform, "configuration_sha256": _sha(config_path)}, "cost_contract": cost,
              "quality": {"numerical": {"status": "not_evaluated", "reason": "native execution models costs without model tensor values"},
                          "task": {"status": "not_evaluated", "reason": "no quality dataset or complete ONNX classifier was executed"}},
              "diagnostics": {"warnings": WARNINGS, "compiled_operation_counts": op_counts,
                              "native_op_count": isa["metadata"]["op_count"], "whole_model_semantics_preserved": False},
              "native_timing_executed": True, "native_arithmetic": False, "lowered_to_common_ir": False,
              "hardware_execution": False, "quality_preserving_speedup_eligible": False}
    _save(output / "result.json", result)
    return result

"""Pinned original NeuPIMs trace-driven, single-layer decode adapter.

This adapter executes author simulator code. It does not lower a transformer to
Foundry's typed IR and does not measure numerical or task accuracy.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ADAPTER = "neupims-artifact/v1"
COMMITS = {"neupims": "f299af3fc8f20077e816f2e48313294cddb4bd7c",
           "npu-only": "57617ee0d8abe8aa87a3551d42ee98f999fecd71"}
SCOPE = "original_single_layer_one_token_decode_batch_including_drain"
SCENARIO = {"kind": "trace_driven_decode", "requests": 512, "modeled_layers": 1,
            "generated_tokens_per_request": 1, "state": "shape_only_kv_cache_reset_each_run",
            "scope": SCOPE}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fields(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(label + " has invalid fields")


def prepare(manifest, assets):
    if manifest.get("adapter") != ADAPTER or set(assets) != {"model_definition", "request_lengths"}:
        raise ValueError("NeuPIMs requires model_definition and request_lengths assets")
    if manifest.get("parameters") != {"model": "GPT3-7B", "precision_bytes": 2} or manifest.get("scenario") != SCENARIO:
        raise ValueError("NeuPIMs v1 supports its audited 512-request one-layer decode scenario")
    for name, role in (("model_definition", "model"), ("request_lengths", "inputs")):
        if manifest["assets"][name]["role"] != role or sha(assets[name]) != manifest["assets"][name]["sha256"]:
            raise ValueError("NeuPIMs asset role/hash mismatch: " + name)
    model = json.loads(Path(assets["model_definition"]).read_text())
    expected = {"model_name": "GPT3-7B", "model_params_b": 7, "model_vocab_size": 50304,
                "model_n_layer": 1, "model_n_head": 32, "model_n_embd": 4096, "n_tp": 4, "n_pp": 1}
    if model != expected:
        raise ValueError("unsupported NeuPIMs model shape")
    lengths = json.loads(Path(assets["request_lengths"]).read_text())
    if (not isinstance(lengths, list) or len(lengths) != 512 or
            any(type(x) is not int or not 1 <= x <= 2048 for x in lengths)):
        raise ValueError("request_lengths must contain 512 integer lengths in 1..2048")
    quality = manifest.get("quality")
    fields(quality, {"kind", "reason"}, "quality")
    if quality["kind"] != "not_evaluated" or not isinstance(quality["reason"], str) or not quality["reason"]:
        raise ValueError("NeuPIMs timing artifact does not evaluate task quality")
    return {"model": model, "model_path": str(assets["model_definition"]), "request_lengths": lengths,
            "model_sha256": sha(assets["model_definition"]), "lengths_sha256": sha(assets["request_lengths"])}


def runtime(path):
    descriptor = json.loads(Path(path).read_text())
    fields(descriptor, {"schema", "method", "source_commit", "source_root", "source_files", "files",
                        "config_paths", "build", "scope"}, "runtime")
    if descriptor["schema"] != "pim-foundry.neupims-runtime.v1" or descriptor["scope"] != SCOPE:
        raise ValueError("unsupported NeuPIMs runtime schema/scope")
    if descriptor["method"] not in COMMITS or descriptor["source_commit"] != COMMITS[descriptor["method"]]:
        raise ValueError("unreviewed NeuPIMs source revision")
    root = Path(descriptor["source_root"]).resolve()
    source = descriptor["source_files"]
    required = {"src/client/Client.cc", "src/scheduler/Scheduler.cc", "src/Simulator.cc", "src/Model.cc",
                "src/ModelProgram.cc", "CMakeLists.txt", "conanfile.txt"}
    if not isinstance(source, dict) or not required <= set(source):
        raise ValueError("runtime lacks required source bindings")
    for rel, digest in source.items():
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or not (root / p).is_file() or sha(root / p) != digest:
            raise ValueError("modified/missing NeuPIMs source: " + str(rel))
    fields(descriptor["files"], {"executable", "library", "dependencies", "flags"}, "runtime files")
    for name, spec in descriptor["files"].items():
        fields(spec, {"path", "sha256"}, name)
        if not Path(spec["path"]).is_file() or sha(spec["path"]) != spec["sha256"]:
            raise ValueError("modified/missing runtime file: " + name)
    fields(descriptor["config_paths"], {"hardware", "memory", "system"}, "runtime configs")
    config = {}
    for name, rel in descriptor["config_paths"].items():
        if rel not in source:
            raise ValueError("config lacks source binding")
        config[name] = json.loads((root / rel).read_text())
    # Simulator.cc resolves this transitive timing file from its own source
    # directory, rather than from the memory JSON or the execution directory.
    memory = config["memory"]
    pim_config = memory.get("pim_config_path") if isinstance(memory, dict) else None
    if not isinstance(pim_config, str) or not pim_config or Path(pim_config).is_absolute():
        raise ValueError("PIM timing config must be a relative source path")
    resolved_ini = (root / "src" / pim_config).resolve()
    try:
        ini_relative = resolved_ini.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("PIM timing config escapes source root") from exc
    if ini_relative not in source:
        raise ValueError("PIM timing config lacks source binding")
    return descriptor, config


def bindings(prepared, experiment):
    variant = experiment.get("variant")
    fields(variant, {"method", "runtime_descriptor", "timeout_seconds"}, "variant")
    if type(variant["timeout_seconds"]) is not int or not 1 <= variant["timeout_seconds"] <= 300:
        raise ValueError("timeout_seconds must be an integer in 1..300")
    descriptor, config = runtime(variant["runtime_descriptor"])
    if variant["method"] != descriptor["method"]:
        raise ValueError("method/runtime mismatch")
    if experiment.get("platform") != {"kind": "neupims-original-system/v1", **config}:
        raise ValueError("declared platform must match effective original configuration files")
    mapping = experiment.get("mapping")
    fields(mapping, {"strategy", "trace", "sha256"}, "mapping")
    if mapping["strategy"] != "pinned_channel_assignment" or sha(mapping["trace"]) != mapping["sha256"]:
        raise ValueError("invalid channel-assignment binding")
    with Path(mapping["trace"]).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["seq_len", "ch_idx"]:
            raise ValueError("expected seq_len,ch_idx trace columns")
        rows = list(reader)
    try:
        lengths = [int(row["seq_len"]) for row in rows]
        channels = [int(row["ch_idx"]) for row in rows]
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("invalid request trace") from exc
    if lengths != prepared["request_lengths"]:
        raise ValueError("channel assignment changes semantic request lengths/order")
    if any(not 0 <= x < config["memory"]["dram_channels"] for x in channels):
        raise ValueError("channel assignment exceeds memory geometry")
    if config["hardware"]["precision"] != 2 or config["memory"]["dram_freq"] != 1000:
        raise ValueError("unaudited clock/precision")
    return descriptor, config


def inspect(prepared, experiment):
    try:
        descriptor, config = bindings(prepared, experiment)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [str(exc)]}
    return {"passed": True, "execution_mode": "original_artifact_system_model", "method": descriptor["method"],
            "lowered_to_common_ir": False, "native_arithmetic": False, "scope": SCOPE,
            "requests": len(prepared["request_lengths"]),
            "warnings": ["Pretrained weights and numerical KV values are not loaded.",
                         "Main branch hardcodes maximum batch/active requests to 1024; JSON limits are not effective.",
                         "Method branches and hardware configs differ; cross-method speedup is not automatically eligible."]}


def parse_output(path, *, expected_requests=512, dram_frequency_mhz=1000):
    text = Path(path).read_text(errors="strict")
    if "Simulation Finished" not in text:
        raise ValueError("native simulation did not finish")
    counts = re.findall(r"Client total request cnt: (\d+)", text)
    parsed_counts = re.findall(r"parsed (\d+) lines from file ", text)
    if not counts:
        counts = parsed_counts
    elif parsed_counts and parsed_counts != counts:
        raise ValueError("client and trace request counters disagree")
    cycles = re.findall(r"DRAM total cycles: (\d+)", text)
    if counts != [str(expected_requests)] or len(cycles) != 1 or int(cycles[0]) <= 0:
        raise ValueError("native request/cycle counters missing, duplicate or inconsistent")
    if not math.isfinite(dram_frequency_mhz) or dram_frequency_mhz <= 0:
        raise ValueError("invalid DRAM clock")
    latency = int(cycles[0]) * 1000 / dram_frequency_mhz
    stages = re.findall(r"Stage ([A-F]) : (\d+) cycles", text)
    return {"dram_cycles": int(cycles[0]), "latency_ns": latency,
            "stage_cycles": [{"stage": stage, "cycles": int(count)} for stage, count in stages]}


def execute(prepared, experiment, output):
    descriptor, config = bindings(prepared, experiment)
    before = sha(experiment["variant"]["runtime_descriptor"])
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    source = Path(descriptor["source_root"])
    # Isolate simulator-generated files and bind the actual inputs used by it.
    model = output / "model.json"; save(model, prepared["model"])
    trace = output / "requests.csv"; trace.write_bytes(Path(experiment["mapping"]["trace"]).read_bytes())
    if sha(trace) != experiment["mapping"]["sha256"]:
        raise ValueError("trace changed during snapshot")
    cmd = [descriptor["files"]["executable"]["path"], "--config", str(source / descriptor["config_paths"]["hardware"]),
           "--mem_config", str(source / descriptor["config_paths"]["memory"]), "--cli_config", str(trace),
           "--model_config", str(model), "--sys_config", str(source / descriptor["config_paths"]["system"]),
           "--log_dir", str(output)]
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None); env.pop("LD_AUDIT", None)
    env["LD_LIBRARY_PATH"] = str(Path(descriptor["files"]["library"]["path"]).parent)
    start = time.monotonic()
    with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(cmd, cwd=output, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=experiment["variant"]["timeout_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True; os.killpg(process.pid, signal.SIGKILL); code = process.wait()
    save(output / "invocation.json", {"command": cmd, "returncode": code, "timeout": timed_out,
                                      "wall_seconds": time.monotonic() - start})
    if timed_out or code != 0:
        raise ValueError("NeuPIMs native execution timed out or failed; raw logs retained")
    bindings(prepared, experiment)
    if sha(trace) != experiment["mapping"]["sha256"] or json.loads(model.read_text()) != prepared["model"]:
        raise ValueError("snapshotted native inputs changed during execution")
    if sha(experiment["variant"]["runtime_descriptor"]) != before:
        raise ValueError("runtime descriptor changed during execution")
    parsed = parse_output(output / "stdout.log")
    runtime_identity = {"source_commit": descriptor["source_commit"], "source_files": descriptor["source_files"],
                        "runtime_files": {key: value["sha256"] for key, value in descriptor["files"].items()},
                        "build": descriptor["build"]}
    metric = {"unit": "ns", "scope": SCOPE, "aggregation": "one_completed_batch",
              "origin": "author_cycle_simulator", "value": parsed["latency_ns"]}
    result = {"execution_mode": "original_artifact_system_model", "metrics": {"latency": metric},
              "platform_identity": experiment["platform"],
              "cost_contract": {"scope": SCOPE, "runtime": runtime_identity, "dram_frequency_mhz": 1000,
                                "includes": "upstream_modeled_memory_npu_pim_and_interconnect_drain",
                                "excludes": "prefill_full_model_replication_host_network_task_quality",
                                "energy": "not_exported_unverified_scope"},
              "execution_identity": {"method": descriptor["method"], "runtime": runtime_identity,
                                     "runtime_descriptor_sha256": before, "mapping_sha256": experiment["mapping"]["sha256"]},
              "quality": {"numerical": {"status": "not_evaluated", "reason": "simulator schedules operations but produces no numerical outputs"},
                          "task": {"status": "not_evaluated", "reason": "pretrained model and evaluation dataset were not executed"}},
              "diagnostics": {"native_counters": parsed, "request_count": 512},
              "native_timing_executed": True, "native_arithmetic": False, "lowered_to_common_ir": False,
              "hardware_execution": False, "quality_preserving_speedup_eligible": False}
    save(output / "result.json", result)
    return result

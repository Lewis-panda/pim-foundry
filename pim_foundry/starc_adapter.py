"""Application adapter for the pinned STARC CPU system-model artifact.

This executes the author's model, not a Foundry IR port or real LLM inference.
Every run starts with an empty private cache and preserves native calls before
the upstream wrapper removes their traces. Model latency never uses wall time.
"""

import ast
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time

from .config import read_json

ADAPTER = "starc-artifact/v1"
SOURCE_COMMIT = "63c8aaa7a02114cd4cf21f32e7621fab5b416957"
SOURCE_PREFIX = "simulator_starc/"
MODEL_DEFINITION = SOURCE_PREFIX + "src/config.py"
DEFAULT_PLATFORM = {"kind": "starc-system/v1", "system": "dgx-attacc", "gpu": "H100",
                    "ngpu": 8, "pim": "bank", "powerlimit": True, "ffopt": True, "pipeopt": True}
MODEL_VALUES = [32, 4096, 32, 128, 3.5, 4]


def _sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _fields(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(label + " requires exactly: " + ", ".join(sorted(expected)))


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _model_definition(path):
    """Read the literal model-table entry without importing application code."""
    def scalar(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return scalar(node.left) / scalar(node.right)
        raise ValueError("unsupported model definition expression")
    entries = []
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == "model_table" and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "Mistral-7B"):
                if not isinstance(node.value, ast.List):
                    raise ValueError("Mistral model definition must be a literal list")
                entries.append([scalar(item) for item in node.value.elts])
    if entries != [MODEL_VALUES]:
        raise ValueError("unsupported or ambiguous Mistral model definition")
    return dict(zip(("layers", "hidden_size", "attention_heads", "head_dimension", "ff_scale", "declared_gqa_size"), entries[0]))


def prepare(manifest, assets):
    """Validate application semantics without launching any external process."""
    if manifest.get("adapter") != ADAPTER:
        raise ValueError("STARC adapter name mismatch")
    _fields(manifest.get("parameters"), {"model"}, "STARC parameters")
    if manifest["parameters"]["model"] != "Mistral-7B":
        raise ValueError("STARC adapter v1 currently supports the audited Mistral-7B artifact model")
    if set(assets) != {"model_definition"}:
        raise ValueError("STARC application requires exactly one model_definition asset")
    asset_spec = manifest.get("assets", {}).get("model_definition", {})
    if asset_spec.get("role") != "model":
        raise ValueError("model_definition must have role=model")
    definition = Path(assets["model_definition"])
    definition_sha = _sha(definition)
    if asset_spec.get("sha256") != definition_sha:
        raise ValueError("model definition asset hash mismatch")
    model = _model_definition(definition)
    scenario = manifest.get("scenario")
    _fields(scenario, {"kind", "batch_size", "prompt_tokens", "generated_tokens", "state", "request_schedule", "scope"}, "STARC scenario")
    if (scenario["kind"] != "llm_prefill_decode_model" or scenario["request_schedule"] != "fixed_batch"
            or scenario["scope"] != "artifact_declared_system_model"
            or scenario["state"] != {"kind": "kv_cache", "representation": "shape_and_budget_only",
                                     "lifetime": "one_sequence", "reset": "before_each_run"}):
        raise ValueError("STARC v1 requires its fixed-batch shape-only prefill/decode scenario")
    _integer(scenario["batch_size"], 1, 64, "batch_size")
    _integer(scenario["prompt_tokens"], 1, 32768, "prompt_tokens")
    _integer(scenario["generated_tokens"], 2, 32000, "generated_tokens")
    quality = manifest.get("quality")
    _fields(quality, {"kind", "reason"}, "STARC quality policy")
    if quality["kind"] != "not_evaluated" or type(quality["reason"]) is not str or not quality["reason"]:
        raise ValueError("STARC system-model adapter cannot evaluate numerical or task quality")
    return {"adapter": ADAPTER, "model": "Mistral-7B", "model_definition_sha256": definition_sha,
            "model_definition": model, "scenario": json.loads(json.dumps(scenario)),
            "execution_mode": "original_artifact_system_model",
            "diagnostics": {"supported": True, "state": "shape_and_budget_only",
                            "lowered_to_common_ir": False,
                            "warnings": ["Model name denotes the artifact's shape model, not loaded pretrained weights.",
                                         "Artifact declares GQA=4 but its model.py does not implement GQA sharing.",
                                         "No tokens, KV tensor values, LongBench answers or perplexity are evaluated."]}}


def validate_platform(platform):
    _fields(platform, set(DEFAULT_PLATFORM), "STARC platform")
    if platform["kind"] != "starc-system/v1" or platform["system"] != "dgx-attacc" or platform["pim"] != "bank":
        raise ValueError("STARC v1 supports the dgx-attacc bank platform")
    if platform["gpu"] not in ("H100", "A100a") or type(platform["ngpu"]) is not int or platform["ngpu"] not in (1, 2, 4, 8):
        raise ValueError("unsupported STARC GPU model or device count")
    if any(type(platform[key]) is not bool for key in ("powerlimit", "ffopt", "pipeopt")):
        raise ValueError("STARC optimization and power flags must be booleans")
    return dict(platform)


def validate_runtime(path):
    """Verify the descriptor's complete, pinned source and runtime bindings."""
    descriptor = read_json(Path(path))
    if descriptor.get("schema") != "pim-foundry.starc-runtime.v1" or descriptor.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("unsupported STARC runtime schema or source revision")
    required = {"source_root", "python", "executable", "shared_library", "source_files", "runtime_files", "patches", "validation"}
    if not required <= set(descriptor):
        raise ValueError("STARC runtime descriptor is missing required bindings")
    source = Path(descriptor["source_root"]).resolve()
    bindings = descriptor["source_files"]
    if type(bindings) is not dict:
        raise ValueError("source_files must map repository-relative paths to hashes")
    minimum = {SOURCE_PREFIX + "main.py", MODEL_DEFINITION, SOURCE_PREFIX + "src/model.py",
               SOURCE_PREFIX + "src/system.py", SOURCE_PREFIX + "src/devices.py",
               SOURCE_PREFIX + "src/type.py", SOURCE_PREFIX + "src/ramulator_wrapper.py",
               SOURCE_PREFIX + "pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py"}
    if not minimum <= set(bindings):
        raise ValueError("runtime is missing application or trace-generator source hashes")
    for name, expected in bindings.items():
        relative = Path(name)
        actual = source / relative
        if (relative.is_absolute() or ".." in relative.parts or not actual.resolve().is_relative_to(source)
                or not _hash(expected) or not actual.is_file() or _sha(actual) != expected):
            raise ValueError("STARC source binding mismatch: " + str(name))
    runtimes = descriptor["runtime_files"]
    if type(runtimes) is not dict:
        raise ValueError("runtime_files must map absolute paths to hashes")
    for key in ("python", "executable", "shared_library"):
        runtime = descriptor[key]
        if type(runtime) is not str or not Path(runtime).is_absolute() or runtime not in runtimes:
            raise ValueError("missing absolute runtime binding: " + key)
    for name, expected in runtimes.items():
        if not Path(name).is_absolute() or not _hash(expected) or not Path(name).is_file() or _sha(name) != expected:
            raise ValueError("STARC runtime binding mismatch: " + str(name))
    if type(descriptor["patches"]) is not list or type(descriptor["validation"]) is not dict:
        raise ValueError("runtime must disclose patches and validation")
    for patch in descriptor["patches"]:
        if (type(patch) is not dict or not {"description", "path", "sha256"} <= set(patch)
                or not _hash(patch["sha256"]) or _sha(patch["path"]) != patch["sha256"]):
            raise ValueError("STARC runtime patch binding mismatch")
    return descriptor


def _read_csv(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV requires distinct column names")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV row width differs from header")
    return rows


def parse_csv(path, *, model, lin, lout, batch):
    """Parse exact upstream units; generation values are per batch-token step."""
    rows = _read_csv(path)
    if len(rows) != 1:
        raise ValueError("STARC output CSV must contain exactly one experiment row")
    row = rows[0]
    for key, value in (("model", model), ("Lin", str(lin)), ("Lout", str(lout)), ("bs", str(batch)), ("dtype", "W16A16")):
        if row.get(key) != value:
            raise ValueError("STARC output does not match requested " + key)
    metrics = {}
    for name, column, unit, scope, aggregation in (
        ("prefill_latency", "s_time", "ms", "artifact_prefill_fixed_batch", "sum_over_model_layers"),
        ("decode_latency", "g_time (ms)", "ms", "artifact_decode_per_batch_token", "mean_over_generated_tokens_minus_one_steps"),
        ("decode_energy", "g_energy (nJ)", "nJ", "artifact_decode_per_batch_token", "mean_over_generated_tokens_minus_one_steps"),
    ):
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("missing or invalid STARC metric: " + column) from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError("STARC metrics must be finite and positive")
        metrics[name] = {"value": value, "unit": unit, "scope": scope, "aggregation": aggregation,
                         "origin": "upstream_system_model_csv:" + column}
    return {"metrics": metrics, "raw_row": row, "decode_steps": lout - 1}


def parse_cache(path, *, lin, lout, batch, ngpu, powerlimit):
    rows = _read_csv(path)
    expected_lengths = set(range(lin + 1, lin + lout))
    if len(rows) != len(expected_lengths):
        raise ValueError("native cache does not cover every decode step exactly once")
    lengths = set()
    for row in rows:
        try:
            length = int(row["L"])
            counts = {key: int(row[key]) for key in ("cycle", "mac", "softmax", "mvgb", "mvsb", "wrgb")}
            nhead, dhead, dbyte = (int(row[key]) for key in ("nhead", "dhead", "dbyte"))
        except (KeyError, ValueError) as error:
            raise ValueError("invalid native cache values") from error
        if (length in lengths or counts["cycle"] <= 0 or any(value < 0 for value in counts.values())
                or nhead != math.ceil((32 // ngpu) * batch / 5) or dhead != 128 or dbyte != 2
                or row.get("pim_type") != "BA" or row.get("power_constraint") != str(powerlimit)):
            raise ValueError("native cache identity or cycle/count validation failed")
        lengths.add(length)
    if lengths != expected_lengths:
        raise ValueError("native cache lengths differ from requested scenario")
    return {"rows": len(rows), "first_length": min(lengths), "last_length": max(lengths),
            "cache_reused_from_other_runs": False, "sha256": _sha(path)}


def verify_native_evidence(artifact):
    """Match every fresh cache row to preserved native counters and trace bytes."""
    artifact = Path(artifact)
    rows = {int(row["L"]): row for row in _read_csv(artifact / "ramulator.out")}
    calls = sorted((artifact / "ramulator2/calls").glob("*.json"))
    if len(calls) != len(rows) or not calls:
        raise ValueError("native call count differs from cache rows")
    covered = set()
    names = {"cycle": "memory_system_cycles", "softmax": "total_num_pim_softmax_requests",
             "mvgb": "total_num_pim_move_to_gemv_buffer_requests",
             "mvsb": "total_num_pim_move_to_softmax_buffer_requests",
             "wrgb": "total_num_pim_write_to_gemv_buffer_requests"}
    for path in calls:
        record = read_json(path)
        if record.get("returncode") != 0:
            raise ValueError("native call failed")
        match = re.search(r"attacc_l([0-9]+)_kv", path.with_suffix(".yaml").read_text())
        if not match or int(match[1]) not in rows or int(match[1]) in covered:
            raise ValueError("native YAML length does not identify exactly one cache row")
        length = int(match[1])
        covered.add(length)
        digest = hashlib.sha256()
        with gzip.open(path.with_suffix(".trace.gz"), "rb") as stream:
            for chunk in iter(lambda: stream.read(1048576), b""):
                digest.update(chunk)
        if digest.hexdigest() != record.get("trace_sha256"):
            raise ValueError("preserved native trace hash differs")
        stdout = path.with_suffix(".stdout").read_text()
        counts = {}
        for key, name in names.items():
            values = re.findall(r"(?m)^\s*" + re.escape(name) + r":\s*([0-9]+)\s*$", stdout)
            if len(values) != 1:
                raise ValueError("native output requires exactly one counter: " + name)
            counts[key] = int(values[0])
        values = re.findall(r"(?m)^\s*total_num_pim_mac_(?:same_bank|all_bank|per_bank)_requests:\s*([0-9]+)\s*$", stdout)
        if len(values) != 3:
            raise ValueError("native output is missing MAC counters")
        counts["mac"] = sum(map(int, values))
        if any(counts[key] != int(rows[length][key]) for key in counts):
            raise ValueError("native statistics differ from cached statistics")
    return {"passed": True, "calls_verified": len(calls),
            "checks": ["native_exit_codes", "trace_content_hashes", "one_call_per_decode_step", "all_native_counts_match_cache"]}


def _budget(path, sha, steps):
    path = Path(path)
    if not _hash(sha) or _sha(path) != sha:
        raise ValueError("KV budget table hash mismatch")
    values = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            value = int(text)
        except ValueError as error:
            raise ValueError("KV budget table must contain one integer per row") from error
        _integer(value, 1, 65536, "KV row budget")
        values.append(value)
    if len(values) < steps:
        raise ValueError("KV budget table is shorter than generated_tokens; implicit upstream repetition is forbidden")
    return values


def inspect(prepared, experiment):
    """Check bindings and scenario without creating files or launching a process."""
    try:
        if prepared.get("adapter") != ADAPTER or experiment.get("mapping") != {"strategy": "artifact_native"}:
            raise ValueError("STARC requires its prepared application and artifact_native mapping")
        platform = validate_platform(experiment.get("platform"))
        variant = experiment.get("variant")
        if (type(variant) is not dict or not {"runtime_descriptor", "method"} <= set(variant)
                or set(variant) - {"runtime_descriptor", "method", "kv_budget_table", "timeout_seconds"}):
            raise ValueError("invalid STARC variant fields")
        if variant["method"] not in ("full", "starc"):
            raise ValueError("STARC adapter supports full and starc variants")
        _integer(variant.get("timeout_seconds", 120), 1, 86400, "timeout_seconds")
        runtime = validate_runtime(variant["runtime_descriptor"])
        if runtime["source_files"][MODEL_DEFINITION] != prepared["model_definition_sha256"]:
            raise ValueError("runtime model definition differs from application asset")
        if variant["method"] == "starc":
            _fields(variant.get("kv_budget_table"), {"path", "sha256"}, "STARC budget binding")
            budget = variant["kv_budget_table"]
            _budget(budget["path"], budget["sha256"], prepared["scenario"]["generated_tokens"])
        elif "kv_budget_table" in variant:
            raise ValueError("full attention must not bind a sparse KV budget")
        return {"passed": True, "execution_mode": "original_artifact_system_model",
                "platform": platform, "runtime_validation": runtime["validation"],
                "diagnostics": prepared["diagnostics"], "native_workspace": "isolated_empty_cache_per_execution"}
    except (ValueError, KeyError, OSError, TypeError, SyntaxError) as error:
        return {"passed": False, "reasons": [str(error)], "execution_mode": "original_artifact_system_model"}


# Executed in a private directory. Keep the upstream wrapper unchanged while
# retaining native output and compressed traces before its cleanup commands.
_NATIVE_LAUNCHER = '''#!/usr/bin/env python
import gzip, hashlib, json, os, pathlib, shutil, subprocess, sys
root = pathlib.Path(__file__).resolve().parent
records = root / "calls"
records.mkdir(exist_ok=True)
index = len(list(records.glob("*.json")))
prefix = records / f"call-{index:05d}"
configuration = pathlib.Path(sys.argv[sys.argv.index("-f") + 1])
shutil.copyfile(configuration, str(prefix) + ".yaml")
trace = None
for line in configuration.read_text().splitlines():
    if line.strip().startswith("path:"):
        trace = pathlib.Path(line.split(":", 1)[1].strip().strip('"'))
        break
if trace is None or not trace.is_file():
    raise RuntimeError("native trace missing")
trace_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
with trace.open("rb") as source, gzip.GzipFile(str(prefix) + ".trace.gz", "wb", mtime=0) as target:
    shutil.copyfileobj(source, target)
command = [str(root / "native_ramulator2"), *sys.argv[1:]]
result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
pathlib.Path(str(prefix) + ".stdout").write_bytes(result.stdout)
pathlib.Path(str(prefix) + ".stderr").write_bytes(result.stderr)
pathlib.Path(str(prefix) + ".json").write_text(json.dumps({"command": command, "returncode": result.returncode, "trace_sha256": trace_sha}) + "\\n")
sys.stdout.buffer.write(result.stdout)
sys.stderr.buffer.write(result.stderr)
sys.exit(result.returncode)
'''


_ENVIRONMENT_PROBE = '''import hashlib, importlib.metadata, json, pathlib, sys
expected = json.loads(sys.argv[1])
packages = {}
for requirement in expected:
    name, wanted = requirement.split("==")
    distribution = importlib.metadata.distribution(name)
    if distribution.version != wanted:
        raise RuntimeError("installed package version differs: " + requirement)
    files = {}
    for entry in distribution.files or ():
        path = pathlib.Path(distribution.locate_file(entry))
        if path.is_file() and not str(entry).endswith((".pyc", ".pyo")):
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1048576), b""):
                    h.update(chunk)
            files[str(entry)] = h.hexdigest()
    if not files:
        raise RuntimeError("package has no auditable files: " + name)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    packages[name] = {"version": distribution.version, "file_count": len(files), "files_sha256": digest}
print(json.dumps({"python_version": sys.version, "packages": packages}, sort_keys=True))
'''


def _environment_identity(runtime, environment):
    packages = runtime.get("python_packages")
    if (type(packages) is not list or not packages or len(set(packages)) != len(packages)
            or any(type(item) is not str or item.count("==") != 1 for item in packages)):
        raise ValueError("runtime must pin Python package versions")
    command = [runtime["python"], "-I", "-c", _ENVIRONMENT_PROBE, json.dumps(packages)]
    result = subprocess.run(command, env=environment, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    if result.returncode:
        raise ValueError("runtime Python package validation failed: " + result.stderr[-2000:])
    return json.loads(result.stdout)


def execute(prepared, experiment, output):
    """Execute an application binding; preserve logs and declare evidence scope."""
    if prepared.get("adapter") != ADAPTER:
        raise ValueError("prepared application is not a STARC application")
    if experiment.get("mapping") != {"strategy": "artifact_native"}:
        raise ValueError("STARC artifact controls its mapping; use artifact_native")
    platform = validate_platform(experiment.get("platform"))
    variant = experiment.get("variant")
    if type(variant) is not dict or not {"runtime_descriptor", "method"} <= set(variant) or set(variant) - {"runtime_descriptor", "method", "kv_budget_table", "timeout_seconds"}:
        raise ValueError("invalid STARC variant fields")
    if variant["method"] not in ("full", "starc"):
        raise ValueError("STARC adapter supports full and starc variants")
    timeout = _integer(variant.get("timeout_seconds", 120), 1, 86400, "timeout_seconds")
    descriptor_path = Path(variant["runtime_descriptor"]).resolve()
    descriptor_sha = _sha(descriptor_path)
    runtime = validate_runtime(descriptor_path)
    if runtime["source_files"][MODEL_DEFINITION] != prepared["model_definition_sha256"]:
        raise ValueError("runtime model definition differs from application asset")
    scenario = prepared["scenario"]
    lin, lout, batch = scenario["prompt_tokens"], scenario["generated_tokens"], scenario["batch_size"]
    budget = None
    if variant["method"] == "starc":
        _fields(variant.get("kv_budget_table"), {"path", "sha256"}, "STARC budget binding")
        budget = variant["kv_budget_table"]
        _budget(budget["path"], budget["sha256"], lout)
    elif "kv_budget_table" in variant:
        raise ValueError("full attention must not bind a sparse KV budget")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("STARC output already exists")
    output.mkdir(parents=True)
    _save(output / "runtime.json", runtime)
    _save(output / "binding.json", experiment)
    source = Path(runtime["source_root"])
    before = {"descriptor": descriptor_sha, "source_files": runtime["source_files"], "runtime_files": runtime["runtime_files"]}
    with tempfile.TemporaryDirectory(prefix="pim-foundry-starc-") as temporary:
        work = Path(temporary)
        # Only descriptor-bound files are copied, never ambient source/cache.
        for relative in runtime["source_files"]:
            if relative == SOURCE_PREFIX + "main.py" or relative.startswith(SOURCE_PREFIX + "src/"):
                destination = work / relative.removeprefix(SOURCE_PREFIX)
            elif relative.startswith(SOURCE_PREFIX + "pim_ramulator_src/trace_gen/"):
                destination = work / "ramulator2/trace_gen" / Path(relative).name
            else:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
        native = work / "ramulator2"
        native.mkdir(exist_ok=True)
        shutil.copyfile(runtime["executable"], native / "native_ramulator2")
        (native / "native_ramulator2").chmod(0o755)
        shutil.copyfile(runtime["shared_library"], native / "libramulator.so")
        (native / "ramulator2").write_text(_NATIVE_LAUNCHER)
        (native / "ramulator2").chmod(0o755)
        if budget:
            shutil.copyfile(budget["path"], work / "budget.txt")
        command = [runtime["python"], "main.py", "--system", platform["system"], "--gpu", platform["gpu"],
                   "--ngpu", str(platform["ngpu"]), "--model", prepared["model"], "--lin", str(lin),
                   "--lout", str(lout), "--batch", str(batch), "--pim", platform["pim"]]
        command += ["--" + key for key in ("powerlimit", "ffopt", "pipeopt") if platform[key]]
        if budget:
            command += ["--sparsity", "--kv_budget_table", "budget.txt"]
        environment = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_AUDIT"):
            environment.pop(key, None)
        environment["PATH"] = str(Path(runtime["python"]).parent) + os.pathsep + environment.get("PATH", "")
        environment["LD_LIBRARY_PATH"] = str(native)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONHASHSEED"] = "0"
        observed_environment = _environment_identity(runtime, environment)
        _save(work / "environment.json", observed_environment)
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=work, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        wall = time.monotonic() - started
        (work / "stdout.log").write_bytes(stdout)
        (work / "stderr.log").write_bytes(stderr)
        if not timed_out and process.returncode == 0 and _environment_identity(runtime, environment) != observed_environment:
            raise ValueError("runtime Python packages changed during execution")
        _save(work / "process.json", {"command": command, "returncode": process.returncode,
                                      "timed_out": timed_out, "host_wall_seconds_not_device_latency": wall})
        shutil.copytree(work, output / "artifact")
    if timed_out or process.returncode:
        _save(output / "failure.json", {"reason": "timeout" if timed_out else "upstream_process_failed", "returncode": process.returncode})
        raise ValueError("STARC artifact execution failed; preserved raw logs in " + str(output))
    after = validate_runtime(descriptor_path)
    if _sha(descriptor_path) != before["descriptor"] or after["source_files"] != before["source_files"] or after["runtime_files"] != before["runtime_files"]:
        raise ValueError("STARC runtime changed during execution")
    if budget and _sha(budget["path"]) != budget["sha256"]:
        raise ValueError("KV budget changed during execution")
    artifact = output / "artifact"
    parsed = parse_csv(artifact / "output.csv", model=prepared["model"], lin=lin, lout=lout, batch=batch)
    cache = parse_cache(artifact / "ramulator.out", lin=lin, lout=lout, batch=batch, ngpu=platform["ngpu"], powerlimit=platform["powerlimit"])
    calls = sorted((artifact / "ramulator2/calls").glob("*.json"))
    if len(calls) != lout - 1 or any(read_json(path).get("returncode") != 0 for path in calls):
        raise ValueError("missing or failed native simulator calls")
    for path in calls:
        stdout_path = path.with_suffix(".stdout")
        if not stdout_path.is_file() or "memory_system_cycles" not in stdout_path.read_text():
            raise ValueError("native simulator did not report cycle statistics")
    native_evidence = verify_native_evidence(artifact)
    runtime_identity = {"source_commit": runtime["source_commit"], "source_files": runtime["source_files"],
                        "binary_sha256": _sha(runtime["executable"]), "library_sha256": _sha(runtime["shared_library"]),
                        "python_sha256": _sha(runtime["python"]),
                        "python_environment": observed_environment,
                        "patches": [{"description": p["description"], "sha256": p["sha256"]} for p in runtime["patches"]]}
    cost_contract = {"name": "starc-original-system-model/v1", "runtime": runtime_identity,
                     "precision": "upstream_W16A16_shape_model", "native_clock_ns": 0.769,
                     "scope": "upstream_fixed_batch_prefill_and_decode_layer_model",
                     "decode_aggregation": "mean_of_lout_minus_one_batch_token_steps_times_model_layers",
                     "energy": "upstream_analytical_energy_tables_and_native_command_traffic",
                     "gqa": "declared_4_but_not_applied_by_upstream_model",
                     "validation": runtime["validation"], "hardware_calibrated_by_this_run": False}
    result = {"status": "completed", "execution_mode": "original_artifact_system_model",
              "metrics": parsed["metrics"], "platform_identity": platform, "cost_contract": cost_contract,
              "execution_identity": {"adapter": ADAPTER, "runtime": runtime_identity, "method": variant["method"],
                                     "runtime_descriptor_sha256": descriptor_sha,
                                     "kv_budget_sha256": budget["sha256"] if budget else None},
              "quality": {"numerical": {"status": "not_evaluated", "reason": "native artifact models costs, not tensor values"},
                          "task": {"status": "not_evaluated", "reason": "no pretrained model or quality dataset was executed"}},
              "diagnostics": {**prepared["diagnostics"], "cache": cache, "native_calls": len(calls),
                              "native_evidence": native_evidence,
                              "runtime_validation": runtime["validation"]},
              "native_timing_executed": True, "native_arithmetic": False, "lowered_to_common_ir": False,
              "hardware_execution": False, "quality_preserving_speedup_eligible": False,
              "raw_csv_row": parsed["raw_row"], "decode_steps": parsed["decode_steps"]}
    _save(output / "result.json", result)
    _save(output / "manifest.json", {"schema_version": 1,
                                    "files": {str(path.relative_to(output)): _sha(path)
                                              for path in sorted(output.rglob("*")) if path.is_file()}})
    return result

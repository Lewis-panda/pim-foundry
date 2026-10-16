"""Bind supplied measurement artifacts to predictions and test held-out error.

This evaluator fits no parameters and does not authenticate hardware, protocols,
or log contents. A passing report means the supplied, hash-bound latency records
meet their declared threshold on disjoint workload identities.
"""

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

from .benchmark import BenchmarkResult, content_hash
from .config import exact_fields, platform_from_dict
from .simulation import MAX_REQUESTS, MAX_TIME_NS, validate_model


_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_SAMPLES = 1024


def _version(value, label):
    if type(value) is not int or value != 1:
        raise ValueError(f"{label} schema_version must be 1")


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


def _digest(value, label):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _number(value, label, positive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}, excluding bool")
    return Fraction(str(value))


def _parse_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON numeric literal exceeds finite range")
        return number

    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=reject, parse_float=finite_float)


def _relative_file(root, name):
    _text(name, "artifact path")
    path = Path(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")) or "\\" in name:
        raise ValueError("artifact paths must be canonical relative paths without traversal")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("artifact paths must not contain symlinks")
    if not current.resolve().is_relative_to(root) or not current.is_file():
        raise ValueError("artifact must be a regular file inside the dataset directory")
    return current


def _placement(value, label):
    if type(value) is not dict or not value:
        raise ValueError(f"{label} must be a nonempty placement object")
    for name, device in value.items():
        _text(name, label)
        _text(device, label)


def _request_parameters(value, label):
    arrivals = value["arrivals_ns"]
    if (type(arrivals) is not list or not 1 <= len(arrivals) <= MAX_REQUESTS
            or any(type(x) is not int or not 0 <= x <= MAX_TIME_NS for x in arrivals)
            or arrivals != sorted(arrivals)):
        raise ValueError(f"{label} arrivals_ns must be sorted signed-int64 nonnegative integer times")
    inflight = value["max_inflight"]
    if type(inflight) is not int or not 1 <= inflight <= len(arrivals):
        raise ValueError(f"{label} max_inflight must be a valid request count")
    if type(value["initial_state"]) is not str or value["initial_state"] not in ("cold", "warm"):
        raise ValueError(f"{label} initial_state must be cold or warm")


def _prediction(value):
    exact_fields(value, ("schema_version", "workload_sha256", "config_sha256", "model_sha256",
                         "functional", "simulation"), "prediction")
    _version(value["schema_version"], "prediction")
    for key in ("workload_sha256", "config_sha256", "model_sha256"):
        _digest(value[key], key)
    functional, simulation = value["functional"], value["simulation"]
    exact_fields(functional, ("schema_version", "status", "outputs", "plan", "kernel_events", "validation"),
                 "functional prediction")
    _version(functional["schema_version"], "functional prediction")
    if functional["status"] != "functional_reference_verified":
        raise ValueError("prediction must contain verified functional execution")
    expected_validation = {"matches_scalar_oracle": True, "logical_resources_checked": True,
                           "vendor_backend_numerics": False, "timing_simulation": False,
                           "hardware_execution": False}
    if (type(functional["validation"]) is not dict
            or functional["validation"] != expected_validation
            or any(type(flag) is not bool for flag in functional["validation"].values())):
        raise ValueError("prediction functional validation flags are invalid")
    required = {"schema_version", "status", "evidence", "calibrated", "model", "plan", "arrivals_ns",
                "max_inflight", "initial_state", "requests", "completion_ns", "latency_ns", "makespan_ns"}
    if type(simulation) is not dict or not required <= set(simulation):
        raise ValueError("prediction must contain an execute_case simulation artifact")
    _version(simulation["schema_version"], "simulation prediction")
    if (simulation["status"] != "analytical_simulation" or simulation["evidence"] != "analytical"
            or simulation["calibrated"] is not False):
        raise ValueError("prediction must contain an uncalibrated analytical simulation")
    if functional["plan"] != simulation["plan"] or type(simulation["plan"]) is not dict:
        raise ValueError("functional and simulation plans disagree")
    recorded_plan = simulation["plan"]
    if not {"platform", "placement"} <= set(recorded_plan):
        raise ValueError("prediction plan is missing platform or placement")
    platform = platform_from_dict(recorded_plan["platform"])
    _placement(recorded_plan["placement"], "prediction placement")
    if any(name not in {device.name for device in platform.devices} for name in recorded_plan["placement"].values()):
        raise ValueError("prediction placement names unknown devices")
    validate_model(platform, simulation["model"])
    if content_hash(simulation["model"]) != value["model_sha256"]:
        raise ValueError("prediction model_sha256 does not match its embedded model")
    _request_parameters(simulation, "prediction")
    count = len(simulation["arrivals_ns"])
    for key in ("latency_ns", "completion_ns"):
        times = simulation[key]
        if (type(times) is not list or len(times) != count
                or any(type(time) is not int or not 0 <= time <= MAX_TIME_NS for time in times)):
            raise ValueError(f"prediction {key} must match the request count with integer times")
    if any(latency <= 0 or latency != complete - arrival for latency, complete, arrival in
           zip(simulation["latency_ns"], simulation["completion_ns"], simulation["arrivals_ns"])):
        raise ValueError("prediction latencies disagree with completion minus arrival")
    requests = simulation["requests"]
    if type(requests) is not list or len(requests) != count:
        raise ValueError("prediction requests must match the arrival count")
    for index, request in enumerate(requests):
        exact_fields(request, ("request", "arrival_ns", "completion_ns", "latency_ns"), "prediction request")
        expected = {"request": index, "arrival_ns": simulation["arrivals_ns"][index],
                    "completion_ns": simulation["completion_ns"][index], "latency_ns": simulation["latency_ns"][index]}
        if request != expected or any(type(item) is not int for item in request.values()):
            raise ValueError("prediction request records disagree with latency arrays")
    if (type(simulation["makespan_ns"]) is not int
            or simulation["makespan_ns"] != max(simulation["completion_ns"]) - simulation["arrivals_ns"][0]):
        raise ValueError("prediction makespan disagrees with request completion")
    return simulation


def _median(values):
    values = sorted(Fraction(str(value)) for value in values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def _report_number(value):
    if value.denominator == 1:
        result = value.numerator
    else:
        try:
            result = float(value)
        except OverflowError as exc:
            raise ValueError("derived validation statistic exceeds finite report range") from exc
    try:
        finite = math.isfinite(result)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("derived validation statistic exceeds finite report range")
    return result


def evaluate_validation(dataset_path: Path) -> dict:
    """Check supplied hardware-measurement records against held-out predictions.

    All referenced paths are relative to the dataset's directory. No file is
    written, no parameters are fit, and measurements/logs remain unauthenticated.
    """
    dataset_path = Path(dataset_path)
    if dataset_path.is_symlink() or not dataset_path.is_file():
        raise ValueError("dataset must be a regular file, not a symlink")
    root = dataset_path.parent.resolve()
    sources, seen_paths, raw_digests = {}, set(), set()

    def read_source(name, parse=True):
        path = _relative_file(root, name)
        canonical = path.relative_to(root).as_posix()
        if canonical in seen_paths:
            raise ValueError(f"duplicate artifact path: {canonical}")
        seen_paths.add(canonical)
        if parse:
            with path.open("rb") as stream:
                data = stream.read(_MAX_JSON_BYTES + 1)
            if len(data) > _MAX_JSON_BYTES:
                raise ValueError("JSON input exceeds the 32 MiB limit")
            digest = hashlib.sha256(data).hexdigest()
            value = _parse_json(data)
        else:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest, value = hasher.hexdigest(), None
        sources[canonical] = digest
        return value, digest

    dataset, _ = read_source(dataset_path.name)
    exact_fields(dataset, ("schema_version", "description", "max_relative_error", "samples"), "validation dataset")
    _version(dataset["schema_version"], "dataset")
    _text(dataset["description"], "dataset description")
    threshold = _number(dataset["max_relative_error"], "max_relative_error")
    if threshold > 1:
        raise ValueError("max_relative_error must be in [0,1]")
    if type(dataset["samples"]) is not list or not 4 <= len(dataset["samples"]) <= _MAX_SAMPLES:
        raise ValueError(f"dataset requires 4 to {_MAX_SAMPLES} samples")
    workloads = {"calibration": set(), "validation": set()}
    errors = {"calibration": [], "validation": []}
    sample_reports = []
    fixed_model = fixed_config = None
    for index, sample in enumerate(dataset["samples"]):
        exact_fields(sample, ("split", "prediction_file", "measurement_file"), "validation sample")
        split = sample["split"]
        if type(split) is not str or split not in workloads:
            raise ValueError("sample split must be calibration or validation")
        prediction, prediction_digest = read_source(sample["prediction_file"])
        simulation = _prediction(prediction)
        measurement, measurement_digest = read_source(sample["measurement_file"])
        exact_fields(measurement, ("schema_version", "evidence", "workload_sha256", "config_sha256", "placement",
                                  "arrivals_ns", "max_inflight", "initial_state", "scope", "latency_ns",
                                  "provenance", "raw_log"), "measurement")
        _version(measurement["schema_version"], "measurement")
        if measurement["evidence"] != "hardware_measurement" or measurement["scope"] != "device_graph":
            raise ValueError("measurement must declare hardware_measurement evidence and device_graph scope")
        for key in ("workload_sha256", "config_sha256"):
            _digest(measurement[key], f"measurement {key}")
            if measurement[key] != prediction[key]:
                raise ValueError(f"measurement {key} does not match prediction")
        _placement(measurement["placement"], "measurement placement")
        if measurement["placement"] != simulation["plan"]["placement"]:
            raise ValueError("measurement placement does not match prediction")
        _request_parameters(measurement, "measurement")
        for key in ("arrivals_ns", "max_inflight", "initial_state"):
            if measurement[key] != simulation[key]:
                raise ValueError(f"measurement {key} does not match prediction")
        latency = measurement["latency_ns"]
        if type(latency) is not list or len(latency) != len(simulation["arrivals_ns"]):
            raise ValueError("measurement latency_ns must match the prediction request count")
        for value in latency:
            _number(value, "measurement latency_ns", positive=True)
        exact_fields(measurement["provenance"], ("hardware", "software", "method"), "measurement provenance")
        for label, value in measurement["provenance"].items():
            _text(value, f"measurement provenance {label}")
        exact_fields(measurement["raw_log"], ("path", "sha256"), "measurement raw_log")
        _digest(measurement["raw_log"]["sha256"], "raw log sha256")
        _, raw_digest = read_source(measurement["raw_log"]["path"], parse=False)
        if raw_digest != measurement["raw_log"]["sha256"]:
            raise ValueError("raw log SHA-256 mismatch")
        if raw_digest in raw_digests:
            raise ValueError("raw log content is reused across samples")
        raw_digests.add(raw_digest)
        if fixed_model is None:
            fixed_model, fixed_config = prediction["model_sha256"], prediction["config_sha256"]
        if prediction["model_sha256"] != fixed_model or prediction["config_sha256"] != fixed_config:
            raise ValueError("all predictions must use the same model and recorded configuration")
        workloads[split].add(prediction["workload_sha256"])
        predicted, measured = _median(simulation["latency_ns"]), _median(latency)
        relative_error = abs(predicted - measured) / measured
        errors[split].append(relative_error)
        sample_reports.append({
            "sample": index, "split": split, "workload_sha256": prediction["workload_sha256"],
            "prediction_file": sample["prediction_file"], "prediction_sha256": prediction_digest,
            "measurement_file": sample["measurement_file"], "measurement_sha256": measurement_digest,
            "raw_log": measurement["raw_log"], "provenance": measurement["provenance"],
            "placement": measurement["placement"], "arrivals_ns": measurement["arrivals_ns"],
            "max_inflight": measurement["max_inflight"], "initial_state": measurement["initial_state"],
            "predicted_median_latency_ns": _report_number(predicted),
            "measured_median_latency_ns": _report_number(measured),
            "absolute_relative_error": _report_number(relative_error), "within_threshold": relative_error <= threshold,
        })
    if workloads["calibration"] & workloads["validation"]:
        raise ValueError("calibration and validation workloads must be disjoint")
    if any(len(identities) < 2 for identities in workloads.values()):
        raise ValueError("each split requires at least two distinct workload identities")
    protocol = {
        "schema_version": 1, "id": "pim-foundry/held-out-latency-validation/v1",
        "scope": "device_graph", "metric": "median_request_latency_ns",
        "error": "abs(predicted_median-measured_median)/measured_median",
        "max_relative_error": _report_number(threshold),
        "split_policy": "disjoint workload_sha256; at least two distinct identities per split",
        "pass_rule": "every validation sample is within the inclusive relative-error threshold",
    }
    summaries = {
        split: {"sample_count": len(values), "distinct_workloads": len(workloads[split]),
                "mean_absolute_relative_error": _report_number(sum(values) / len(values)),
                "max_absolute_relative_error": _report_number(max(values)),
                "all_within_threshold": all(error <= threshold for error in values)}
        for split, values in errors.items()
    }
    return {
        "schema_version": 1, "status": "held_out_validation_report", "description": dataset["description"],
        "model_sha256": fixed_model, "config_sha256": fixed_config,
        "measurement_protocol_sha256": content_hash(protocol), "measurement_protocol": protocol,
        "dataset_sha256": content_hash({"dataset": dataset, "source_file_sha256": sources}),
        "source_file_sha256": dict(sorted(sources.items())), "samples": sample_reports,
        "split_summaries": summaries, "passed": summaries["validation"]["all_within_threshold"],
        "independently_audited": False, "parameters_fitted": False,
        "limitations": [
            "Hardware/software/protocol declarations and raw-log contents are supplied evidence, not independently authenticated measurements.",
            "The embedded cost-model hash is recomputed; the configuration hash is bound to the prediction's recorded digest, not an independently audited configuration manifest.",
            "Workload hashes include concrete inputs and weights; disjoint hashes do not establish disjoint architecture families or prove the validation split was never used for tuning.",
            "Calibration-split errors are diagnostic only. No parameters are fitted; passing depends only on the held-out validation samples and declared threshold.",
        ],
    }


def qualify_latency(record_path: Path, result_path: Path, dataset_path: Path) -> dict:
    """Attach validation evidence only to the exact, validated median-latency artifact."""
    from .experiments import simulation_contract

    report = evaluate_validation(dataset_path)
    if not report["passed"]:
        raise ValueError("held-out validation failed; latency record cannot be qualified")

    def read_file(path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("qualification inputs must be regular files, not symlinks")
        with path.open("rb") as stream:
            data = stream.read(_MAX_JSON_BYTES + 1)
        if len(data) > _MAX_JSON_BYTES:
            raise ValueError("JSON input exceeds the 32 MiB limit")
        return _parse_json(data), hashlib.sha256(data).hexdigest()

    raw_record, _ = read_file(record_path)
    result, result_digest = read_file(result_path)
    record = BenchmarkResult.from_dict(raw_record)
    simulation = _prediction(result)
    if record.metric != "latency_ns" or record.evidence != "analytical":
        raise ValueError("qualification requires an analytical latency_ns benchmark record")
    if record.contract.statistic != "median" or record.contract.measurement_scope != "device_graph":
        raise ValueError("held-out evidence qualifies only median latency in device_graph scope")
    if record.contract != simulation_contract(result, "latency_ns") or record.quality_value != 1:
        raise ValueError("record contract or numerical quality does not match the prediction producer")
    if record.raw_artifact_sha256 != content_hash(result):
        raise ValueError("benchmark raw artifact hash does not match the supplied result")
    if not any(sample["split"] == "validation" and sample["prediction_sha256"] == result_digest
               for sample in report["samples"]):
        raise ValueError("result file must be an exact prediction artifact from the validation split")
    if result["model_sha256"] != report["model_sha256"] or result["config_sha256"] != report["config_sha256"]:
        raise ValueError("result model/configuration do not match the validation report")
    if record.system_manifest_sha256 != result["config_sha256"]:
        raise ValueError("record configuration digest does not match the result")
    if record.contract.workload_sha256 != result["workload_sha256"]:
        raise ValueError("record workload does not match the result")
    if list(record.samples) != simulation["latency_ns"]:
        raise ValueError("record latency samples do not match the result")
    if (record.contract.request_stream_sha256 != content_hash(simulation["arrivals_ns"])
            or record.contract.initial_state != simulation["initial_state"]):
        raise ValueError("record request stream or initial state does not match the result")
    return replace(record, calibration_protocol_sha256=report["measurement_protocol_sha256"],
                   calibration_artifact_sha256=content_hash(report)).to_dict()

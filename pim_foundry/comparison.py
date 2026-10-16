"""Explicit comparison tracks over sealed application bundles.

Hash binding establishes which claims were compared, not the truth of supplied
resource declarations or measurement logs. Failed evidence is never averaged out.
"""

import hashlib
import math
from pathlib import Path
import random
import statistics
import sys

from . import application, calibration, unified

PROTOCOL_SCHEMA = "pim-foundry.comparison-protocol.v1"
RESOURCE_SCHEMA = "pim-foundry.resource-disclosure.v1"
EVIDENCE_SCHEMA = "pim-foundry.comparison-evidence.v1"
CALIBRATION_SCHEMA = "pim-foundry.metric-calibration.v1"
UNITS = {"cpu_count": "count", "gpu_count": "count", "npu_count": "count", "pim_count": "count",
         "capacity": "byte", "compute": "op/s", "external_bandwidth": "byte/s",
         "internal_bandwidth": "byte/s", "power": "W", "cost": "USD"}
BASES = {"author_reported", "observed", "externally_measured", "model_assumption", "undisclosed"}
KINDS = {"correctness", "replay", "direct_parity", "study_reproducibility", "hardware_measurement", "calibration"}


def _fields(value, fields, label):
    application._fields(value, set(fields), label)


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be nonempty text")


def _number(value, label, positive=False):
    calibration._number(value, label, positive=positive)
    return value


def _read(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ValueError("sources must be regular files without symlink components: " + str(path))
    data = path.read_bytes()
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("JSON source exceeds 32 MiB")
    return calibration._parse_json(data), hashlib.sha256(data).hexdigest()


def _source(root, spec, sources):
    _fields(spec, {"path", "sha256"}, "source")
    _text(spec["path"], "source path")
    calibration._digest(spec["sha256"], "source sha256")
    path = (root / spec["path"]).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ValueError("missing or symlink source: " + str(path))
    digest = application.file_sha256(path)
    if digest != spec["sha256"]:
        raise ValueError("source SHA-256 mismatch: " + str(path))
    sources[str(path.resolve())] = digest
    return path.resolve()


def file_ref(path):
    """Create an immutable content reference; files remain external to the report."""
    path = Path(path).resolve()
    return {"path": str(path), "sha256": application.file_sha256(path)}


def _resource(path, sources):
    value, digest = _read(path)
    if str(path) in sources and sources[str(path)] != digest:
        raise ValueError("resource changed while reading")
    sources[str(path)] = digest
    _fields(value, {"schema", "platform_sha256", "resources", "compute_convention", "cost_basis", "sources"}, "resource disclosure")
    if value["schema"] != RESOURCE_SCHEMA:
        raise ValueError("unsupported resource disclosure")
    calibration._digest(value["platform_sha256"], "resource platform")
    _fields(value["resources"], UNITS, "resources")
    _text(value["compute_convention"], "compute convention")
    _text(value["cost_basis"], "cost basis")
    for name, unit in UNITS.items():
        item = value["resources"][name]
        _fields(item, {"value", "unit", "basis", "description"}, "resource " + name)
        if item["unit"] != unit or item["basis"] not in BASES:
            raise ValueError("resource unit or evidence basis is invalid: " + name)
        _text(item["description"], "resource description")
        if item["value"] is None:
            if item["basis"] != "undisclosed" or name.endswith("_count"):
                raise ValueError("unknown resources must be undisclosed; device counts must be supplied")
        else:
            _number(item["value"], name)
            if item["basis"] == "undisclosed" or (name.endswith("_count") and type(item["value"]) is not int):
                raise ValueError("invalid disclosed resource: " + name)
    if not isinstance(value["sources"], list) or not value["sources"]:
        raise ValueError("resource disclosure requires content-bound sources")
    for spec in value["sources"]:
        _source(path.parent, spec, sources)
    return value


def _statistics_options(value):
    _fields(value, {"method", "seed", "resamples", "confidence", "sampling_unit"}, "statistics")
    if value["method"] not in {"deterministic", "bootstrap_percentile"}:
        raise ValueError("unsupported confidence method")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**64:
        raise ValueError("bootstrap seed must be an unsigned 64-bit integer")
    _number(value["confidence"], "confidence", positive=True)
    if not value["confidence"] < 1:
        raise ValueError("confidence must be below one")
    if type(value["resamples"]) is not int or (value["method"] == "deterministic" and value["resamples"] != 0):
        raise ValueError("deterministic statistics use zero resamples")
    if value["method"] == "bootstrap_percentile" and not 1000 <= value["resamples"] <= 100000:
        raise ValueError("bootstrap requires 1000..100000 resamples")
    if value["sampling_unit"] != "independent_runs":
        raise ValueError("the supported sampling unit is independent_runs")


def inspect(protocol_path):
    """Validate declarations and source hashes without executing a simulator."""
    path = Path(protocol_path).resolve()
    value, digest = _read(path)
    _fields(value, {"schema", "id", "track", "metric", "quality_requirement", "resources", "fixed_axes",
                    "budget_constraints", "cost_model_bindings", "evidence_sha256", "repetitions", "statistics"}, "comparison protocol")
    if value["schema"] != PROTOCOL_SCHEMA or value["track"] not in {"same_platform", "equal_budget", "whole_system"}:
        raise ValueError("unsupported comparison schema or track")
    for key in ("id", "metric"):
        _text(value[key], key)
    if value["quality_requirement"] not in {"numerical", "task"}:
        raise ValueError("quality_requirement must be numerical or task")
    _fields(value["resources"], {"left", "right"}, "resource bindings")
    sources, resources = {str(path): digest}, {}
    for side, ref in value["resources"].items():
        resources[side] = _resource(_source(path.parent, ref, sources), sources)
    axes = value["fixed_axes"]
    if not isinstance(axes, list) or any(x not in UNITS for x in axes) or len(set(axes)) != len(axes):
        raise ValueError("fixed axes must be unique resource names")
    if not isinstance(value["budget_constraints"], list):
        raise ValueError("budget_constraints must be a list")
    seen = set()
    for item in value["budget_constraints"]:
        _fields(item, {"resource", "relation", "value", "relative_tolerance"}, "budget constraint")
        if item["resource"] not in UNITS or item["resource"] in seen or item["relation"] not in {"equal", "at_most"}:
            raise ValueError("invalid or duplicate budget constraint")
        seen.add(item["resource"])
        _number(item["value"], "budget")
        _number(item["relative_tolerance"], "budget tolerance")
        if item["relative_tolerance"] > 0.1:
            raise ValueError("relative budget tolerance must not exceed 0.1")
    if value["track"] == "equal_budget" and not seen:
        raise ValueError("equal_budget requires at least one explicit shared constraint")
    bindings = value["cost_model_bindings"]
    if not isinstance(bindings, list) or len(bindings) not in (0, 2):
        raise ValueError("cost_model_bindings must contain zero or two source references")
    for ref in bindings:
        _source(path.parent, ref, sources)
    evidence = value["evidence_sha256"]
    if not isinstance(evidence, list) or len(set(evidence)) != len(evidence):
        raise ValueError("evidence hashes must be unique")
    for digest in evidence:
        calibration._digest(digest, "evidence digest")
    _fields(value["repetitions"], {"left", "right"}, "repetitions")
    for entries in value["repetitions"].values():
        if not isinstance(entries, list) or len(entries) > 10000:
            raise ValueError("repetitions must be a list of at most 10000 sealed bundles")
        for entry in entries:
            _fields(entry, {"path", "manifest_sha256"}, "repetition")
            _text(entry["path"], "repetition path")
            calibration._digest(entry["manifest_sha256"], "repetition manifest")
    _statistics_options(value["statistics"])
    return {"passed": True, "protocol": value, "resources": resources, "source_file_sha256": sources}


def _percentile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize_samples(values, method="deterministic", seed=0, resamples=0, confidence=0.95):
    """Run-level statistics; percentile intervals resample independent runs.

    The minimum of five is an admission guard, not a claim of adequate power.
    A deterministic simulation has reproducibility checks, not sampling error.
    """
    options = {"method": method, "seed": seed, "resamples": resamples, "confidence": confidence,
               "sampling_unit": "independent_runs"}
    _statistics_options(options)
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 10000:
        raise ValueError("samples require 1..10000 independent-run values")
    for value in values:
        _number(value, "sample")
    result = {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
              "min": min(values), "max": max(values), "p05": _percentile(values, .05),
              "p95": _percentile(values, .95), "method": method, "confidence_interval": None,
              "sampling_unit": "independent_runs", "independence_authenticated": False}
    if method == "deterministic":
        result["reproducibility_passed"] = len(values) >= 2 and len(set(values)) == 1
        result["reason"] = "exact repeat agreement; no statistical confidence interval for deterministic predictions"
    else:
        if len(values) < 5:
            raise ValueError("bootstrap requires at least five independent runs")
        rng = random.Random(seed)
        medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(resamples)]
        alpha = (1 - confidence) / 2
        result["confidence_interval"] = {"statistic": "median", "level": confidence,
                                         "low": _percentile(medians, alpha), "high": _percentile(medians, 1 - alpha),
                                         "method": "iid_percentile_bootstrap", "seed": seed, "resamples": resamples}
    return result


def _bundle(path, sources):
    path = Path(path).resolve()
    application.verify(path)
    result, digest = _read(path / "result.json")
    manifest, _ = _read(path / "application.json")
    experiment, _ = _read(path / "experiment.json")
    if result.get("schema") != application.RESULT_SCHEMA or result.get("adapter") != manifest.get("adapter"):
        raise ValueError("result schema or adapter differs from the verified application")
    sources[str(path / "result.json")] = digest
    sources[str(path / "manifest.json")] = application.file_sha256(path / "manifest.json")
    application._observations(result)
    for key in ("task_sha256", "platform_sha256", "cost_contract_sha256", "execution_sha256"):
        calibration._digest(result.get(key), key)
    if unified.digest(result["platform_identity"]) != result["platform_sha256"] or unified.digest(result["cost_contract"]) != result["cost_contract_sha256"]:
        raise ValueError("result platform/cost digest does not match its embedded declaration")
    identity = {"adapter": manifest["adapter"], "engine": application._engine_identity(path / "engine/pim_foundry"),
                "implementation": result["execution_identity"], "experiment": experiment}
    if unified.digest(identity) != result["execution_sha256"]:
        raise ValueError("execution digest disagrees with the sealed engine, implementation, or experiment")
    return result, digest


def _evidence(path, sources):
    value, digest = _read(path)
    sources[str(path)] = digest
    _fields(value, {"schema", "kind", "basis", "status", "result_sha256", "scope", "sources", "details"}, "comparison evidence")
    if value["schema"] != EVIDENCE_SCHEMA or value["kind"] not in KINDS:
        raise ValueError("unsupported evidence schema or kind")
    if value["basis"] not in {"author_reported", "observed", "externally_measured"} or value["status"] not in {"passed", "failed", "not_evaluated"}:
        raise ValueError("invalid evidence basis or status")
    _text(value["scope"], "evidence scope")
    if not isinstance(value["details"], dict) or not isinstance(value["result_sha256"], list) or not value["result_sha256"]:
        raise ValueError("evidence requires details and exact result bindings")
    for result in value["result_sha256"]:
        calibration._digest(result, "evidence result")
    if not isinstance(value["sources"], list) or not value["sources"]:
        raise ValueError("evidence requires source artifacts")
    for ref in value["sources"]:
        _source(Path(path).parent, ref, sources)
    if value["kind"] == "replay" and value["basis"] != "author_reported":
        _fields(value["details"], {"original_bundle", "replay_bundle", "checks"}, "observed replay evidence")
        expected = replay_evidence(value["details"]["original_bundle"], value["details"]["replay_bundle"], value["scope"])
        for key in ("status", "result_sha256", "details"):
            if value[key] != expected[key]:
                raise ValueError("replay evidence disagrees with verified original and replay bundles")
        if not {ref["sha256"] for ref in expected["sources"]} <= {ref["sha256"] for ref in value["sources"]}:
            raise ValueError("replay evidence must pin both manifests and result files")
    return value, digest


def replay_evidence(original_bundle, replay_bundle, scope):
    """Construct replay evidence by checking both sealed bundles, never a status assertion."""
    sources = {}
    paths = [Path(original_bundle).resolve(), Path(replay_bundle).resolve()]
    if paths[0] == paths[1]:
        raise ValueError("replay must be a distinct run bundle")
    loaded = [_bundle(path, sources) for path in paths]
    first, second = (item[0] for item in loaded)
    fields = ("status", "task_sha256", "execution_sha256", "platform_sha256", "cost_contract_sha256",
              "metrics", "quality", "execution_identity")
    checks = {key: first[key] == second[key] for key in fields}
    return {"schema": EVIDENCE_SCHEMA, "kind": "replay", "basis": "observed",
            "status": "passed" if all(checks.values()) else "failed", "result_sha256": [loaded[0][1]],
            "scope": scope, "sources": [file_ref(path / name) for path in paths for name in ("manifest.json", "result.json")],
            "details": {"original_bundle": str(paths[0]), "replay_bundle": str(paths[1]), "checks": checks}}


def _study_failure(bundle, sources):
    """Conservatively inherit an enclosing study's known reproduction failure.

    This covers the released CIMFlow v0.7 study, whose individual sealed runs
    complete successfully although repeated native timing is inconsistent.
    """
    path = Path(bundle).resolve().parent / "summary.json"
    if not path.is_file():
        return False
    value, digest = _read(path)
    sources[str(path)] = digest
    return isinstance(value, dict) and (value.get("status") == "reproducibility_failed" or
                                       value.get("validation_status") == "reproducibility_failed")


def _cost_bindings(protocol, root, results, sources):
    same_cost = results[0]["cost_contract_sha256"] == results[1]["cost_contract_sha256"]
    if same_cost and not protocol["cost_model_bindings"]:
        return True
    if (protocol["track"] == "same_platform" and not same_cost) or len(protocol["cost_model_bindings"]) != 2:
        return False
    records = []
    for ref, result in zip(protocol["cost_model_bindings"], results):
        path = _source(root, ref, sources)
        binding, digest = _read(path)
        if digest != sources[str(path)]:
            raise ValueError("cost-model binding changed while reading")
        _fields(binding, {"schema", "cost_contract", "metric", "accounting", "rationale"}, "cost-model binding")
        if binding["schema"] != "pim-foundry.cost-model-binding.v1" or binding["cost_contract"] != result["cost_contract"]:
            raise ValueError("cost-model binding must contain the exact result contract")
        metric = result["metrics"].get(protocol["metric"])
        if metric is None or binding["metric"] != {k: v for k, v in metric.items() if k != "value"}:
            raise ValueError("cost-model binding metric does not match the result")
        _fields(binding["accounting"], {"included", "excluded"}, "accounting")
        for values in binding["accounting"].values():
            if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values) or len(values) != len(set(values)):
                raise ValueError("accounting must contain unique text lists")
        if not binding["accounting"]["included"] or set(binding["accounting"]["included"]) & set(binding["accounting"]["excluded"]):
            raise ValueError("accounting needs nonempty, disjoint inclusion/exclusion lists")
        _text(binding["rationale"], "cost-model mapping rationale")
        records.append({key: sorted(values) for key, values in binding["accounting"].items()})
    return records[0] == records[1]


def compare(left_bundle, right_bundle, protocol_path, evidence_paths=()):
    """Compare sealed results; explicit budget/resource disclosure never changes task semantics."""
    inspected = inspect(protocol_path)
    protocol, resources, sources = inspected["protocol"], inspected["resources"], inspected["source_file_sha256"]
    root, reasons, qualification = Path(protocol_path).resolve().parent, [], []
    paths = [Path(left_bundle).resolve(), Path(right_bundle).resolve()]
    loaded = [_bundle(path, sources) for path in paths]
    results, result_hashes = [x[0] for x in loaded], [x[1] for x in loaded]
    for name in ("task_sha256", "execution_mode"):
        if results[0].get(name) != results[1].get(name):
            reasons.append("mismatch:" + name)
    for side, result, path in zip(("left", "right"), results, paths):
        if result["status"] != "completed":
            reasons.append(side + ":unsuccessful_application")
        if any(q["status"] == "failed" for q in result["quality"].values()):
            reasons.append(side + ":quality_failed")
        for kind in (["numerical", "task"] if protocol["quality_requirement"] == "task" else ["numerical"]):
            if result["quality"][kind]["status"] != "passed":
                qualification.append(side + ":" + kind + "_quality_not_passed")
        if resources[side]["platform_sha256"] != result["platform_sha256"]:
            reasons.append(side + ":resource_platform_mismatch")
        experiment, _ = _read(path / "experiment.json")
        configured = experiment.get("platform", {}).get("devices")
        if isinstance(configured, list) and all(isinstance(d, dict) and d.get("kind") in {"cpu", "gpu", "npu", "sram", "hbm"} and type(d.get("capacity_bytes")) is int for d in configured):
            known = {"cpu_count": 0, "gpu_count": 0, "npu_count": 0, "pim_count": 0, "capacity": 0}
            for device in configured:
                key = device["kind"] + "_count" if device["kind"] in {"cpu", "gpu", "npu"} else "pim_count"
                known[key] += 1
                known["capacity"] += device["capacity_bytes"]
            for key, expected in known.items():
                disclosed = resources[side]["resources"][key]["value"]
                if disclosed is not None and disclosed != expected:
                    reasons.append(side + ":resource_disclosure_disagrees_with_executed_config:" + key)
        if _study_failure(path, sources):
            reasons.append(side + ":study_reproducibility_failed")
    if protocol["track"] == "same_platform" and results[0]["platform_sha256"] != results[1]["platform_sha256"]:
        reasons.append("mismatch:platform_sha256")
    if protocol["track"] == "same_platform":
        if any(resources["left"]["resources"][axis]["value"] != resources["right"]["resources"][axis]["value"] for axis in UNITS):
            reasons.append("mismatch:same_platform_resource_disclosure")
    for axis in protocol["fixed_axes"]:
        a, b = (resources[s]["resources"][axis]["value"] for s in ("left", "right"))
        if a is None or b is None or a != b:
            reasons.append("mismatch:fixed_resource:" + axis)
    constrained = set(protocol["fixed_axes"]) | {x["resource"] for x in protocol["budget_constraints"]}
    for axis, convention in (("compute", "compute_convention"), ("cost", "cost_basis")):
        if axis in constrained and resources["left"][convention] != resources["right"][convention]:
            reasons.append("mismatch:" + convention)
    budget_checks = []
    for constraint in protocol["budget_constraints"]:
        for side in ("left", "right"):
            actual = resources[side]["resources"][constraint["resource"]]["value"]
            bound, tolerance = constraint["value"], constraint["relative_tolerance"]
            passed = actual is not None and (abs(actual - bound) <= bound * tolerance if constraint["relation"] == "equal" else actual <= bound * (1 + tolerance))
            budget_checks.append({"side": side, **constraint, "actual": actual, "passed": passed})
            if not passed:
                reasons.append(side + ":resource_budget_failed:" + constraint["resource"])
    metric = protocol["metric"]
    descriptors = [result["metrics"].get(metric) for result in results]
    cost_ok = _cost_bindings(protocol, root, results, sources)
    if not cost_ok:
        reasons.append("mismatch:cost_contract_without_valid_binding")
    if any(item is None for item in descriptors):
        reasons.append("metric_unavailable:" + metric)
    else:
        fields = ("unit", "scope", "aggregation")
        if any(descriptors[0][key] != descriptors[1][key] for key in fields):
            reasons.append("mismatch:metric_unit_scope_or_aggregation")
        if descriptors[0]["origin"] != descriptors[1]["origin"] and (not cost_ok or not protocol["cost_model_bindings"]):
            reasons.append("mismatch:metric_origin_without_valid_binding")
        if descriptors[1]["value"] <= 0:
            reasons.append("nonpositive_denominator")
    evidence = []
    for path in evidence_paths:
        item, digest = _evidence(Path(path).resolve(), sources)
        if digest not in protocol["evidence_sha256"]:
            raise ValueError("evidence attachment is not pinned by the protocol")
        if digest in [entry["sha256"] for entry in evidence]:
            raise ValueError("duplicate evidence attachment")
        evidence.append({"sha256": digest, **item})
    if set(protocol["evidence_sha256"]) != {item["sha256"] for item in evidence}:
        reasons.append("missing_pinned_evidence")
    for side, result_hash, descriptor, result in zip(("left", "right"), result_hashes, descriptors, results):
        relevant = [item for item in evidence if result_hash in item["result_sha256"]]
        for item in relevant:
            if item["status"] == "failed":
                reasons.append(side + ":failed_evidence:" + item["kind"])
        for kind in ("correctness", "replay"):
            if not any(item["kind"] == kind and item["status"] == "passed" and item["basis"] != "author_reported" and
                       descriptor is not None and item["scope"] == descriptor["scope"] for item in relevant):
                qualification.append(side + ":missing_observed_" + kind)
        if result.get("adapter", "").endswith("-artifact/v1") and not any(
                item["kind"] == "study_reproducibility" and item["status"] == "passed" and item["basis"] != "author_reported"
                for item in relevant):
            qualification.append(side + ":missing_observed_study_reproducibility")
    summaries = {}
    for side, path, result in zip(("left", "right"), paths, results):
        values, seen = [], set()
        for entry in protocol["repetitions"][side]:
            sample_path = (root / entry["path"]).resolve()
            if sample_path in seen:
                raise ValueError("the same run cannot count as multiple samples")
            seen.add(sample_path)
            if application.file_sha256(sample_path / "manifest.json") != entry["manifest_sha256"]:
                raise ValueError("repetition manifest SHA-256 mismatch")
            sample, _ = _bundle(sample_path, sources)
            for key in ("task_sha256", "execution_sha256", "platform_sha256", "cost_contract_sha256", "quality", "status"):
                if sample[key] != result[key]:
                    raise ValueError("repetition differs from its base result: " + key)
            if metric not in sample["metrics"] or metric not in result["metrics"] or any(sample["metrics"][metric][k] != result["metrics"][metric][k] for k in ("unit", "scope", "aggregation", "origin")):
                raise ValueError("repetition metric contract differs")
            if _study_failure(sample_path, sources):
                reasons.append(side + ":repetition_study_reproducibility_failed")
            values.append(sample["metrics"][metric]["value"])
        if path not in seen:
            qualification.append(side + ":base_run_missing_from_repetitions")
        if values:
            options = {k: v for k, v in protocol["statistics"].items() if k != "sampling_unit"}
            try:
                summaries[side] = summarize_samples(values, **options)
            except ValueError as exc:
                qualification.append(side + ":insufficient_statistics:" + str(exc))
            if protocol["statistics"]["method"] == "deterministic" and summaries.get(side, {}).get("reproducibility_passed") is not True:
                qualification.append(side + ":deterministic_repeat_not_established")
                if len(set(values)) > 1:
                    reasons.append(side + ":deterministic_reproducibility_failed")
        else:
            qualification.append(side + ":missing_repeat_samples")
    reasons = sorted(set(reasons))
    qualification = sorted(set(qualification))
    ratio = None
    if not reasons:
        a = summaries.get("left", {}).get("median", descriptors[0]["value"])
        b = summaries.get("right", {}).get("median", descriptors[1]["value"])
        if b <= 0:
            reasons.append("nonpositive_median_denominator")
        else:
            ratio = a / b
            if not math.isfinite(ratio):
                reasons.append("ratio_exceeds_finite_range")
                ratio = None
    return {"schema": "pim-foundry.comparison-report.v1", "track": protocol["track"], "protocol_id": protocol["id"],
            "comparable": not reasons, "qualified": not reasons and not qualification,
            "qualification_scope": "declared_" + protocol["quality_requirement"] + "_quality_and_observed_replay_only",
            "blocking_reasons": reasons, "qualification_gaps": qualification, "metric": metric,
            "left_over_right_ratio": ratio, "ratio_statistic": "median_of_run_metrics_when_available_else_single_run",
            "ratio_confidence_interval": None, "metric_contracts": descriptors, "statistics": summaries,
            "resources": resources, "budget_checks": budget_checks, "evidence": evidence,
            "result_sha256": dict(zip(("left", "right"), result_hashes)), "source_file_sha256": dict(sorted(sources.items())),
            "quality_requirement": protocol["quality_requirement"], "hardware_performance_comparison_eligible": False,
            "validator_identity": {"python": sys.version, "module_sha256": {
                name: application.file_sha256(Path(__file__).parent / name)
                for name in ("comparison.py", "application.py", "calibration.py", "unified.py")}},
            "causal_isolated_method_effect": False,
            "limitations": ["Source hashes bind supplied declarations; they do not authenticate hardware, costs, or independence of runs.",
                            "Whole-system ratios include every disclosed resource difference and do not isolate a scheduling effect.",
                            "Bootstrap intervals describe individual run medians; no ratio interval or significance claim is inferred.",
                            "Known study failures and attached failed evidence block comparison even when an individual bundle is intact."]}


def evaluate_calibration(dataset_path):
    """Evaluate generic metric records on disjoint concrete input identities.

    Existing detailed analytical-latency datasets retain their original validator.
    No parameter fitting or independent authentication is performed.
    """
    path = Path(dataset_path).resolve()
    dataset, digest = _read(path)
    if dataset.get("schema_version") == 1 and "samples" in dataset:
        return calibration.evaluate_validation(path)
    _fields(dataset, {"schema", "metric", "model_sha256", "platform_sha256", "model_source", "platform_source", "max_relative_error", "samples"}, "metric calibration")
    if dataset["schema"] != CALIBRATION_SCHEMA:
        raise ValueError("unsupported calibration schema")
    _fields(dataset["metric"], {"name", "unit", "scope", "aggregation"}, "calibration metric")
    for value in dataset["metric"].values():
        _text(value, "calibration metric field")
    for key in ("model_sha256", "platform_sha256"):
        calibration._digest(dataset[key], key)
    _number(dataset["max_relative_error"], "max relative error")
    if not isinstance(dataset["samples"], list) or not 4 <= len(dataset["samples"]) <= 1024:
        raise ValueError("calibration requires 4..1024 samples")
    sources, groups, rows, raw_hashes = {str(path): digest}, {"train": set(), "held_out": set()}, [], set()
    for kind in ("model", "platform"):
        source = _source(path.parent, dataset[kind + "_source"], sources)
        if sources[str(source)] != dataset[kind + "_sha256"]:
            raise ValueError("calibration model/platform source digest does not match its identity")
    for sample in dataset["samples"]:
        _fields(sample, {"split", "prediction", "measurement"}, "calibration sample")
        if sample["split"] not in groups:
            raise ValueError("split must be train or held_out")
        records = []
        for kind in ("prediction", "measurement"):
            source = _source(path.parent, sample[kind], sources)
            record, record_digest = _read(source)
            if record_digest != sources[str(source)]:
                raise ValueError("calibration record changed while reading")
            _fields(record, {"schema", "input_sha256", "input_source", "model_sha256", "platform_sha256", "metric", "value", "basis", "raw_log"}, "metric record")
            if record["schema"] != "pim-foundry.metric-record.v1":
                raise ValueError("unsupported metric record")
            for key in ("input_sha256", "model_sha256", "platform_sha256"):
                calibration._digest(record[key], key)
            input_source = _source(source.parent, record["input_source"], sources)
            if sources[str(input_source)] != record["input_sha256"]:
                raise ValueError("calibration input source digest does not match its identity")
            if any(record[key] != dataset[key] for key in ("model_sha256", "platform_sha256", "metric")):
                raise ValueError("calibration record model, platform, or metric identity mismatch")
            if record["basis"] != ("observed" if kind == "prediction" else "externally_measured"):
                raise ValueError("calibration prediction must be observed and measurement externally_measured")
            _number(record["value"], "record value", positive=kind == "measurement")
            raw_path = _source(source.parent, record["raw_log"], sources)
            if kind == "measurement":
                raw_hash = sources[str(raw_path)]
                if raw_hash in raw_hashes:
                    raise ValueError("measurement raw log reused across calibration samples")
                raw_hashes.add(raw_hash)
            records.append(record)
        prediction, measured = records
        if prediction["input_sha256"] != measured["input_sha256"]:
            raise ValueError("prediction and measurement inputs differ")
        group, input_hash = groups[sample["split"]], prediction["input_sha256"]
        if input_hash in group:
            raise ValueError("duplicate calibration input within split")
        group.add(input_hash)
        error = abs(prediction["value"] - measured["value"]) / measured["value"]
        _number(error, "derived relative error")
        rows.append({"split": sample["split"], "input_sha256": input_hash, "relative_error": error,
                     "within_threshold": error <= dataset["max_relative_error"]})
    if groups["train"] & groups["held_out"]:
        raise ValueError("train and held-out inputs must be disjoint")
    if any(len(group) < 2 for group in groups.values()):
        raise ValueError("each split requires at least two distinct inputs")
    held_out = [row for row in rows if row["split"] == "held_out"]
    return {"schema": "pim-foundry.metric-calibration-report.v1", "passed": all(row["within_threshold"] for row in held_out),
            "metric": dataset["metric"], "model_sha256": dataset["model_sha256"], "platform_sha256": dataset["platform_sha256"],
            "max_relative_error": dataset["max_relative_error"], "samples": rows,
            "held_out_mean_relative_error": statistics.mean(row["relative_error"] for row in held_out),
            "held_out_max_relative_error": max(row["relative_error"] for row in held_out),
            "source_file_sha256": sources, "parameters_fitted": False, "independently_authenticated": False,
            "limitation": "Disjoint content hashes do not prove absence of prior tuning or independent workload families."}

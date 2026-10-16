"""Proposed comparison protocol, not a certification or hardware benchmark.

This checker detects inconsistent declarations. Hashes identify artifacts; their
contents, quality scores, calibration and measurements still require audit.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import re
import statistics


END_TO_END_COSTS = (
    "queueing", "preprocessing", "layout_conversion", "input_transfer", "compute",
    "inter_device_transfer", "synchronization", "host_fallback", "output_transfer", "postprocessing",
)
LOGICAL_COSTS = ("input_transfer", "weight_load", "inter_device_transfer", "output_transfer")
BASE_EVIDENCE = ("functional", "analytical", "cycle_simulation", "hardware_measurement")
METRIC_UNITS = {"logical_transfer_bytes": "byte", "latency_ns": "ns",
                "throughput_per_s": "completed_request/s", "energy_j": "joule"}


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _hash(name, value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _text(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _number(name, value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number, excluding booleans")


@dataclass(frozen=True)
class BenchmarkContract:
    benchmark_id: str
    workload_sha256: str
    request_stream_sha256: str
    numeric_policy: str
    quality_metric: str
    quality_direction: str
    quality_threshold: float
    scenario: str
    initial_state: str
    measurement_scope: str
    included_costs: tuple[str, ...]
    measurement_protocol_sha256: str
    comparison_track: str = "whole_system"
    resource_budget_sha256: str | None = None
    statistic: str = "total"
    min_samples: int = 1

    def __post_init__(self):
        for name in ("benchmark_id", "numeric_policy", "quality_metric", "scenario"):
            _text(name, getattr(self, name))
        for name in ("workload_sha256", "request_stream_sha256", "measurement_protocol_sha256"):
            _hash(name, getattr(self, name))
        _number("quality_threshold", self.quality_threshold)
        if self.quality_direction not in ("at_least", "at_most"):
            raise ValueError("quality_direction must be at_least or at_most")
        if self.initial_state not in ("cold", "warm"):
            raise ValueError("initial_state must be cold or warm")
        if self.measurement_scope not in ("logical_dataflow", "kernel", "device_graph", "end_to_end"):
            raise ValueError("unknown measurement_scope")
        if type(self.included_costs) is not tuple or not self.included_costs or any(
            not isinstance(x, str) or not x for x in self.included_costs
        ) or len(set(self.included_costs)) != len(self.included_costs):
            raise ValueError("included_costs must be a nonempty tuple of unique names")
        required = set()
        if self.measurement_scope == "end_to_end":
            required.update(END_TO_END_COSTS)
            if self.initial_state == "cold":
                required.add("model_load")
        elif self.measurement_scope == "logical_dataflow":
            required.update(LOGICAL_COSTS)
        elif self.measurement_scope == "kernel":
            required.add("compute")
        elif self.measurement_scope == "device_graph":
            required.update(("compute", "input_transfer", "inter_device_transfer", "synchronization", "output_transfer"))
            if self.initial_state == "cold":
                required.add("model_load")
        if not required.issubset(self.included_costs):
            raise ValueError("measurement scope omits required costs: " + ", ".join(sorted(required - set(self.included_costs))))
        if self.comparison_track not in ("whole_system", "equal_budget"):
            raise ValueError("unknown comparison_track")
        if self.comparison_track == "equal_budget":
            _hash("resource_budget_sha256", self.resource_budget_sha256)
        elif self.resource_budget_sha256 is not None:
            raise ValueError("whole_system does not claim an equal resource budget")
        if self.statistic not in ("total", "mean", "median", "p95", "p99"):
            raise ValueError("unknown statistic")
        if type(self.min_samples) is not int or self.min_samples <= 0:
            raise ValueError("min_samples must be a positive integer")


@dataclass(frozen=True)
class BenchmarkResult:
    run_id: str
    contract: BenchmarkContract
    evidence: str
    component_evidence: tuple[tuple[str, str], ...]
    metric: str
    samples: tuple[float, ...]
    quality_value: float
    system_manifest_sha256: str
    engine_revision_sha256: str
    raw_artifact_sha256: str
    calibration_protocol_sha256: str | None = None
    calibration_artifact_sha256: str | None = None

    def __post_init__(self):
        _text("run_id", self.run_id)
        if not isinstance(self.contract, BenchmarkContract):
            raise ValueError("contract must be a BenchmarkContract")
        if self.evidence not in (*BASE_EVIDENCE, "hybrid_estimate"):
            raise ValueError("unknown evidence class")
        if type(self.component_evidence) is not tuple or not self.component_evidence:
            raise ValueError("component_evidence must be nonempty")
        names = set()
        modes = set()
        for item in self.component_evidence:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("component_evidence entries must be name/evidence pairs")
            name, mode = item
            _text("component name", name)
            if name in names or mode not in BASE_EVIDENCE:
                raise ValueError("component names must be unique with known evidence classes")
            names.add(name)
            modes.add(mode)
        if self.evidence == "hybrid_estimate":
            if len(modes) < 2:
                raise ValueError("hybrid_estimate must disclose multiple evidence classes")
        elif modes != {self.evidence}:
            raise ValueError("mixed component evidence must be labeled hybrid_estimate")
        if type(self.metric) is not str or self.metric not in METRIC_UNITS:
            raise ValueError("unknown metric; units are fixed by metric name")
        if type(self.samples) is not tuple or not self.samples:
            raise ValueError("samples must be a nonempty tuple")
        for value in self.samples:
            _number("sample", value)
            if value < 0 or (self.metric != "logical_transfer_bytes" and value == 0):
                raise ValueError("samples must be nonnegative; physical metrics must be positive")
            if self.metric == "logical_transfer_bytes" and type(value) is not int:
                raise ValueError("logical byte samples must be integers")
        _number("quality_value", self.quality_value)
        for name in ("system_manifest_sha256", "engine_revision_sha256", "raw_artifact_sha256"):
            _hash(name, getattr(self, name))
        for name in ("calibration_protocol_sha256", "calibration_artifact_sha256"):
            if getattr(self, name) is not None:
                _hash(name, getattr(self, name))
        if (self.calibration_protocol_sha256 is None) != (self.calibration_artifact_sha256 is None):
            raise ValueError("calibration protocol and artifact must be provided together")
        _number("summary", self.value)

    @property
    def value(self):
        statistic = self.contract.statistic
        if statistic == "total":
            return sum(self.samples)
        if statistic == "mean":
            return statistics.mean(self.samples)
        if statistic == "median":
            return statistics.median(self.samples)
        percentile = 0.95 if statistic == "p95" else 0.99
        return sorted(self.samples)[math.ceil(percentile * len(self.samples)) - 1]

    def to_dict(self) -> dict:
        # Return JSON types so direct API roundtrips and file roundtrips agree.
        return json.loads(json.dumps({"schema_version": 1, **asdict(self)}, allow_nan=False))

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResult":
        try:
            if type(data) is not dict or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
                raise ValueError("unsupported benchmark result schema")
            payload = dict(data)
            del payload["schema_version"]
            if type(payload["contract"]) is not dict:
                raise ValueError("JSON contract must be an object")
            contract = dict(payload["contract"])
            if type(contract["included_costs"]) is not list:
                raise ValueError("JSON included_costs must be an array")
            contract["included_costs"] = tuple(contract["included_costs"])
            payload["contract"] = BenchmarkContract(**contract)
            if type(payload["samples"]) is not list or type(payload["component_evidence"]) is not list:
                raise ValueError("JSON samples and component_evidence must be arrays")
            if any(type(x) is not list for x in payload["component_evidence"]):
                raise ValueError("JSON component_evidence pairs must be arrays")
            payload["samples"] = tuple(payload["samples"])
            payload["component_evidence"] = tuple(tuple(x) for x in payload["component_evidence"])
            return cls(**payload)
        except (TypeError, KeyError) as exc:
            raise ValueError(f"invalid benchmark record: {exc}") from exc


def _eligibility(result: BenchmarkResult) -> list[str]:
    reasons = []
    contract = result.contract
    good_quality = (result.quality_value >= contract.quality_threshold
                    if contract.quality_direction == "at_least"
                    else result.quality_value <= contract.quality_threshold)
    if not good_quality:
        reasons.append("quality_threshold_failed")
    if len(result.samples) < contract.min_samples:
        reasons.append("insufficient_samples")
    if result.metric == "logical_transfer_bytes":
        if contract.statistic != "total" or len(result.samples) != 1:
            reasons.append("logical_bytes_require_one_total_observation")
        if contract.measurement_scope != "logical_dataflow":
            reasons.append("logical_bytes_require_logical_dataflow_scope")
    else:
        if result.metric == "throughput_per_s" and contract.statistic == "total":
            reasons.append("throughput_rate_cannot_be_summed")
        if any(mode == "functional" for _, mode in result.component_evidence):
            reasons.append("functional_evidence_cannot_support_physical_metrics")
        if contract.measurement_scope == "logical_dataflow":
            reasons.append("physical_metrics_require_a_physical_measurement_scope")
        if result.evidence in ("analytical", "cycle_simulation", "hybrid_estimate") and result.calibration_artifact_sha256 is None:
            reasons.append("performance_model_calibration_not_declared")
    return reasons


def compare(left: BenchmarkResult, right: BenchmarkResult) -> dict:
    """Check declared comparability; never certify measurements or significance."""
    reasons = [f"left:{reason}" for reason in _eligibility(left)]
    reasons += [f"right:{reason}" for reason in _eligibility(right)]
    for field in fields(BenchmarkContract):
        a, b = getattr(left.contract, field.name), getattr(right.contract, field.name)
        if field.name == "included_costs":
            a, b = set(a), set(b)
        if a != b:
            reasons.append(f"contract_mismatch:{field.name}")
    if left.metric != right.metric:
        reasons.append("metric_mismatch")
    if left.evidence != right.evidence:
        reasons.append("evidence_class_mismatch")
    if left.calibration_protocol_sha256 != right.calibration_protocol_sha256:
        reasons.append("calibration_protocol_mismatch")
    if left.evidence == "hybrid_estimate" and set(left.component_evidence) != set(right.component_evidence):
        reasons.append("hybrid_evidence_basis_mismatch")
    return {
        "schema_version": 1, "metadata_compatible": not reasons,
        "reasons": reasons, "left_run": left.run_id, "right_run": right.run_id,
        "observations": None if reasons else {
            "metric": left.metric, "unit": METRIC_UNITS[left.metric],
            "statistic": left.contract.statistic, "left": left.value, "right": right.value,
        },
        "performance_ranking_produced": False,
        "independently_audited": False,
        "statistical_significance_established": False,
        "external_review_required": [
            "Verify artifact contents, system descriptions and quality measurements, not just their hashes.",
            "Verify scope accounting and calibration applicability/error bounds for each subsystem and interconnect.",
            "For equal_budget, audit actual resource use against the declared budget.",
            "Review sampling uncertainty; compatible metadata does not establish a significant performance difference.",
        ],
    }


def reference_records(experiment: dict) -> dict[str, BenchmarkResult]:
    """Attach the proposed protocol to real functional-reference observations."""
    protocol = {
        "name": "pim-foundry-logical-dataflow-v1", "cold": True,
        "weight_residency": "host_then_stream_per_consumer", "execution": "sequential",
        "metric": "logical_link_payload_bytes", "statistic": "one_total",
        "physical_metric": None,
    }
    contract = BenchmarkContract(
        benchmark_id="pim-foundry/reference-linear/v1",
        workload_sha256=experiment["workload_sha256"],
        request_stream_sha256=content_hash({"inputs": experiment["inputs"], "requests": 1}),
        numeric_policy="signed_int64_storage_exact_integer_intermediates",
        quality_metric="exact_output_match", quality_direction="at_least", quality_threshold=1,
        scenario="one_functional_graph", initial_state="cold", measurement_scope="logical_dataflow",
        included_costs=LOGICAL_COSTS, measurement_protocol_sha256=content_hash(protocol),
    )
    records = {}
    for name, run in experiment["runs"].items():
        components = tuple((device["name"], "functional") for device in run["plan"]["platform"]["devices"])
        records[name] = BenchmarkResult(
            run_id=name, contract=contract, evidence="functional", component_evidence=components,
            metric="logical_transfer_bytes", samples=(run["plan"]["logical_transfer_bytes"],),
            quality_value=int(run["validation"]["matches_scalar_oracle"]),
            system_manifest_sha256=content_hash(run["plan"]["platform"]),
            engine_revision_sha256=content_hash(experiment["engine_source_sha256"]),
            raw_artifact_sha256=content_hash(run),
        )
    return records

"""Deterministic analytical scheduling from explicit, uncalibrated costs.

This module schedules logical work and payload transfers.  It does not execute
vendor instructions, infer costs from Python execution time, or model DRAM
commands, caches, refresh, numerical precision, or physical cycle behavior.
"""

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import math

from .core import Graph, Platform
from .research import plan


MAX_REQUESTS = 1024
MAX_EVENTS = 200000
MAX_TIME_NS = 2**63 - 1
_DEVICE_KEYS = {
    "macs_per_ns", "elements_per_ns", "launch_ns", "mac_energy_pj",
    "element_energy_pj", "idle_power_w", "local_bytes_per_ns", "local_energy_pj_per_byte",
}
_LINK_KEYS = {
    "source", "destination", "bytes_per_ns", "latency_ns",
    "energy_pj_per_byte", "resource",
}


def _exact_keys(value, keys, label):
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {', '.join(sorted(keys))}")


def _number(value, label, positive=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (positive and value == 0):
        qualifier = "strictly positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be a finite {qualifier} number, excluding bool")


def validate_model(platform: Platform, model: dict) -> None:
    """Reject missing, extra, nonfinite, or ambiguous cost-model fields."""
    platform.validate()
    _exact_keys(model, {"schema_version", "provenance", "devices", "links"}, "model")
    if type(model["schema_version"]) is not int or model["schema_version"] != 1:
        raise ValueError("model schema_version must be 1")
    provenance = model["provenance"]
    _exact_keys(provenance, {"kind", "description"}, "model provenance")
    if provenance["kind"] not in ("synthetic", "supplied", "measured"):
        raise ValueError("provenance kind must be synthetic, supplied, or measured")
    if type(provenance["description"]) is not str or not provenance["description"].strip():
        raise ValueError("provenance description must be nonempty")
    devices = model["devices"]
    _exact_keys(devices, {device.name for device in platform.devices}, "model devices")
    for name, costs in devices.items():
        _exact_keys(costs, _DEVICE_KEYS, f"device {name}")
        for key, value in costs.items():
            _number(value, f"device {name}.{key}", positive=key.endswith("_per_ns"))
    if type(model["links"]) is not list:
        raise ValueError("model links must be a list")
    expected = set(platform.links)
    seen = set()
    for costs in model["links"]:
        _exact_keys(costs, _LINK_KEYS, "link")
        if any(type(costs[key]) is not str for key in ("source", "destination", "resource")):
            raise ValueError("link endpoints and resource must be strings")
        edge = costs["source"], costs["destination"]
        if edge not in expected or edge in seen:
            raise ValueError("model links must match directed platform links without duplicates")
        seen.add(edge)
        if not costs["resource"].strip():
            raise ValueError("link resource must be nonempty")
        for key in ("bytes_per_ns", "latency_ns", "energy_pj_per_byte"):
            _number(costs[key], f"link {edge}.{key}", positive=key == "bytes_per_ns")
    if seen != expected:
        raise ValueError("model links must cover every directed platform link")


def example_model(platform: Platform) -> dict:
    """Return deliberately identical synthetic device costs, with no vendor fit."""
    platform.validate()
    return {
        "schema_version": 1,
        "provenance": {
            "kind": "synthetic",
            "description": "Uncalibrated illustrative constants; not measurements of any device.",
        },
        "devices": {
            device.name: {
                "macs_per_ns": 16, "elements_per_ns": 8, "local_bytes_per_ns": 32,
                "launch_ns": 20, "mac_energy_pj": 1, "element_energy_pj": 0.5,
                "local_energy_pj_per_byte": 0.25, "idle_power_w": 0.1,
            }
            for device in platform.devices
        },
        "links": [
            {"source": source, "destination": destination,
             "bytes_per_ns": 8, "latency_ns": 10, "energy_pj_per_byte": 2,
             "resource": "<->".join(sorted((source, destination)))}
            for source, destination in sorted(platform.links)
        ],
    }


def _duration(work, rate, fixed):
    # JSON decimal rates use their shortest decimal representation, rather
    # than a binary-float residue (e.g. ceil(3 / 0.3) must remain 10).
    # Fraction also avoids float overflow for tiny but finite rates.
    duration = math.ceil(Fraction(work) / Fraction(str(rate))) + math.ceil(fixed)
    if duration > MAX_TIME_NS:
        raise ValueError("derived task duration exceeds the signed 64-bit nanosecond limit")
    return duration


def _finite_energy(value):
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("derived energy exceeds the finite report range")
    return value


def _tasks(graph, platform, placement, model, execution_plan, initial_state):
    """One invocation's serial task chain, with explicit routed weight tiles."""
    tasks = []
    devices = {device.name: device for device in platform.devices}
    links = {(item["source"], item["destination"]): item for item in model["links"]}
    transfers = {}
    for item in execution_plan["transfers"]:
        transfers.setdefault(item["consumer"], []).append(item)

    def append(task):
        if len(tasks) >= MAX_EVENTS:
            raise ValueError(f"simulation exceeds the {MAX_EVENTS}-event limit")
        tasks.append(task)

    def transfer(item, payload_bytes=None, tile=None):
        size = item["payload_bytes"] if payload_bytes is None else payload_bytes
        for source, destination in zip(item["path"], item["path"][1:]):
            cost = links[source, destination]
            task = {
                "kind": "transfer", "node": item["consumer"], "tensor": item["tensor"],
                "source": source, "destination": destination, "bytes": size,
                "resources": sorted({f"link:{cost['resource']}", f"dma:{source}", f"dma:{destination}"}),
                "duration_ns": _duration(size, cost["bytes_per_ns"], cost["latency_ns"]),
                "dynamic_energy_pj": _finite_energy(size * cost["energy_pj_per_byte"]),
            }
            if tile is not None:
                task["tile"] = tile
            append(task)

    for node in graph.nodes:
        device = devices[placement[node.name]]
        costs = model["devices"][device.name]
        # The plan's output consumer label can also be a legal node name.
        # Device destinations distinguish node inputs from host readback.
        items = [item for item in transfers.get(node.name, []) if item["destination"] == device.name]
        for item in items:
            if not item["streamed"]:
                transfer(item)
        if node.op in ("matvec", "matmul"):
            weight = graph.tensor_map[node.inputs[1]]
            k, n = weight.shape
            rows = 1 if node.op == "matvec" else graph.tensor_map[node.inputs[0]].shape[0]
            weight_items = [item for item in items if item["streamed"] and item["tensor"] == weight.name]
            if len(weight_items) != 1:
                raise ValueError("plan must provide exactly one streamed weight record per linear node")
            item = weight_items[0]
            if item.get("repetitions", 1) != rows:
                raise ValueError("plan weight repetitions disagree with matmul row count")
            first_tile = True
            for row in range(rows):
                for n0 in range(0, n, device.tile_n):
                    n1 = min(n, n0 + device.tile_n)
                    for k0 in range(0, k, device.tile_k):
                        k1 = min(k, k0 + device.tile_k)
                        macs = (k1 - k0) * (n1 - n0)
                        tile = {"m": row, "k0": k0, "k1": k1, "n0": n0, "n1": n1}
                        if initial_state == "cold":
                            transfer(item, macs * 8, tile)
                        local_bytes = 8 * (macs + (k1 - k0) + 2 * (n1 - n0))
                        work_ns = max(_duration(macs, costs["macs_per_ns"], 0),
                                      _duration(local_bytes, costs["local_bytes_per_ns"], 0))
                        append({
                            "kind": "compute", "node": node.name, "tensor": node.output,
                            "device": device.name, "op": node.op, "macs": macs, "tile": tile,
                            "resources": [f"compute:{device.name}"], "local_bytes": local_bytes,
                            "duration_ns": work_ns + math.ceil(costs["launch_ns"] if first_tile else 0),
                            "launch_charged": first_tile,
                            "dynamic_energy_pj": _finite_energy(macs * costs["mac_energy_pj"] + local_bytes * costs["local_energy_pj_per_byte"]),
                        })
                        first_tile = False
        elif node.op in ("relu", "add"):
            elements = math.prod(graph.tensor_map[node.output].shape)
            local_bytes = 8 * elements * (len(node.inputs) + 1)
            work_ns = max(_duration(elements, costs["elements_per_ns"], 0),
                          _duration(local_bytes, costs["local_bytes_per_ns"], 0))
            append({
                "kind": "compute", "node": node.name, "tensor": node.output,
                "device": device.name, "op": node.op, "elements": elements,
                "resources": [f"compute:{device.name}"], "launch_charged": True,
                "local_bytes": local_bytes,
                "duration_ns": work_ns + math.ceil(costs["launch_ns"]),
                "dynamic_energy_pj": _finite_energy(elements * costs["element_energy_pj"] + local_bytes * costs["local_energy_pj_per_byte"]),
            })
        else:
            raise ValueError(f"simulation does not support {node.op!r}")
    for item in transfers.get("graph_output", []):
        if item["destination"] == "host":
            transfer(item)
    return tasks


def simulate(graph: Graph, platform: Platform, placement: dict, model: dict,
             arrivals_ns=(0,), max_inflight=1, initial_state="cold") -> dict:
    """Schedule serial invocations with overlap only across independent requests.

    A fixed request slot is released on completion: request i may begin only
    after request i-max_inflight finishes.  Ready tasks are selected by earliest
    feasible start time, then request index; ties therefore never depend on hash
    iteration.  Each transfer holds its directed link's declared shared resource
    and both endpoint DMA resources for its complete duration.
    """
    validate_model(platform, model)
    model = deepcopy(model)
    if initial_state not in ("cold", "warm"):
        raise ValueError("initial_state must be cold or warm")
    if type(arrivals_ns) not in (list, tuple) or not 1 <= len(arrivals_ns) <= MAX_REQUESTS:
        raise ValueError(f"arrivals_ns must contain 1 to {MAX_REQUESTS} arrival times")
    if any(type(value) is not int or not 0 <= value <= MAX_TIME_NS for value in arrivals_ns):
        raise ValueError("arrival times must be nonnegative signed 64-bit integer nanoseconds")
    if list(arrivals_ns) != sorted(arrivals_ns):
        raise ValueError("arrival times must be sorted")
    if type(max_inflight) is not int or not 1 <= max_inflight <= len(arrivals_ns):
        raise ValueError("max_inflight must be a positive integer no greater than request count")
    execution_plan = plan(graph, platform, placement)
    persistent = {device.name: 0 for device in platform.devices}
    preload_bytes = Counter()
    if initial_state == "warm":
        weights = set()
        for node in graph.nodes:
            if node.op in ("matvec", "matmul"):
                key = placement[node.name], node.inputs[1]
                if key in weights:
                    continue
                weights.add(key)
                device_name, tensor_name = key
                size = graph.tensor_map[tensor_name].nbytes
                persistent[device_name] += size
                route = platform.path("host", device_name)
                for source, destination in zip(route, route[1:]):
                    preload_bytes[f"{source}->{destination}"] += size
    reserved = {name: amount * max_inflight + persistent[name]
                for name, amount in execution_plan["peak_live_bytes"].items()}
    for device in platform.devices:
        if reserved[device.name] > device.capacity_bytes:
            raise ValueError(f"{device.name}: max_inflight reserves {reserved[device.name]} bytes, capacity is {device.capacity_bytes}")
    template = _tasks(graph, platform, placement, model, execution_plan, initial_state)
    request_count = len(arrivals_ns)
    if len(template) * request_count > MAX_EVENTS:
        raise ValueError(f"simulation exceeds the {MAX_EVENTS}-event limit")

    resources = {f"compute:{device.name}" for device in platform.devices}
    resources.update(f"dma:{endpoint}" for endpoint in ("host", *(d.name for d in platform.devices)))
    resources.update(f"link:{item['resource']}" for item in model["links"])
    available = dict.fromkeys(resources, 0)
    busy = dict.fromkeys(resources, 0)
    position = [0] * request_count
    last_end = list(arrivals_ns)
    completion = [None] * request_count
    timeline = []

    def task_id(request, index):
        return f"r{request}:t{index}"

    while len(timeline) < len(template) * request_count:
        ready = []
        for request, index in enumerate(position):
            if index == len(template):
                continue
            predecessor = request - max_inflight
            if index == 0 and predecessor >= 0 and completion[predecessor] is None:
                continue
            earliest = last_end[request]
            if index == 0 and predecessor >= 0:
                earliest = max(earliest, completion[predecessor])
            task = template[index]
            start = max(earliest, *(available[resource] for resource in task["resources"]))
            ready.append((start, request, index))
        if not ready:
            raise ValueError("simulation dependency graph has no ready task")
        start, request, index = min(ready)
        task = template[index]
        end = start + task["duration_ns"]
        if end > MAX_TIME_NS:
            raise ValueError("simulation completion exceeds the signed 64-bit nanosecond limit")
        dependencies = []
        if index:
            dependencies.append(task_id(request, index - 1))
        elif request >= max_inflight:
            dependencies.append(task_id(request - max_inflight, len(template) - 1))
        timeline.append({
            "id": task_id(request, index), "request": request,
            "start_ns": start, "end_ns": end, "dependencies": dependencies, **task,
        })
        for resource in task["resources"]:
            available[resource] = end
            busy[resource] += task["duration_ns"]
        last_end[request] = end
        position[request] += 1
        if position[request] == len(template):
            completion[request] = end

    logical_bytes = Counter()
    compute_energy = Counter()
    transfer_energy = Counter()
    for task in timeline:
        if task["kind"] == "transfer":
            edge = f"{task['source']}->{task['destination']}"
            logical_bytes[edge] += task["bytes"]
            transfer_energy[edge] += task["dynamic_energy_pj"]
        else:
            compute_energy[task["device"]] += task["dynamic_energy_pj"]
    expected_bytes = Counter()
    for transfer in execution_plan["transfers"]:
        if initial_state == "warm" and transfer["streamed"]:
            continue
        amount = transfer["payload_bytes"] * transfer.get("repetitions", 1) * request_count
        for source, destination in zip(transfer["path"], transfer["path"][1:]):
            expected_bytes[f"{source}->{destination}"] += amount
    if dict(logical_bytes) != expected_bytes:
        raise ValueError("simulated transfer payloads disagree with the logical memory plan")
    makespan = max(completion) - arrivals_ns[0]
    idle_energy = {
        device.name: _finite_energy(model["devices"][device.name]["idle_power_w"] * makespan * 1000)
        for device in platform.devices
    }
    dynamic = _finite_energy(sum(compute_energy.values()) + sum(transfer_energy.values()))
    idle = _finite_energy(sum(idle_energy.values()))
    latencies = [finished - arrived for finished, arrived in zip(completion, arrivals_ns)]
    return {
        "schema_version": 1, "status": "analytical_simulation", "evidence": "analytical",
        "calibrated": False, "model": model, "plan": execution_plan,
        "arrivals_ns": list(arrivals_ns), "max_inflight": max_inflight, "initial_state": initial_state,
        "requests": [{"request": i, "arrival_ns": arrival, "completion_ns": completion[i],
                      "latency_ns": latencies[i]} for i, arrival in enumerate(arrivals_ns)],
        "completion_ns": completion, "latency_ns": latencies,
        "makespan_ns": makespan,
        "throughput_requests_per_second": request_count * 1_000_000_000 / makespan,
        "timeline": timeline, "resource_busy_ns": dict(sorted(busy.items())),
        "peak_reserved_bytes": reserved,
        "persistent_weight_bytes": persistent,
        "excluded_setup_logical_bytes": sum(preload_bytes.values()),
        "excluded_setup_logical_bytes_by_link": dict(sorted(preload_bytes.items())),
        "logical_transfer_bytes": sum(logical_bytes.values()),
        "logical_bytes_by_link": dict(sorted(logical_bytes.items())),
        "energy": {
            "dynamic_pj": dynamic, "idle_pj": idle, "total_pj": _finite_energy(dynamic + idle),
            "compute_dynamic_pj_by_device": {device.name: compute_energy[device.name] for device in platform.devices},
            "transfer_dynamic_pj_by_link": dict(sorted(transfer_energy.items())),
            "idle_pj_by_device": idle_energy,
        },
        "assumptions": [
            "Analytical costs are supplied inputs, not Python timings, physical measurements, or cycle-accurate hardware evidence.",
            "Every request follows the plan's serial node order; different requests may overlap on distinct resources.",
            "One compute resource per device; macs_per_ns and elements_per_ns already describe aggregate device throughput.",
            "Transfers serialize on their declared shared link resource and on conservative half-duplex DMA resources at both endpoints, including host.",
            "Routes minimize hop count; each hop transfers the complete chunk before the next hop starts.",
            "Weights start on host, stream one k/n tile before each compute tile, and reload for each matmul row; no full-matrix buffering or weight cache.",
            "The preceding weight-stream assumption applies to cold mode. Warm mode assumes unique device/weight pairs already resident, excludes preload time and energy, and reports excluded one-time routed preload bytes.",
            "Warm memory reservation adds complete persistent weights to every request slot's cold-plan peak, conservatively retaining the weight-tile scratch allocation too. The nested plan describes the cold baseline.",
            "Input/output activations are stored in full. max_inflight reserves each device's full per-invocation peak for every fixed request slot.",
            "Each tile's work/rate is rounded upward to integer nanoseconds; launch cost is charged once per node, link latency once per hop and chunk.",
            "Compute tile duration is max(arithmetic time, local-memory time) plus launch. Linear tile local bytes = 8*(Ktile*Ntile + Ktile + 2*Ntile): weight and input reads plus conservative partial-output read/write. Element operations read each operand and write one output; duplicate operands still count separate reads.",
            "idle_power_w is an always-on baseline charged to every platform device across the observed interval, including active periods and idle gaps; host baseline power is excluded.",
            "Observed interval runs from first arrival to final completion; throughput includes all completed requests in that interval, not a steady-state extrapolation.",
            "Provenance labels alone do not establish calibration. Numerical correctness, vendor execution, temperature, refresh, caches, and operating-system scheduling are not inferred.",
        ],
    }

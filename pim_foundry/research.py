"""Placement, explicit transfers, liveness and execution of logical models.

All byte counts are logical int64 payloads. No bandwidth, DRAM transaction,
latency, energy or hardware performance is inferred from these experiments.
"""

from collections import Counter
from dataclasses import asdict
from itertools import product
from math import prod
import hashlib
import json
from pathlib import Path

from .core import Graph, Node, Platform, ReferenceDevice, Tensor, checked_integer
from .reference import check_event_budget, product_event_count


def _activation_inputs(node: Node) -> tuple[str, ...]:
    """Distinct full activation buffers; matrix weights use separate streaming."""
    names = node.inputs[:1] if node.op in ("matvec", "matmul") else node.inputs
    return tuple(dict.fromkeys(names))


def plan(graph: Graph, platform: Platform, placement: dict[str, str]) -> dict:
    graph.validate()
    platform.validate()
    devices = {d.name: d for d in platform.devices}
    if type(placement) is not dict or set(placement) != {node.name for node in graph.nodes} or any(
        not isinstance(target, str) or target not in devices for target in placement.values()
    ):
        raise ValueError("placement must assign every node exactly once to a known device")
    tensors = graph.tensor_map
    event_counts = {}
    for node in graph.nodes:
        if node.op in ("matvec", "matmul"):
            rows = tensors[node.inputs[0]].shape[0] if node.op == "matmul" else 1
            k, n = tensors[node.inputs[1]].shape
            count = product_event_count(rows, k, n, devices[placement[node.name]])
        else:
            count = 1
        event_counts[node.name] = count
    total_events = sum(event_counts.values())
    check_event_budget(total_events)
    origins = {name: "host" for name in graph.inputs}
    origins.update({t.name: "host" for t in graph.tensors if t.data is not None})
    uses = Counter(name for node in graph.nodes for name in node.inputs)
    live = {name: {} for name in devices}
    peaks = {name: 0 for name in devices}
    transfers, nodes = [], []

    def resident(device):
        return sum(live[device].values())

    def check_memory(device, total):
        peaks[device] = max(peaks[device], total)
        if total > devices[device].capacity_bytes:
            raise ValueError(
                f"{device}: needs {total} logical bytes, capacity is {devices[device].capacity_bytes}"
            )

    def transfer(tensor, source, destination, consumer, chunk_bytes=None, repetitions=1):
        route = platform.path(source, destination)
        if len(route) == 1:
            return
        # Staging is modeled at intermediate endpoints, one sequential transfer
        # at a time. Target working storage is included in the compute peak.
        for endpoint in route[1:-1]:
            if endpoint != "host":
                check_memory(endpoint, resident(endpoint) + (chunk_bytes or tensor.nbytes))
        transfers.append({
            "tensor": tensor.name, "consumer": consumer, "source": source,
            "destination": destination, "path": list(route),
            "payload_bytes": tensor.nbytes,
            "repetitions": repetitions,
            "logical_link_bytes": tensor.nbytes * repetitions * (len(route) - 1),
            "streamed": chunk_bytes is not None,
            "max_chunk_bytes": chunk_bytes or tensor.nbytes,
        })

    for node in graph.nodes:
        target = placement[node.name]
        device = devices[target]
        out = tensors[node.output]
        input_staging = 0
        for name in _activation_inputs(node):
            activation = tensors[name]
            transfer(activation, origins[name], target, node.name)
            if origins[name] != target:
                input_staging += activation.nbytes
        weight_staging = 0
        weight_repetitions = 0
        if node.op in ("matvec", "matmul"):
            weight = tensors[node.inputs[1]]
            k, n = weight.shape
            weight_staging = 8 * min(k, device.tile_k) * min(n, device.tile_n)
            weight_repetitions = tensors[node.inputs[0]].shape[0] if node.op == "matmul" else 1
            transfer(weight, "host", target, node.name, weight_staging, weight_repetitions)
        peak = resident(target) + input_staging + out.nbytes + weight_staging
        check_memory(target, peak)
        nodes.append({"node": node.name, "op": node.op, "device": target,
                      "working_peak_bytes": peak, "input_staging_bytes": input_staging,
                      "weight_tile_bytes": weight_staging, "weight_load_repetitions": weight_repetitions,
                      "kernel_event_count": event_counts[node.name]})
        origins[out.name] = target
        live[target][out.name] = out.nbytes
        for name in node.inputs:
            uses[name] -= 1
            owner = origins[name]
            if uses[name] == 0 and name not in graph.outputs and owner != "host":
                live[owner].pop(name, None)
        if uses[out.name] == 0 and out.name not in graph.outputs:
            live[target].pop(out.name)

    for name in graph.outputs:
        transfer(tensors[name], origins[name], "host", "graph_output")
    link_bytes = Counter()
    for item in transfers:
        for source, destination in zip(item["path"], item["path"][1:]):
            link_bytes[f"{source}->{destination}"] += item["payload_bytes"] * item["repetitions"]
    return {
        "schema_version": 1, "status": "logical_reference_plan",
        "numeric_contract": "signed_int64_storage_exact_integer_intermediates",
        "placement": {node.name: placement[node.name] for node in graph.nodes}, "platform": asdict(platform),
        "nodes": nodes, "transfers": transfers, "peak_live_bytes": peaks,
        "kernel_event_count": total_events,
        "logical_transfer_bytes": sum(item["logical_link_bytes"] for item in transfers),
        "logical_bytes_by_link": dict(sorted(link_bytes.items())),
        "assumptions": [
            "Sequential execution; transfer paths minimize hops, not latency.",
            "Cold invocation: constants and inputs start on host; matvec weights stream once, matmul weights once per input row.",
            "No weight cache, activation replication cache, overlap, timing, energy or physical bus model.",
            "Full activation input/output tensors must fit; repeated operands share a buffer; weight matrices are tiled.",
            "SRAM/HBM/GPU/NPU profiles describe illustrative logical resources, not vendor devices.",
        ],
    }


def _input_values(graph: Graph, inputs: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    if set(inputs) != set(graph.inputs):
        raise ValueError("provide exactly the graph's named inputs")
    values = {t.name: t.data for t in graph.tensors if t.data is not None}
    for name, data in inputs.items():
        if type(data) is not tuple or len(data) != prod(graph.tensor_map[name].shape):
            raise ValueError("input values must be flat tuples matching the declared shape")
        values[name] = tuple(checked_integer(value) for value in data)
    return values


def scalar_oracle(graph: Graph, inputs: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    """Independent untiled scalar evaluation of the same mathematical contract."""
    graph.validate()
    values = _input_values(graph, inputs)
    for node in graph.nodes:
        x = values[node.inputs[0]]
        if node.op in ("matvec", "matmul"):
            w = graph.tensor_map[node.inputs[1]]
            k, n = w.shape
            rows = graph.tensor_map[node.inputs[0]].shape[0] if node.op == "matmul" else 1
            result = tuple(checked_integer(sum(x[row * k + i] * w.data[i * n + j] for i in range(k)))
                           for row in range(rows) for j in range(n))
        elif node.op == "add":
            y = values[node.inputs[1]]
            result = tuple(checked_integer(x[i] + y[i]) for i in range(len(x)))
        elif node.op == "relu":
            result = tuple(max(0, value) for value in x)
        values[node.output] = result
    return {name: values[name] for name in graph.outputs}


def execute(graph: Graph, inputs: dict[str, tuple[int, ...]], platform: Platform,
            placement: dict[str, str]) -> dict:
    from .reference import evaluate_add, evaluate_matmul, evaluate_matvec, evaluate_relu

    execution_plan = plan(graph, platform, placement)
    values = _input_values(graph, inputs)
    devices = {device.name: device for device in platform.devices}
    # Activation copies follow the planned path in the functional execution.
    # Immutable tuples represent payloads; this is not a physical byte simulator.
    buffers = {"host": dict(values), **{name: {} for name in devices}}
    homes = {name: "host" for name in values}
    uses = Counter(name for node in graph.nodes for name in node.inputs)
    events = []
    for node in graph.nodes:
        device = devices[placement[node.name]]
        routes = {}
        for input_name in _activation_inputs(node):
            route = platform.path(homes[input_name], device.name)
            routes[input_name] = route
            for source, destination in zip(route, route[1:]):
                buffers[destination][input_name] = tuple(value for value in buffers[source][input_name])
                if source != homes[input_name]:
                    # Relays release staging before the next transfer; distinct
                    # inputs remain together at the target until computation.
                    buffers[source].pop(input_name)
        x = buffers[device.name][node.inputs[0]]
        if node.op == "matvec":
            result = evaluate_matvec(x, graph.tensor_map[node.inputs[1]], device)
        elif node.op == "matmul":
            result = evaluate_matmul(x, graph.tensor_map[node.inputs[1]], device)
        elif node.op == "add":
            result = evaluate_add(x, buffers[device.name][node.inputs[1]], device)
        elif node.op == "relu":
            result = evaluate_relu(x, device)
        values[node.output] = result.values
        buffers[device.name][node.output] = result.values
        homes[node.output] = device.name
        events.extend({"node": node.name, "device": device.name, **event} for event in result.events)
        for input_name, route in routes.items():
            for endpoint in route[1:]:
                buffers[endpoint].pop(input_name, None)
        for name in node.inputs:
            uses[name] -= 1
            if uses[name] == 0 and name not in graph.outputs:
                buffers[homes[name]].pop(name, None)
        if uses[node.output] == 0 and node.output not in graph.outputs:
            buffers[device.name].pop(node.output, None)
    for name in graph.outputs:
        route = platform.path(homes[name], "host")
        for source, destination in zip(route, route[1:]):
            buffers[destination][name] = tuple(value for value in buffers[source][name])
            if source != homes[name]:
                buffers[source].pop(name)
        for endpoint in route[1:-1]:
            buffers[endpoint].pop(name, None)
    output = {name: buffers["host"][name] for name in graph.outputs}
    reference = scalar_oracle(graph, inputs)
    if output != reference:
        raise ValueError("reference backend disagrees with independent scalar oracle")
    return {
        "schema_version": 1, "status": "functional_reference_verified",
        "outputs": {name: list(data) for name, data in output.items()},
        "plan": execution_plan, "kernel_events": events,
        "validation": {"matches_scalar_oracle": True, "logical_resources_checked": True,
                       "vendor_backend_numerics": False, "timing_simulation": False,
                       "hardware_execution": False},
    }


def rank_placements(graph: Graph, platform: Platform) -> dict:
    """Enumerate legal assignments, ordered only by logical transfer volume."""
    graph.validate()
    platform.validate()
    names = sorted(device.name for device in platform.devices)
    if len(names) ** len(graph.nodes) > 4096:
        raise ValueError("placement enumeration exceeds the 4096-candidate limit")
    candidates, rejected = [], []
    for targets in product(names, repeat=len(graph.nodes)):
        placement = {node.name: target for node, target in zip(graph.nodes, targets)}
        try:
            result = plan(graph, platform, placement)
            candidates.append({"placement": placement,
                               "logical_transfer_bytes": result["logical_transfer_bytes"],
                               "peak_live_bytes": result["peak_live_bytes"]})
        except ValueError as exc:
            rejected.append({"placement": placement, "reason": str(exc)})
    candidates.sort(key=lambda x: (x["logical_transfer_bytes"], tuple(x["placement"].values())))
    return {"objective": "logical_transfer_bytes_only", "performance_optimum_claimed": False,
            "candidates": candidates, "rejected": rejected}


def platform_demo(kinds: tuple[str, ...]) -> dict:
    """Compare selected logical profiles using an identical small workload.

    Profiles are illustrative capacities and tile shapes, not vendor devices,
    executable GPU/NPU kernels or performance models. The demo topology alone is
    fully connected. Each run retains all selected devices, even if inactive.
    """
    profiles = {
        "sram": (256, 4, 3, 2),
        "hbm": (4096, 8, 4, 4),
        "gpu": (8192, 8, 8, 8),
        "npu": (2048, 4, 4, 4),
    }
    if (type(kinds) is not tuple or not 1 <= len(kinds) <= len(profiles)
            or any(type(kind) is not str or kind not in profiles for kind in kinds)
            or len(set(kinds)) != len(kinds)):
        raise ValueError("kinds must be a nonempty tuple of unique sram, hbm, gpu or npu options")
    # A set of options has one stable report order, independent of caller order.
    selected = tuple(kind for kind in profiles if kind in kinds)
    graph = Graph(
        tensors=(Tensor("x", (8,)),
                 Tensor("w1", (8, 6), tuple((i * 7) % 11 - 5 for i in range(48))),
                 Tensor("hidden", (6,)), Tensor("activated", (6,)),
                 Tensor("w2", (6, 4), tuple((i * 3) % 7 - 3 for i in range(24))),
                 Tensor("y", (4,))),
        nodes=(Node("linear1", "matvec", ("x", "w1"), "hidden"),
               Node("activation", "relu", ("hidden",), "activated"),
               Node("linear2", "matvec", ("activated", "w2"), "y")),
        inputs=("x",), outputs=("y",),
    )
    endpoints = ("host", *selected)
    platform = Platform(
        devices=tuple(ReferenceDevice(kind, kind, *profiles[kind]) for kind in selected),
        links=tuple((source, destination) for source in endpoints for destination in endpoints
                    if source != destination),
    )
    inputs = {"x": tuple(range(-4, 4))}
    placements = {
        f"{kind}_only": {node.name: kind for node in graph.nodes} for kind in selected
    }
    if len(selected) >= 2:
        downstream = next(kind for kind in ("gpu", "npu", "sram") if kind in selected)
        upstream = next(kind for kind in ("hbm", "sram", "npu", "gpu")
                        if kind in selected and kind != downstream)
        placements["heterogeneous"] = {
            "linear1": upstream, "activation": downstream, "linear2": downstream,
        }
    workload = {"graph": asdict(graph), "inputs": inputs}
    identity = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("core.py", "reference.py", "research.py")}
    return {"schema_version": 1, "kind": "illustrative_reference_experiment",
            "workload_sha256": identity, "engine_source_sha256": source_hashes,
            "graph": asdict(graph), "inputs": {name: list(data) for name, data in inputs.items()},
            "runs": {name: execute(graph, inputs, platform, mapping) for name, mapping in placements.items()},
            "placement_search": rank_placements(graph, platform)}


def demo() -> dict:
    """Preserve the original SRAM/HBM example and its three named runs."""
    return platform_demo(("sram", "hbm"))

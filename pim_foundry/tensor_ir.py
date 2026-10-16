"""Versioned n-dimensional tensor graph contract, independent of hardware models.

State is explicit: a decoder cache is a graph input and its replacement a graph
output. No hidden mutable state, implicit dtype conversions, or executable JSON.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping

SCHEMA = "pim-foundry.tensor-graph.v2"
DTYPES = {"float16": 2, "float32": 4, "int8": 1, "int32": 4}
DEFAULT_LIMITS = {"max_tensor_elements": 250_000_000,
                  "max_total_elements": 500_000_000,
                  "max_operations": 100_000_000_000}
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]*$")
_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Arity and accepted attributes are part of this version's public contract.
OPS = {
    "matmul": ((2,), {"transpose_a", "transpose_b"}),
    "add": ((2,), set()), "sub": ((2,), set()),
    "mul": ((2,), set()), "div": ((2,), set()),
    "reshape": ((1,), {"shape"}), "transpose": ((1,), {"perm"}),
    "gather": ((2,), {"axis"}), "embedding": ((2,), set()),
    "slice": ((1,), {"starts", "ends", "axes", "steps"}),
    "concat": (None, {"axis"}), "softmax": ((1,), {"axis"}),
    "layer_norm": ((1, 2, 3), {"axis", "epsilon"}),
    "rms_norm": ((1, 2), {"axis", "epsilon"}),
    "gelu": ((1,), {"approximate"}), "silu": ((1,), set()),
    "relu": ((1,), set()), "cast": ((1,), {"dtype"}),
    "reduce_mean": ((1,), {"axes", "keepdims"}),
    "argmax": ((1,), {"axis", "keepdims"}),
    "causal_attention": ((3,), {"query_offset", "scale"}),
    "rotary": ((3,), {"rotary_dim", "interleaved"}),
    "identity": ((1,), set()),
    "conv2d": ((2, 3), {"strides", "pads", "dilations", "groups"}),
    "max_pool2d": ((1,), {"kernel", "strides", "pads"}),
    "avg_pool2d": ((1,), {"kernel", "strides", "pads", "count_include_pad"}),
}


def _name(value, label):
    if not isinstance(value, str) or not _NAME.fullmatch(value) or len(value) > 256:
        raise ValueError(f"{label} must be a nonempty tensor identifier")


def _json(value, label):
    """Metadata and attrs must be data, never Python objects or nonfinite JSON."""
    import json
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON values") from exc


def validate_graph(graph):
    """Validate SSA, operator declarations, explicit dtypes, and shape contracts.

    Return a normalized independent structural copy. Initializer array storage is
    deliberately not copied here; execute() validates and copies it before use.
    Runtime shapes are checked again after every node and after backend callbacks.
    """
    if not isinstance(graph, Mapping):
        raise ValueError("tensor graph must be an object")
    required = {"schema", "values", "inputs", "outputs", "nodes"}
    optional = {"initializers", "shape_contract", "limits", "metadata"}
    if not required <= graph.keys() or graph.keys() - required - optional:
        raise ValueError("tensor graph has missing or unknown fields")
    if graph["schema"] != SCHEMA:
        raise ValueError(f"tensor graph schema must be {SCHEMA}")
    values = graph["values"]
    if not isinstance(values, Mapping) or not values:
        raise ValueError("values must be a nonempty mapping")
    symbols = set()
    for name, spec in values.items():
        _name(name, "value name")
        if not isinstance(spec, Mapping) or set(spec) != {"shape", "dtype"}:
            raise ValueError(f"{name}: value requires exactly shape and dtype")
        if not isinstance(spec["dtype"], str) or spec["dtype"] not in DTYPES:
            raise ValueError(f"{name}: unsupported dtype")
        shape = spec["shape"]
        if not isinstance(shape, (list, tuple)) or len(shape) > 16:
            raise ValueError(f"{name}: shape must have at most 16 dimensions")
        for dim in shape:
            if type(dim) is int and dim >= 0:
                continue
            if isinstance(dim, str) and _SYMBOL.fullmatch(dim):
                symbols.add(dim)
                continue
            raise ValueError(f"{name}: dimensions must be nonnegative integers or symbols")
    for field in ("inputs", "outputs"):
        names = graph[field]
        if not isinstance(names, (list, tuple)) or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
            raise ValueError(f"{field} must be a list of unique value names")
        if field == "outputs" and not names:
            raise ValueError("at least one output is required")
        for name in names:
            if name not in values:
                raise ValueError(f"{field}: undeclared value {name}")
    initializers = graph.get("initializers", {})
    if not isinstance(initializers, Mapping) or initializers.keys() - values.keys():
        raise ValueError("initializers must name declared values")
    if initializers.keys() & set(graph["inputs"]):
        raise ValueError("initializers and runtime inputs must be disjoint")
    for name in initializers:
        if any(type(dim) is not int for dim in values[name]["shape"]):
            raise ValueError(f"{name}: initializer shapes must be concrete")
    contract = graph.get("shape_contract", {})
    if not isinstance(contract, Mapping) or contract.keys() - symbols:
        raise ValueError("shape_contract must name declared dimension symbols")
    for symbol, bounds in contract.items():
        if not isinstance(bounds, Mapping) or bounds.keys() - {"min", "max"}:
            raise ValueError(f"{symbol}: only min and max bounds are accepted")
        low, high = bounds.get("min", 0), bounds.get("max")
        if type(low) is not int or low < 0 or ("max" in bounds and (type(high) is not int or high < low)):
            raise ValueError(f"{symbol}: invalid symbol bounds")
    limits = dict(DEFAULT_LIMITS)
    requested = graph.get("limits", {})
    if not isinstance(requested, Mapping) or requested.keys() - limits.keys():
        raise ValueError("unknown work limit")
    limits.update(requested)
    if any(type(v) is not int or v <= 0 for v in limits.values()):
        raise ValueError("work limits must be positive integers")
    nodes = graph["nodes"]
    if not isinstance(nodes, (list, tuple)):
        raise ValueError("nodes must be a list")
    available = set(graph["inputs"]) | set(initializers)
    node_names = set()
    normalized_nodes = []
    for node in nodes:
        if not isinstance(node, Mapping) or not {"name", "op", "inputs", "output"} <= node.keys() or node.keys() - {"name", "op", "inputs", "output", "attrs"}:
            raise ValueError("node has missing or unknown fields")
        _name(node["name"], "node name")
        if node["name"] in node_names:
            raise ValueError(f"duplicate node name: {node['name']}")
        node_names.add(node["name"])
        if not isinstance(node["op"], str) or node["op"] not in OPS:
            raise ValueError(f"unsupported operation: {node['op']}")
        arities, allowed_attrs = OPS[node["op"]]
        if not isinstance(node["inputs"], (list, tuple)) or not node["inputs"] or any(not isinstance(n, str) for n in node["inputs"]) or (arities is not None and len(node["inputs"]) not in arities):
            raise ValueError(f"{node['name']}: invalid input arity")
        for name in node["inputs"]:
            if name not in available:
                raise ValueError(f"{node['name']}: input {name} is unavailable; graph must be topological SSA")
        output = node["output"]
        if not isinstance(output, str) or output not in values or output in available:
            raise ValueError(f"{node['name']}: output must be declared and assigned once")
        attrs = node.get("attrs", {})
        if not isinstance(attrs, Mapping) or attrs.keys() - allowed_attrs:
            raise ValueError(f"{node['name']}: unknown operation attribute")
        _json(attrs, f"{node['name']} attrs")
        normalized_nodes.append({"name": node["name"], "op": node["op"], "inputs": list(node["inputs"]),
                                 "output": output, "attrs": copy.deepcopy(dict(attrs))})
        available.add(output)
    if set(graph["outputs"]) - available:
        raise ValueError("graph output has no producer")
    if set(values) - available:
        raise ValueError("declared values without an input, initializer, or producer")
    _json(graph.get("metadata", {}), "metadata")
    return {"schema": SCHEMA, "values": {n: {"shape": list(s["shape"]), "dtype": s["dtype"]} for n, s in values.items()},
            "inputs": list(graph["inputs"]), "outputs": list(graph["outputs"]),
            "initializers": dict(initializers), "nodes": normalized_nodes,
            "shape_contract": copy.deepcopy(dict(contract)), "limits": limits,
            "metadata": copy.deepcopy(graph.get("metadata", {}))}

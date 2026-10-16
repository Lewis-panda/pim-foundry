"""Versioned, deterministic synthetic workloads and strict graph interchange.

Suite version 1 exercises GEMV, GEMM, a two-layer MLP and branching dataflow.
It is a functional research suite, not trained models or an application-quality
benchmark. Values and initializers are flat row-major signed int64 integers;
the execution contract uses exact integer intermediates.

The interchange API accepts arbitrary supported Graph objects, including multiple
runtime inputs. ``workload_from_dict`` accepts already parsed JSON objects: a
JSON-file reader must reject duplicate object keys before calling it, because
ordinary JSON parsing has already discarded that information.
"""

from math import prod
from random import Random

from .benchmark import content_hash
from .core import Graph, Node, Tensor, checked_integer


WORKLOAD_NAMES = ("gemv", "gemm", "mlp", "residual")
WORKLOAD_SUITE_VERSION = 1
WORKLOAD_SCHEMA_VERSION = 1


def _validate_inputs(graph: Graph, inputs: dict[str, tuple[int, ...]]) -> None:
    if not isinstance(graph, Graph):
        raise ValueError("workload graph must be a Graph")
    graph.validate()
    if type(inputs) is not dict or any(type(name) is not str for name in inputs):
        raise ValueError("workload inputs must be a dictionary with string names")
    if set(inputs) != set(graph.inputs):
        raise ValueError("provide exactly the graph's named inputs")
    for name, values in inputs.items():
        if type(values) is not tuple or len(values) != prod(graph.tensor_map[name].shape):
            raise ValueError(f"input {name!r} must be a flat tuple matching its tensor shape")
        for value in values:
            checked_integer(value)


def make_workload(name: str, scale: int = 1, seed: int = 0) -> tuple[Graph, dict[str, tuple[int, ...]]]:
    """Build a suite-v1 graph; all dimensions scale linearly by 1 through 4.

    A local Random instance produces values in [-2, 2] without changing global
    random state. Seed is an integer in [0, 2**32-1]. Dimensions at every allowed
    scale satisfy the reference graph work limit; generated results fit int64.
    """
    if type(name) is not str or name not in WORKLOAD_NAMES:
        raise ValueError("unknown workload; choose gemv, gemm, mlp or residual")
    if type(scale) is not int or not 1 <= scale <= 4:
        raise ValueError("scale must be an integer between 1 and 4")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    rng = Random(seed)

    def values(shape):
        return tuple(rng.randrange(-2, 3) for _ in range(prod(shape)))

    if name == "gemv":
        k, n = 64 * scale, 48 * scale
        graph = Graph(
            (Tensor("x", (k,)), Tensor("w", (k, n), values((k, n))), Tensor("y", (n,))),
            (Node("linear", "matvec", ("x", "w"), "y"),), ("x",), ("y",),
        )
    elif name == "gemm":
        m, k, n = 8 * scale, 32 * scale, 48 * scale
        graph = Graph(
            (Tensor("x", (m, k)), Tensor("w", (k, n), values((k, n))), Tensor("y", (m, n))),
            (Node("linear", "matmul", ("x", "w"), "y"),), ("x",), ("y",),
        )
    elif name == "mlp":
        m, k, h, n = 8 * scale, 32 * scale, 48 * scale, 24 * scale
        graph = Graph(
            (Tensor("x", (m, k)), Tensor("w1", (k, h), values((k, h))),
             Tensor("hidden", (m, h)), Tensor("activated", (m, h)),
             Tensor("w2", (h, n), values((h, n))), Tensor("y", (m, n))),
            (Node("linear1", "matmul", ("x", "w1"), "hidden"),
             Node("activation", "relu", ("hidden",), "activated"),
             Node("linear2", "matmul", ("activated", "w2"), "y")),
            ("x",), ("y",),
        )
    else:
        m, k, n = 8 * scale, 32 * scale, 32 * scale
        graph = Graph(
            (Tensor("x", (m, k)), Tensor("w1", (k, n), values((k, n))),
             Tensor("w2", (k, n), values((k, n))), Tensor("left", (m, n)),
             Tensor("right", (m, n)), Tensor("combined", (m, n)), Tensor("y", (m, n))),
            (Node("left_linear", "matmul", ("x", "w1"), "left"),
             Node("right_linear", "matmul", ("x", "w2"), "right"),
             Node("merge", "add", ("left", "right"), "combined"),
             Node("activation", "relu", ("combined",), "y")),
            ("x",), ("y",),
        )
    inputs = {"x": values(graph.tensor_map["x"].shape)}
    _validate_inputs(graph, inputs)
    return graph, inputs


def workload_to_dict(graph: Graph, inputs: dict[str, tuple[int, ...]]) -> dict:
    """Export a validated supported graph and all its runtime input values.

    The returned object contains only JSON primitives, with tuples represented
    by arrays and absent initializers represented by null. Generator metadata
    belongs in the surrounding experiment manifest, outside this strict schema.
    """
    _validate_inputs(graph, inputs)
    return {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "graph": {
            "tensors": [{"name": tensor.name, "shape": list(tensor.shape),
                         "data": None if tensor.data is None else list(tensor.data)}
                        for tensor in graph.tensors],
            "nodes": [{"name": node.name, "op": node.op, "inputs": list(node.inputs),
                       "output": node.output} for node in graph.nodes],
            "inputs": list(graph.inputs), "outputs": list(graph.outputs),
        },
        "inputs": {name: list(inputs[name]) for name in graph.inputs},
    }


def _object(value, expected: set[str], label: str) -> dict:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    if set(value) != expected:
        raise ValueError(f"{label} must contain exactly these fields: {', '.join(sorted(expected))}")
    return value


def _array(value, label: str) -> list:
    if type(value) is not list:
        raise ValueError(f"{label} must be a JSON array")
    return value


def _text(value, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _names(value, label: str) -> tuple[str, ...]:
    return tuple(_text(name, label) for name in _array(value, label))


def workload_from_dict(data: dict) -> tuple[Graph, dict[str, tuple[int, ...]]]:
    """Import strict schema-v1 JSON; unknown/missing fields are rejected.

    JSON arrays must be flat and contain actual integers, not booleans or floats.
    Graph validation checks ranks, dimensions, shapes, topology, operators and
    the reference work limit. Runtime values must match every named graph input.
    Import validates structure and int64 input storage, not the arithmetic output
    of an arbitrary graph; execution is responsible for detecting output overflow.
    """
    _object(data, {"schema_version", "graph", "inputs"}, "workload")
    if type(data["schema_version"]) is not int or data["schema_version"] != WORKLOAD_SCHEMA_VERSION:
        raise ValueError("unsupported workload schema_version")
    description = _object(data["graph"], {"tensors", "nodes", "inputs", "outputs"}, "graph")
    tensors = []
    for item in _array(description["tensors"], "graph.tensors"):
        _object(item, {"name", "shape", "data"}, "tensor")
        shape = tuple(_array(item["shape"], "tensor.shape"))
        initializer = item["data"]
        if initializer is not None:
            initializer = tuple(_array(initializer, "tensor.data"))
        tensors.append(Tensor(_text(item["name"], "tensor.name"), shape, initializer))
    nodes = []
    for item in _array(description["nodes"], "graph.nodes"):
        _object(item, {"name", "op", "inputs", "output"}, "node")
        nodes.append(Node(_text(item["name"], "node.name"), _text(item["op"], "node.op"),
                          _names(item["inputs"], "node.inputs"), _text(item["output"], "node.output")))
    graph = Graph(tuple(tensors), tuple(nodes), _names(description["inputs"], "graph.inputs"),
                  _names(description["outputs"], "graph.outputs"))
    graph.validate()
    raw_inputs = _object(data["inputs"], set(graph.inputs), "inputs")
    inputs = {name: tuple(_array(values, f"input {name!r}")) for name, values in raw_inputs.items()}
    _validate_inputs(graph, inputs)
    return graph, inputs


def workload_identity(graph: Graph, inputs: dict[str, tuple[int, ...]]) -> str:
    """SHA-256 of the canonical schema-v1 graph and its concrete runtime inputs."""
    return content_hash(workload_to_dict(graph, inputs))

"""Shared functional contracts, independent of any vendor instruction set.

The initial reference domain stores signed int64 values and uses exact integer
intermediates, checking the final result of each node. It is NOT a claim about
the arithmetic supported by SRAM or HBM hardware.
"""

from dataclasses import dataclass
from math import prod


def checked_integer(value: int) -> int:
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError("tensor values must be signed int64 integers")
    return value


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    data: tuple[int, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tensor name must be nonempty")
        if type(self.shape) is not tuple or len(self.shape) not in (1, 2) or any(
            type(d) is not int or d <= 0 for d in self.shape
        ):
            raise ValueError("tensor shape must be a tuple of one or two positive integers")
        if prod(self.shape) > 1_000_000:
            raise ValueError("reference tensor exceeds the 1000000-element limit")
        if self.data is not None:
            if type(self.data) is not tuple or len(self.data) != prod(self.shape):
                raise ValueError("initializer must be a flat tuple matching the tensor shape")
            for value in self.data:
                checked_integer(value)

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * 8


@dataclass(frozen=True)
class Node:
    name: str
    op: str
    inputs: tuple[str, ...]
    output: str


@dataclass(frozen=True)
class Graph:
    tensors: tuple[Tensor, ...]
    nodes: tuple[Node, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def validate(self) -> None:
        if type(self.tensors) is not tuple or any(not isinstance(t, Tensor) for t in self.tensors):
            raise ValueError("graph tensors must be a tuple of Tensor objects")
        if type(self.nodes) is not tuple or any(not isinstance(n, Node) for n in self.nodes):
            raise ValueError("graph nodes must be a tuple of Node objects")
        if any(type(names) is not tuple or any(not isinstance(name, str) for name in names)
               for names in (self.inputs, self.outputs)):
            raise ValueError("graph input/output names must be tuples of strings")
        if not self.nodes or not self.outputs:
            raise ValueError("graph must have nodes and outputs")
        tensors = self.tensor_map
        if len(tensors) != len(self.tensors):
            raise ValueError("tensor names must be unique")
        if len(set(self.inputs)) != len(self.inputs) or len(set(self.outputs)) != len(self.outputs):
            raise ValueError("graph input/output names must be unique")
        for name in self.inputs:
            if name not in tensors or tensors[name].data is not None:
                raise ValueError("graph inputs must name tensors without initializers")
        defined = set(self.inputs) | {t.name for t in self.tensors if t.data is not None}
        node_names = set()
        work = 0
        for node in self.nodes:
            if not isinstance(node.name, str) or not node.name or node.name in node_names:
                raise ValueError("node names must be nonempty and unique")
            node_names.add(node.name)
            if type(node.inputs) is not tuple or any(
                not isinstance(name, str) or name not in defined for name in node.inputs
            ):
                raise ValueError(f"{node.name}: inputs must be defined before use")
            if not isinstance(node.output, str) or node.output not in tensors or node.output in defined:
                raise ValueError(f"{node.name}: output must be declared and have a single producer")
            shapes = [tensors[name].shape for name in node.inputs]
            result_shape = tensors[node.output].shape
            if node.op in ("matvec", "matmul"):
                input_rank = 1 if node.op == "matvec" else 2
                if len(shapes) != 2 or len(shapes[0]) != input_rank or len(shapes[1]) != 2:
                    raise ValueError(f"{node.op} needs rank-{input_rank} input and weights[K,N]")
                expected = (*shapes[0][:-1], shapes[1][1])
                if shapes[0][-1] != shapes[1][0] or result_shape != expected:
                    raise ValueError(f"{node.op} shape mismatch")
                if tensors[node.inputs[1]].data is None:
                    raise ValueError(f"reference {node.op} currently requires constant weights")
                work += prod(shapes[0]) * shapes[1][1]
            elif node.op == "relu":
                if len(shapes) != 1 or result_shape != shapes[0]:
                    raise ValueError("relu requires one rank-1/2 input with matching output shape")
                work += prod(shapes[0])
            elif node.op == "add":
                if len(shapes) != 2 or shapes[0] != shapes[1] or result_shape != shapes[0]:
                    raise ValueError("add requires two inputs and an output with identical shapes")
                work += prod(shapes[0])
            else:
                raise ValueError(f"unsupported operation: {node.op}")
            defined.add(node.output)
        if any(name not in defined for name in self.outputs):
            raise ValueError("graph outputs must be defined")
        if any(t.name not in defined for t in self.tensors):
            raise ValueError("graph has undeclared external inputs or unproduced tensors")
        if work > 2_000_000:
            raise ValueError("graph exceeds the reference work limit")

    @property
    def tensor_map(self) -> dict[str, Tensor]:
        return {tensor.name: tensor for tensor in self.tensors}


@dataclass(frozen=True)
class ReferenceDevice:
    """Illustrative logical resources, not a measured or vendor hardware model."""

    name: str
    kind: str
    capacity_bytes: int
    tile_k: int
    tile_n: int
    units: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.name == "host":
            raise ValueError("device name must be nonempty and different from host")
        if type(self.kind) is not str or self.kind not in ("sram", "hbm", "gpu", "npu"):
            raise ValueError("reference kind must be sram, hbm, gpu or npu")
        for value in (self.capacity_bytes, self.tile_k, self.tile_n, self.units):
            if type(value) is not int or value <= 0:
                raise ValueError("reference resource counts must be positive integers")


@dataclass(frozen=True)
class KernelResult:
    values: tuple[int, ...]
    events: tuple[dict, ...]


@dataclass(frozen=True)
class Platform:
    devices: tuple[ReferenceDevice, ...]
    links: tuple[tuple[str, str], ...]

    def validate(self):
        if type(self.devices) is not tuple or any(not isinstance(d, ReferenceDevice) for d in self.devices):
            raise ValueError("platform devices must be a tuple of ReferenceDevice objects")
        if type(self.links) is not tuple:
            raise ValueError("platform links must be a tuple")
        names = [d.name for d in self.devices]
        if not names or len(set(names)) != len(names):
            raise ValueError("platform device names must be nonempty and unique")
        endpoints = set(names) | {"host"}
        for link in self.links:
            if type(link) is not tuple or len(link) != 2 or any(
                not isinstance(x, str) or x not in endpoints for x in link
            ):
                raise ValueError("links must name two known endpoints")
            if link[0] == link[1]:
                raise ValueError("self links are implicit and must not be declared")
        if len(set(self.links)) != len(self.links):
            raise ValueError("links must be unique")

    def path(self, source: str, destination: str) -> tuple[str, ...]:
        """Deterministic shortest-hop path, with no latency optimization claim."""
        self.validate()
        endpoints = {d.name for d in self.devices} | {"host"}
        if source not in endpoints or destination not in endpoints:
            raise ValueError("unknown transfer endpoint")
        queue = [(source,)]
        seen = {source}
        for route in queue:
            if route[-1] == destination:
                return route
            for a, b in sorted(self.links):
                if a == route[-1] and b not in seen:
                    seen.add(b)
                    queue.append((*route, b))
        raise ValueError(f"no transfer route from {source} to {destination}")

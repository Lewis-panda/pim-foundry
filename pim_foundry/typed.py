"""Typed tensor semantics shared by backend lowerers and a scalar reference.

This module describes arithmetic, not hardware support or timing. Floating point
MACs are explicitly non-fused: binary32-rounded product, then binary32-rounded
addition, in increasing K order. Integer accumulators apply their overflow policy
at every addition. Tensor storage is row-major and there are no implicit casts
between tensors of different dtypes.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
from math import prod
import struct
from typing import Mapping

DTYPES = {"int8": 1, "int32": 4, "float16": 2, "float32": 4}
MAX_TENSOR_ELEMENTS = 1_000_000
MAX_REFERENCE_OPERATIONS = 20_000_000


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _name(value, label):
    if type(value) is not str or not value or len(value) > 128:
        raise ValueError(f"{label} must be a nonempty string of at most 128 characters")


def round_float(value, dtype):
    """Round a finite scalar to IEEE binary16/binary32, ties to even.

    Underflow and subnormals follow Python's IEEE struct conversion; signed zero
    is retained. Overflow and all nonfinite values are rejected, never clipped.
    A number in a floating tensor initializer is converted because its storage
    dtype was explicitly declared; this does not permit mixed-dtype graph edges.
    """
    if dtype not in ("float16", "float32"):
        raise ValueError("floating conversion requires float16 or float32")
    if type(value) not in (int, float):
        raise ValueError("floating tensor values must be finite numbers")
    try:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("nonfinite values are forbidden by the numeric contract")
        fmt = "<e" if dtype == "float16" else "<f"
        rounded = struct.unpack(fmt, struct.pack(fmt, value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{dtype} overflow") from exc
    if not math.isfinite(rounded):
        raise ValueError(f"{dtype} overflow")
    return rounded


def storage_value(value, dtype):
    """Validate an integer or explicitly round a declared floating storage value."""
    if dtype == "int8":
        return _integer(value, -128, 127, "int8 value")
    if dtype == "int32":
        return _integer(value, -(2**31), 2**31 - 1, "int32 value")
    if dtype in ("float16", "float32"):
        return round_float(value, dtype)
    raise ValueError(f"unsupported dtype: {dtype}")


@dataclass(frozen=True)
class Requantization:
    """INT32/finite float -> INT8: clamp(round_even(x*multiplier/2**shift)+zero_point).

    The multiplication and rational rounding use exact integers; this is an
    explicitly specified software policy, not an implied native instruction.
    """

    multiplier: int = 1
    shift: int = 0
    zero_point: int = 0

    def __post_init__(self):
        _integer(self.multiplier, 1, 2**31 - 1, "requantization multiplier")
        _integer(self.shift, 0, 62, "requantization shift")
        _integer(self.zero_point, -128, 127, "requantization zero point")

    def apply(self, value):
        if type(value) is int:
            _integer(value, -(2**31), 2**31 - 1, "requantization input")
            numerator, denominator = value, 1
        elif type(value) is float and math.isfinite(value):
            numerator, denominator = value.as_integer_ratio()
        else:
            raise ValueError("requantization input must be int32 or a finite floating value")
        numerator *= self.multiplier
        denominator *= 1 << self.shift
        quotient, remainder = divmod(abs(numerator), denominator)
        if 2 * remainder > denominator or (2 * remainder == denominator and quotient % 2):
            quotient += 1
        rounded = -quotient if numerator < 0 else quotient
        return min(127, max(-128, rounded + self.zero_point))


@dataclass(frozen=True)
class Arithmetic:
    accumulator: str = "int32"
    overflow: str = "error"
    nonfinite: str = "error"
    rounding: str = "nearest_even"
    requantize: Requantization | None = None
    mac: str = "separate_multiply_add"

    def __post_init__(self):
        if self.accumulator not in ("int32", "float32"):
            raise ValueError("accumulator must be int32 or float32")
        if self.overflow not in ("error", "saturate", "wrap"):
            raise ValueError("overflow must be error, saturate or wrap")
        if self.nonfinite != "error" or self.rounding != "nearest_even":
            raise ValueError("only finite values and round-to-nearest-even are supported")
        if self.mac != "separate_multiply_add":
            raise ValueError("only explicit separate multiply/add arithmetic is supported")
        if self.requantize is not None and not isinstance(self.requantize, Requantization):
            raise ValueError("requantize must be Requantization or None")
        if self.accumulator == "float32" and (self.overflow != "error" or self.requantize is not None):
            raise ValueError("floating arithmetic requires overflow=error and no integer requantization")


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    data: tuple[int | float, ...] | None = None

    def __post_init__(self):
        _name(self.name, "tensor name")
        if type(self.shape) is not tuple or len(self.shape) not in (1, 2) or any(
            type(x) is not int or x <= 0 for x in self.shape
        ):
            raise ValueError("tensor shape must contain one or two positive integer dimensions")
        if prod(self.shape) > MAX_TENSOR_ELEMENTS:
            raise ValueError("tensor exceeds the 1000000-element limit")
        if type(self.dtype) is not str or self.dtype not in DTYPES:
            raise ValueError("tensor dtype must be int8, int32, float16 or float32")
        if self.data is not None:
            if type(self.data) is not tuple or len(self.data) != prod(self.shape):
                raise ValueError("tensor initializer must be a flat tuple matching its shape")
            object.__setattr__(self, "data", tuple(storage_value(x, self.dtype) for x in self.data))

    @property
    def nbytes(self):
        return prod(self.shape) * DTYPES[self.dtype]


@dataclass(frozen=True)
class Node:
    name: str
    op: str
    inputs: tuple[str, ...]
    output: str
    arithmetic: Arithmetic = field(default_factory=Arithmetic)

    def __post_init__(self):
        _name(self.name, "node name")
        _name(self.output, "output name")
        if self.op not in ("matvec", "matmul", "relu", "add", "cast"):
            raise ValueError(f"unsupported operation: {self.op}")
        if type(self.inputs) is not tuple or any(type(x) is not str or not x for x in self.inputs):
            raise ValueError("node inputs must be a tuple of nonempty names")
        if not isinstance(self.arithmetic, Arithmetic):
            raise ValueError("node arithmetic must be an Arithmetic policy")


def validate_node(node, tensors):
    if any(name not in tensors for name in (*node.inputs, node.output)):
        raise ValueError(f"{node.name}: every tensor must be declared")
    inputs = [tensors[name] for name in node.inputs]
    output = tensors[node.output]
    if node.op in ("matvec", "matmul"):
        rank = 1 if node.op == "matvec" else 2
        if len(inputs) != 2 or len(inputs[0].shape) != rank or len(inputs[1].shape) != 2:
            raise ValueError(f"{node.name}: {node.op} requires rank-{rank} input and weights[K,N]")
        a, b = inputs
        if a.shape[-1] != b.shape[0] or output.shape != (*a.shape[:-1], b.shape[1]):
            raise ValueError(f"{node.name}: matrix dimensions do not agree")
        if a.dtype not in ("int8", "float16", "float32"):
            raise ValueError(f"{node.name}: linear inputs must be int8, float16 or float32")
    elif node.op in ("relu", "cast"):
        if len(inputs) != 1 or inputs[0].shape != output.shape:
            raise ValueError(f"{node.name}: {node.op} requires one equally shaped input")
    elif len(inputs) != 2 or inputs[0].shape != inputs[1].shape or inputs[0].shape != output.shape:
        raise ValueError(f"{node.name}: add requires two equally shaped inputs")
    if any(t.dtype != inputs[0].dtype for t in inputs):
        raise ValueError(f"{node.name}: mixed input dtypes require an explicit conversion outside this IR")
    source_dtype = inputs[0].dtype
    arithmetic = node.arithmetic
    if node.op == "cast":
        if output.dtype.startswith("float"):
            if arithmetic.accumulator != "float32":
                raise ValueError(f"{node.name}: floating cast requires accumulator=float32")
        else:
            if arithmetic.accumulator != "int32":
                raise ValueError(f"{node.name}: integer cast requires accumulator=int32")
            if output.dtype == "int8" and arithmetic.requantize is None and source_dtype != "int8":
                raise ValueError(f"{node.name}: cast to int8 requires explicit requantization")
            if output.dtype == "int32" and arithmetic.requantize is not None:
                raise ValueError(f"{node.name}: requantization requires int8 output")
        return
    if source_dtype.startswith("int"):
        if arithmetic.accumulator != "int32" or output.dtype not in ("int8", "int32"):
            raise ValueError(f"{node.name}: integer arithmetic requires int32 accumulation and integer output")
        identity_relu = node.op == "relu" and source_dtype == output.dtype == "int8"
        if output.dtype == "int8" and arithmetic.requantize is None and not identity_relu:
            raise ValueError(f"{node.name}: int8 output requires explicit requantization")
        if output.dtype == "int32" and arithmetic.requantize is not None:
            raise ValueError(f"{node.name}: requantization requires int8 output")
    elif arithmetic.accumulator != "float32" or output.dtype not in ("float16", "float32"):
        raise ValueError(f"{node.name}: floating arithmetic requires float32 accumulation and floating output")


def node_operations(node, tensors):
    if node.op in ("matvec", "matmul"):
        return prod(tensors[node.inputs[0]].shape) * tensors[node.inputs[1]].shape[1]
    return prod(tensors[node.output].shape)


@dataclass(frozen=True)
class Graph:
    tensors: tuple[Tensor, ...]
    nodes: tuple[Node, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    @property
    def tensor_map(self):
        return {tensor.name: tensor for tensor in self.tensors}

    def validate(self):
        if type(self.tensors) is not tuple or not self.tensors or any(not isinstance(t, Tensor) for t in self.tensors):
            raise ValueError("graph tensors must be a nonempty tuple of Tensor objects")
        if type(self.nodes) is not tuple or not self.nodes or len(self.nodes) > 256 or any(
            not isinstance(node, Node) for node in self.nodes
        ):
            raise ValueError("graph needs between 1 and 256 Node objects")
        for names in (self.inputs, self.outputs):
            if type(names) is not tuple or any(type(name) is not str for name in names) or len(set(names)) != len(names):
                raise ValueError("graph input/output names must be unique tuples of strings")
        if not self.outputs:
            raise ValueError("graph needs at least one output")
        tensors = self.tensor_map
        if len(tensors) != len(self.tensors):
            raise ValueError("tensor names must be unique")
        if any(name not in tensors or tensors[name].data is not None for name in self.inputs):
            raise ValueError("graph inputs must name declared tensors without initializers")
        defined = set(self.inputs) | {tensor.name for tensor in self.tensors if tensor.data is not None}
        seen = set()
        for node in self.nodes:
            if node.name in seen:
                raise ValueError("node names must be unique")
            seen.add(node.name)
            if any(name not in defined for name in node.inputs):
                raise ValueError(f"{node.name}: every input must be defined before use")
            if node.output in defined:
                raise ValueError(f"{node.name}: output must have a single producer")
            validate_node(node, tensors)
            defined.add(node.output)
        if any(name not in defined for name in self.outputs) or defined != set(tensors):
            raise ValueError("all tensors and graph outputs must be defined")


def _accumulate(value, arithmetic):
    if arithmetic.accumulator == "float32":
        return round_float(value, "float32")
    if -(2**31) <= value < 2**31:
        return value
    if arithmetic.overflow == "error":
        raise ValueError("int32 accumulator overflow")
    if arithmetic.overflow == "saturate":
        return min(2**31 - 1, max(-(2**31), value))
    return (value + 2**31) % 2**32 - 2**31


def _output(value, tensor, arithmetic):
    if arithmetic.requantize is not None:
        return arithmetic.requantize.apply(value)
    return storage_value(value, tensor.dtype)


def prepare_inputs(graph, inputs):
    """Validate external inputs and return all initial values, including constants."""
    graph.validate()
    if not isinstance(inputs, Mapping) or set(inputs) != set(graph.inputs):
        raise ValueError("input mapping must contain exactly the declared graph inputs")
    values = {tensor.name: tensor.data for tensor in graph.tensors if tensor.data is not None}
    for name in graph.inputs:
        tensor = graph.tensor_map[name]
        data = inputs[name]
        if type(data) not in (tuple, list) or len(data) != prod(tensor.shape):
            raise ValueError(f"{name}: expected a flat input sequence matching {tensor.shape}")
        values[name] = tuple(storage_value(value, tensor.dtype) for value in data)
    return values


def execute_node(node, tensors, values):
    """Return one output tuple; validate dtype, shape and scalar input values."""
    validate_node(node, tensors)
    if node_operations(node, tensors) > MAX_REFERENCE_OPERATIONS:
        raise ValueError("node exceeds the scalar reference operation limit")
    args = []
    for name in node.inputs:
        tensor = tensors[name]
        if name not in values or type(values[name]) not in (list, tuple) or len(values[name]) != prod(tensor.shape):
            raise ValueError(f"{name}: missing or incorrectly sized node input")
        args.append(tuple(storage_value(value, tensor.dtype) for value in values[name]))
    arithmetic = node.arithmetic
    floating = arithmetic.accumulator == "float32"
    output = tensors[node.output]
    result = []
    if node.op in ("matvec", "matmul"):
        a, b = args
        shape = tensors[node.inputs[0]].shape
        k = shape[-1]
        n = tensors[node.inputs[1]].shape[1]
        m = 1 if node.op == "matvec" else shape[0]
        for row in range(m):
            for col in range(n):
                total = 0.0 if floating else 0
                for index in range(k):
                    product = a[row * k + index] * b[index * n + col]
                    if floating:
                        product = round_float(product, "float32")
                    total = _accumulate(total + product, arithmetic)
                result.append(_output(total, output, arithmetic))
    elif node.op == "cast":
        for value in args[0]:
            if output.dtype == "int32":
                value = _accumulate(round(value) if type(value) is float else value, arithmetic)
            result.append(_output(value, output, arithmetic))
    elif node.op == "add":
        for a, b in zip(*args):
            result.append(_output(_accumulate(a + b, arithmetic), output, arithmetic))
    else:
        for value in args[0]:
            # ReLU defines a positive zero for both negative values and -0.0.
            value = value if value > 0 else (0.0 if floating else 0)
            result.append(_output(value, output, arithmetic))
    return tuple(result)


def evaluate(graph, inputs, *, max_operations=MAX_REFERENCE_OPERATIONS):
    """Return every tensor value with a bounded deterministic scalar evaluation."""
    values = prepare_inputs(graph, inputs)
    _integer(max_operations, 1, MAX_REFERENCE_OPERATIONS, "reference operation limit")
    if sum(node_operations(node, graph.tensor_map) for node in graph.nodes) > max_operations:
        raise ValueError("graph exceeds the scalar reference operation limit")
    for node in graph.nodes:
        values[node.output] = execute_node(node, graph.tensor_map, values)
    return values


def execute(graph, inputs, *, max_operations=MAX_REFERENCE_OPERATIONS):
    values = evaluate(graph, inputs, max_operations=max_operations)
    return {name: values[name] for name in graph.outputs}


def compare_outputs(graph, actual, expected, *, atol=0.0, rtol=0.0):
    """Compare exact integer results or finite floats under an explicit tolerance.

    This checks values supplied by the caller; it does not certify that a timing
    simulator produced them. Floating results must already have their declared
    storage precision; rounding an invalid result here would conceal a mismatch.
    """
    graph.validate()
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ValueError("comparison tolerances must be finite nonnegative numbers")
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping) or set(actual) != set(graph.outputs) or set(expected) != set(graph.outputs):
        raise ValueError("comparison mappings must contain exactly the graph outputs")
    failures = []
    compared = 0
    max_absolute_error = 0.0
    for name in graph.outputs:
        tensor = graph.tensor_map[name]
        count = prod(tensor.shape)
        a, e = actual[name], expected[name]
        if type(a) not in (tuple, list) or type(e) not in (tuple, list) or len(a) != count or len(e) != count:
            raise ValueError(f"{name}: comparison shape mismatch")
        for index, (value, reference) in enumerate(zip(a, e)):
            checked_reference = storage_value(reference, tensor.dtype)
            if checked_reference != reference:
                raise ValueError(f"{name}: expected value does not match declared storage precision")
            try:
                checked = storage_value(value, tensor.dtype)
                representable = checked == value
                error = abs(value - reference)
                matches = representable and (value == reference if tensor.dtype.startswith("int") else error <= atol + rtol * abs(reference))
            except (ValueError, TypeError, OverflowError):
                matches, error = False, None
            compared += 1
            if error is not None and math.isfinite(error):
                max_absolute_error = max(max_absolute_error, error)
            if not matches and len(failures) < 20:
                failures.append({"tensor": name, "index": index, "actual": repr(value), "expected": reference})
    return {"passed": not failures, "compared_elements": compared, "max_absolute_error": max_absolute_error,
            "atol": atol, "rtol": rtol, "mismatches": failures,
            "policy": "exact_integer_finite_float_declared_storage_precision"}


@dataclass(frozen=True)
class BackendCapabilities:
    """Declared support used for rejection; it is not evidence of native execution."""

    name: str
    operations: tuple[str, ...]
    input_dtypes: tuple[str, ...]
    output_dtypes: tuple[str, ...]
    accumulators: tuple[str, ...]
    constraints: tuple[str, ...] = ()

    def __post_init__(self):
        _name(self.name, "backend name")
        for values, allowed, label in (
            (self.operations, ("matvec", "matmul", "relu", "add", "cast"), "operations"),
            (self.input_dtypes, tuple(DTYPES), "input dtypes"),
            (self.output_dtypes, tuple(DTYPES), "output dtypes"),
            (self.accumulators, ("int32", "float32"), "accumulators"),
        ):
            if type(values) is not tuple or any(type(x) is not str or x not in allowed for x in values) or len(set(values)) != len(values):
                raise ValueError(f"backend {label} must be a tuple of unique supported names")
        if type(self.constraints) is not tuple or any(type(x) is not str for x in self.constraints):
            raise ValueError("backend constraints must be a tuple of descriptions")

    def check(self, node, tensors):
        validate_node(node, tensors)
        reasons = []
        if node.op not in self.operations:
            reasons.append(f"operation {node.op} is unsupported")
        if any(tensors[name].dtype not in self.input_dtypes for name in node.inputs):
            reasons.append("input dtype is unsupported")
        if tensors[node.output].dtype not in self.output_dtypes:
            reasons.append("output dtype is unsupported")
        if node.arithmetic.accumulator not in self.accumulators:
            reasons.append("accumulator dtype is unsupported")
        return tuple(reasons)


def graph_to_dict(graph):
    graph.validate()
    return {"schema": "pim-foundry.typed-graph.v1", "layout": "row_major",
            "tensors": [{"name": t.name, "shape": list(t.shape), "dtype": t.dtype,
                         "data": None if t.data is None else list(t.data)} for t in graph.tensors],
            "nodes": [{"name": n.name, "op": n.op, "inputs": list(n.inputs), "output": n.output,
                       "arithmetic": {"accumulator": n.arithmetic.accumulator, "overflow": n.arithmetic.overflow,
                                      "nonfinite": n.arithmetic.nonfinite, "rounding": n.arithmetic.rounding,
                                      "mac": n.arithmetic.mac,
                                      "requantize": None if n.arithmetic.requantize is None else {
                                          "multiplier": n.arithmetic.requantize.multiplier,
                                          "shift": n.arithmetic.requantize.shift,
                                          "zero_point": n.arithmetic.requantize.zero_point}}} for n in graph.nodes],
            "inputs": list(graph.inputs), "outputs": list(graph.outputs)}


def _fields(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label} requires exactly these fields: {', '.join(keys)}")


def _list(value, label):
    if type(value) is not list:
        raise ValueError(f"{label} must be a JSON array")
    return value


def graph_from_dict(data):
    _fields(data, ("schema", "layout", "tensors", "nodes", "inputs", "outputs"), "typed graph")
    if data["schema"] != "pim-foundry.typed-graph.v1" or data["layout"] != "row_major":
        raise ValueError("unsupported typed graph schema or layout")
    tensors = []
    for tensor in _list(data["tensors"], "tensors"):
        _fields(tensor, ("name", "shape", "dtype", "data"), "tensor")
        values = None if tensor["data"] is None else tuple(_list(tensor["data"], "tensor data"))
        tensors.append(Tensor(tensor["name"], tuple(_list(tensor["shape"], "shape")), tensor["dtype"], values))
    nodes = []
    for node in _list(data["nodes"], "nodes"):
        _fields(node, ("name", "op", "inputs", "output", "arithmetic"), "node")
        arithmetic = node["arithmetic"]
        _fields(arithmetic, ("accumulator", "overflow", "nonfinite", "rounding", "mac", "requantize"), "arithmetic")
        quantization = arithmetic["requantize"]
        if quantization is not None:
            _fields(quantization, ("multiplier", "shift", "zero_point"), "requantization")
            quantization = Requantization(**quantization)
        policy = Arithmetic(**{**arithmetic, "requantize": quantization})
        nodes.append(Node(node["name"], node["op"], tuple(_list(node["inputs"], "node inputs")), node["output"], policy))
    graph = Graph(tuple(tensors), tuple(nodes), tuple(_list(data["inputs"], "graph inputs")), tuple(_list(data["outputs"], "graph outputs")))
    graph.validate()
    return graph


def graph_digest(graph):
    canonical = json.dumps(graph_to_dict(graph), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

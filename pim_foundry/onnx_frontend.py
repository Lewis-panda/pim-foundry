"""Optional, strict ONNX frontend for bounded feed-forward applications.

Importing this module does not import ONNX or NumPy. The importer preserves
declared tensor storage types and never inserts backend precision conversions.
It selects Foundry's explicit scalar reduction policy; floating equivalence to
another ONNX implementation must be checked for the actual model and inputs.
"""

from dataclasses import dataclass, field
import hashlib
from math import prod
from pathlib import Path

from .typed import Arithmetic, Graph, Node, Tensor, graph_digest, prepare_inputs as typed_prepare_inputs, compare_outputs

MAX_MODEL_BYTES = 32 * 1024 * 1024
MAX_ELEMENTS = 1_000_000
SUPPORTED_OPS = ("MatMul", "Gemm", "Add", "Relu", "Cast", "Flatten", "Identity")


class OnnxImportError(ValueError):
    """A model cannot be faithfully represented by this frontend's subset."""


@dataclass(frozen=True)
class ImportedModel:
    graph: Graph
    input_schema: dict
    report: dict
    source_bytes: bytes = field(repr=False)

    def prepare_inputs(self, raw_inputs):
        """Validate original-shaped or flat application values for the typed graph."""
        _, np = _dependencies()
        if type(raw_inputs) is not dict or set(raw_inputs) != set(self.input_schema):
            raise OnnxImportError("application inputs must contain exactly the ONNX graph input names")
        flattened = {}
        for name, schema in self.input_schema.items():
            array = np.asarray(raw_inputs[name])
            shape = tuple(schema["shape"])
            if array.shape not in (shape, (prod(shape),)):
                raise OnnxImportError(f"input {name!r}: expected source shape {shape} or a flat list of {prod(shape)} values")
            flattened[name] = array.reshape(-1).tolist()
        values = typed_prepare_inputs(self.graph, flattened)
        return {name: values[name] for name in self.graph.inputs}

    def reference_outputs(self, raw_inputs):
        """Run ONNX's independent reference evaluator on the original model bytes."""
        onnx, np = _dependencies()
        from onnx.reference import ReferenceEvaluator
        prepared = self.prepare_inputs(raw_inputs)
        feeds = {name: np.asarray(prepared[name], dtype=schema["dtype"]).reshape(schema["shape"])
                 for name, schema in self.input_schema.items()}
        evaluator = ReferenceEvaluator(onnx.load_model_from_string(self.source_bytes))
        outputs = evaluator.run(None, feeds)
        result = {}
        for name, output in zip(evaluator.output_names, outputs):
            schema = self.report["output_mapping"][name]
            if tuple(output.shape) != tuple(schema["shape"]) or output.dtype != np.dtype(schema["dtype"]):
                raise OnnxImportError(f"reference output {name!r} violates the declared output schema")
            if not np.isfinite(output).all():
                raise OnnxImportError(f"reference output {name!r} contains nonfinite values")
            result[name] = tuple(output.reshape(-1).tolist())
        return result

    def compare_reference(self, raw_inputs, actual_typed_outputs, *, atol, rtol):
        """Compare actual shared-IR/backend outputs with the original ONNX model."""
        onnx, np = _dependencies()
        source_expected = self.reference_outputs(raw_inputs)
        expected = {}
        for name, spec in self.report["output_mapping"].items():
            target = spec["tensor"]
            if target in expected and expected[target] != source_expected[name]:
                raise OnnxImportError("aliased source outputs disagree in the ONNX reference evaluator")
            expected[target] = source_expected[name]
        result = compare_outputs(self.graph, actual_typed_outputs, expected, atol=atol, rtol=rtol)
        return {**result, "reference": "onnx.reference.ReferenceEvaluator", "onnx_version": onnx.__version__,
                "numpy_version": np.__version__, "source_sha256": self.report["source_sha256"],
                "scope": "these_input_values_only", "performance_evidence": False}


def _dependencies():
    try:
        import onnx
        import numpy as np
    except ImportError as exc:
        raise OnnxImportError(
            "ONNX import requires optional dependencies: install onnx==1.19.0 numpy==2.2.6 "
            "in an application environment; the core package does not require them"
        ) from exc
    return onnx, np


def load_model(path, *, input_shapes=None):
    """Load a self-contained ONNX file into the shared typed graph.

    ``input_shapes`` maps input names to complete positive concrete dimensions;
    it may bind unknown dimensions but cannot change a declared static dimension.
    Inputs are fed to the resulting graph as flat row-major values. ``report``
    records input views, constant expansion, source hashes and output aliases.
    Unsupported semantics raise OnnxImportError with the source node's identity.
    """
    onnx, np = _dependencies()
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_MODEL_BYTES:
        raise OnnxImportError("ONNX file must exist and be at most 32 MiB")
    source = path.read_bytes()
    try:
        model = onnx.load_model_from_string(source)
    except Exception as exc:
        raise OnnxImportError(f"cannot parse ONNX model: {exc}") from exc
    if model.functions or model.training_info or model.graph.sparse_initializer:
        raise OnnxImportError("local functions, training graphs and sparse initializers are unsupported")
    if len(model.graph.node) > 128:
        raise OnnxImportError("ONNX application exceeds the 128-source-node limit")
    versions = {item.domain: item.version for item in model.opset_import}
    if len(versions) != len(model.opset_import) or set(versions) != {""} or not 13 <= versions[""] <= 22:
        raise OnnxImportError("exactly one standard ONNX opset in [13, 22] is supported")
    for initializer in model.graph.initializer:
        if initializer.external_data or initializer.data_location == onnx.TensorProto.EXTERNAL:
            raise OnnxImportError(f"initializer {initializer.name!r}: external data is unsupported; embed tensor data")
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        raise OnnxImportError(f"ONNX checker rejected model: {exc}") from exc
    dtype_names = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.FLOAT16: "float16",
                   onnx.TensorProto.INT8: "int8", onnx.TensorProto.INT32: "int32"}
    bindings = {} if input_shapes is None else input_shapes
    if type(bindings) is not dict:
        raise OnnxImportError("input_shapes must be a mapping of input names to dimension lists")
    input_names = tuple(value.name for value in model.graph.input)
    if set(bindings) - set(input_names):
        raise OnnxImportError("input_shapes contains unknown input names")
    constants, specs, symbols = {}, {}, {}
    input_schema, aliases, ir_shapes = {}, {}, {}
    tensors, nodes, lowerings = {}, [], []
    used_node_names = set()
    initializer_names = {x.name for x in model.graph.initializer}
    source_names = set(input_names) | initializer_names
    source_names |= {name for node in model.graph.node for name in node.output}
    generated_names = set(source_names)

    def dtype_name(code, label):
        if code not in dtype_names:
            raise OnnxImportError(f"{label}: unsupported ONNX dtype {code}; expected float16/float32/int8/int32")
        return dtype_names[code]

    def checked_shape(shape, label, *, scalar=False):
        if (not shape and not scalar) or len(shape) > 8 or any(type(x) is not int or x <= 0 for x in shape):
            raise OnnxImportError(f"{label}: only positive static dimensions (rank <= 8) are supported")
        if prod(shape) > MAX_ELEMENTS:
            raise OnnxImportError(f"{label}: tensor exceeds {MAX_ELEMENTS} elements")
        return tuple(shape)

    def fresh(base):
        index, name = 0, base
        while name in generated_names:
            index += 1
            name = f"{base}_{index}"
        generated_names.add(name)
        return name

    for value in model.graph.input:
        if not value.type.HasField("tensor_type"):
            raise OnnxImportError(f"input {value.name!r}: only tensors are supported")
        desc = value.type.tensor_type
        if not desc.HasField("shape"):
            raise OnnxImportError(f"input {value.name!r}: rank must be declared")
        provided = bindings.get(value.name)
        if provided is not None and (type(provided) not in (tuple, list) or len(provided) != len(desc.shape.dim)):
            raise OnnxImportError(f"input {value.name!r}: dimension binding must match declared rank")
        shape = []
        for index, dimension in enumerate(desc.shape.dim):
            size = provided[index] if provided is not None else (dimension.dim_value if dimension.HasField("dim_value") else None)
            if type(size) is not int or size <= 0:
                raise OnnxImportError(f"input {value.name!r}, dimension {index}: bind a positive static dimension in input_shapes")
            if dimension.HasField("dim_value") and size != dimension.dim_value:
                raise OnnxImportError(f"input {value.name!r}: binding changes a static dimension")
            if dimension.dim_param:
                if dimension.dim_param in symbols and symbols[dimension.dim_param] != size:
                    raise OnnxImportError(f"symbol {dimension.dim_param!r} has conflicting input dimension bindings")
                symbols[dimension.dim_param] = size
            shape.append(size)
        shape = checked_shape(shape, f"input {value.name!r}")
        dtype = dtype_name(desc.elem_type, f"input {value.name!r}")
        specs[value.name] = (shape, dtype)
        input_schema[value.name] = {"shape": list(shape), "dtype": dtype, "layout": "row_major", "ir_shape": list(shape)}

    for initializer in model.graph.initializer:
        name = initializer.name
        if name in input_names:
            raise OnnxImportError(f"initializer {name!r} is overridable via a graph input; freeze it or make it a runtime input")
        shape = checked_shape(list(initializer.dims), f"initializer {name!r}", scalar=True)
        dtype = dtype_name(initializer.data_type, f"initializer {name!r}")
        try:
            data = onnx.numpy_helper.to_array(initializer)
        except Exception as exc:
            raise OnnxImportError(f"initializer {name!r}: invalid tensor bytes: {exc}") from exc
        if data.shape != shape or not np.isfinite(data).all():
            raise OnnxImportError(f"initializer {name!r}: invalid shape or nonfinite value")
        constants[name], specs[name] = data, (shape, dtype)

    def canonical(name):
        while name in aliases:
            name = aliases[name]
        return name

    def tensor(name):
        name = canonical(name)
        if name in tensors:
            return name
        shape, dtype = specs[name]
        shape = ir_shapes.get(name, shape)
        if len(shape) not in (1, 2):
            raise OnnxImportError(f"tensor {name!r}: runtime IR requires rank 1 or 2; flatten inputs explicitly")
        values = None if name not in constants else tuple(constants[name].reshape(-1).tolist())
        tensors[name] = Tensor(name, shape, dtype, values)
        return name

    def add_node(name, op, inputs, output, shape, dtype, *, overflow="error"):
        checked_shape(shape, name)
        if name in used_node_names:
            name = fresh(f"{name}_node")
        used_node_names.add(name)
        specs[output] = (tuple(shape), dtype)
        arguments = tuple(tensor(x) for x in inputs)
        tensor(output)
        nodes.append(Node(name, op, arguments, output,
                          Arithmetic("float32" if dtype.startswith("float") else "int32", overflow=overflow)))

    def constant_copy(source_name, output, shape=None, transpose=False):
        source_name = canonical(source_name)
        array = constants[source_name]
        if transpose:
            array = array.T.copy()
        if shape is not None:
            array = np.broadcast_to(array, shape).copy()
        constants[output] = array
        specs[output] = (tuple(array.shape), specs[source_name][1])
        return output

    def same_dtype(names):
        types = {specs[name][1] for name in names}
        if len(types) != 1:
            raise OnnxImportError("mixed input dtypes require explicit ONNX Cast nodes")
        return types.pop()

    consumers = {}
    for index, source_node in enumerate(model.graph.node):
        for name in source_node.input:
            consumers.setdefault(name, []).append(index)
    output_names = {value.name for value in model.graph.output}

    for index, source_node in enumerate(model.graph.node):
        label = source_node.name or f"node_{index}_{source_node.op_type}"
        op = source_node.op_type
        try:
            if source_node.domain or op not in SUPPORTED_OPS:
                raise OnnxImportError(f"unsupported operator {source_node.domain or 'ai.onnx'}::{op}")
            if len(source_node.output) != 1 or not source_node.output[0]:
                raise OnnxImportError("exactly one named output is required")
            args = list(source_node.input)
            if any(name not in specs for name in args if name):
                raise OnnxImportError("node input is not yet defined")
            attrs = {item.name: onnx.helper.get_attribute_value(item) for item in source_node.attribute}
            allowed = {"Gemm": {"alpha", "beta", "transA", "transB"}, "Cast": {"to", "saturate"},
                       "Flatten": {"axis"}}.get(op, set())
            if set(attrs) - allowed:
                raise OnnxImportError(f"unsupported attributes: {sorted(set(attrs) - allowed)}")
            output = source_node.output[0]
            before = len(nodes)
            transforms = []
            if op in ("MatMul", "Gemm"):
                if len(args) not in ((2, 3) if op == "Gemm" else (2,)) or not all(args[:2]):
                    raise OnnxImportError("invalid linear operator input arity")
                dtype = same_dtype(args[:2])
                if dtype not in ("float16", "float32"):
                    raise OnnxImportError("integer ONNX MatMul/Gemm output and overflow policies do not match the current linear IR")
                a, b = args[:2]
                ashape, bshape = specs[a][0], specs[b][0]
                if op == "Gemm":
                    if attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0:
                        raise OnnxImportError("Gemm currently requires alpha=beta=1; scaling cannot be dropped")
                    if attrs.get("transA", 0) != 0 or attrs.get("transB", 0) not in (0, 1):
                        raise OnnxImportError("Gemm requires transA=0 and transB in {0,1}")
                    if attrs.get("transB", 0):
                        if canonical(b) not in constants:
                            raise OnnxImportError("Gemm transB=1 requires an initializer weight")
                        b = constant_copy(b, fresh(f"{b}_transposed"), transpose=True)
                        bshape = specs[b][0]
                        transforms.append("transpose_constant_weight")
                    if len(ashape) != 2:
                        raise OnnxImportError("Gemm activation must have rank 2")
                if len(ashape) not in (1, 2) or len(bshape) != 2 or ashape[-1] != bshape[0]:
                    raise OnnxImportError("linear lowering requires input[K] or input[M,K] and weights[K,N]")
                if canonical(b) not in constants:
                    raise OnnxImportError("application linear weights must be frozen initializers")
                outshape = (*ashape[:-1], bshape[1])
                bias = args[2] if len(args) == 3 else ""
                product_name = fresh(f"{output}_product") if bias else output
                add_node(label + ("_matmul" if bias else ""), "matvec" if len(ashape) == 1 else "matmul",
                         (a, b), product_name, outshape, dtype)
                if bias:
                    if specs[bias][1] != dtype:
                        raise OnnxImportError("Gemm bias dtype differs from the matrix dtype")
                    if specs[bias][0] != outshape:
                        if canonical(bias) not in constants:
                            raise OnnxImportError("Gemm broadcast bias must be an initializer")
                        bias = constant_copy(bias, fresh(f"{bias}_expanded"), shape=outshape)
                        transforms.append("materialize_broadcast_constant")
                    add_node(label + "_bias", "add", (product_name, bias), output, outshape, dtype)
            elif op == "Add":
                if len(args) != 2 or not all(args):
                    raise OnnxImportError("Add requires two inputs")
                dtype = same_dtype(args)
                if dtype == "int8":
                    raise OnnxImportError("ONNX int8 Add wrapping cannot use saturating IR requantization")
                shape = np.broadcast_shapes(*(specs[name][0] for name in args))
                checked_shape(shape, label)
                for i, name in enumerate(args):
                    if specs[name][0] != shape:
                        if canonical(name) not in constants:
                            raise OnnxImportError("broadcasting a runtime tensor is unsupported; only initializer expansion is available")
                        args[i] = constant_copy(name, fresh(f"{name}_expanded"), shape=shape)
                        transforms.append("materialize_broadcast_constant")
                add_node(label, "add", args, output, shape, dtype, overflow="wrap" if dtype == "int32" else "error")
            elif op in ("Relu", "Cast", "Identity", "Flatten"):
                if len(args) != 1 or not args[0]:
                    raise OnnxImportError(f"{op} requires one input")
                name = args[0]
                shape, dtype = specs[name]
                if op == "Relu":
                    add_node(label, "relu", args, output, shape, dtype)
                elif op == "Cast":
                    target = dtype_name(attrs.get("to"), "Cast output")
                    if attrs.get("saturate", 1) != 1:
                        raise OnnxImportError("non-default Cast saturate attribute is unsupported")
                    if target.startswith("int") and target != dtype and (dtype.startswith("float") or target == "int8"):
                        raise OnnxImportError("this Cast needs truncation/wrapping; typed Cast uses rounding/requantization, so conversion is rejected")
                    add_node(label, "cast", args, output, shape, target)
                elif op == "Identity":
                    aliases[output], specs[output] = canonical(name), specs[name]
                    transforms.append("identity_view")
                else:
                    axis = attrs.get("axis", 1)
                    if type(axis) is not int or not -len(shape) <= axis <= len(shape):
                        raise OnnxImportError("Flatten axis is outside the input rank")
                    axis = axis if axis >= 0 else axis + len(shape)
                    flattened = (prod(shape[:axis]), prod(shape[axis:]))
                    base = canonical(name)
                    if base in constants:
                        constants[output] = constants[base].reshape(flattened)
                        specs[output] = (flattened, dtype)
                        transforms.append("constant_reshape_view")
                    elif flattened == shape:
                        aliases[output], specs[output] = base, (flattened, dtype)
                        transforms.append("identity_view")
                    elif name in input_names and consumers.get(name) == [index] and name not in output_names:
                        aliases[output], specs[output], ir_shapes[name] = name, (flattened, dtype), flattened
                        input_schema[name]["ir_shape"] = list(flattened)
                        transforms.append("boundary_row_major_reshape_view")
                    else:
                        raise OnnxImportError("nonidentity Flatten requires a constant or a directly consumed, unshared graph input")
            lowerings.append({"source_index": index, "source_name": source_node.name, "source_op": op,
                              "ir_nodes": [node.name for node in nodes[before:]], "transformations": transforms})
        except (OnnxImportError, ValueError, TypeError, KeyError, OverflowError) as exc:
            raise OnnxImportError(f"node {index} {label!r} ({op}): {exc}") from exc

    # Preserve every declared input, even if unused by a supported source graph.
    for name in input_names:
        tensor(name)
    output_mapping = {}
    for value in model.graph.output:
        if value.name not in specs:
            raise OnnxImportError(f"output {value.name!r} was not produced")
        shape, dtype = specs[value.name]
        desc = value.type.tensor_type
        if dtype_name(desc.elem_type, f"output {value.name!r}") != dtype:
            raise OnnxImportError(f"output {value.name!r}: declared dtype differs from inferred dtype")
        if desc.HasField("shape"):
            if len(desc.shape.dim) != len(shape):
                raise OnnxImportError(f"output {value.name!r}: declared rank differs from inferred rank")
            for dimension, size in zip(desc.shape.dim, shape):
                if dimension.HasField("dim_value") and dimension.dim_value != size:
                    raise OnnxImportError(f"output {value.name!r}: declared shape differs from inferred shape")
                if dimension.dim_param in symbols and symbols[dimension.dim_param] != size:
                    raise OnnxImportError(f"output {value.name!r}: symbolic shape differs from input binding")
        output_mapping[value.name] = {"tensor": tensor(value.name), "shape": list(shape), "dtype": dtype}
    if not nodes:
        raise OnnxImportError("view-only graphs do not contain an executable operator")
    graph = Graph(tuple(tensors.values()), tuple(nodes), input_names,
                  tuple(dict.fromkeys(item["tensor"] for item in output_mapping.values())))
    try:
        graph.validate()
    except ValueError as exc:
        raise OnnxImportError(f"imported graph does not meet the typed IR contract: {exc}") from exc
    report = {
        "schema": "pim-foundry.onnx-import.v1", "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_file": path.name, "onnx_version": onnx.__version__, "onnx_opset": versions[""],
        "onnx_ir_version": model.ir_version, "graph_sha256": graph_digest(graph),
        "source_nodes": len(model.graph.node), "ir_nodes": len(nodes), "lowerings": lowerings,
        "input_schema": input_schema, "output_mapping": output_mapping,
        "initializer_bytes": sum(value.nbytes for name, value in constants.items() if name in initializer_names),
        "ir_constant_bytes": sum(value.nbytes for value in graph.tensors if value.data is not None),
        "precision_conversions": [item for item in lowerings if item["source_op"] == "Cast"],
        "numeric_policy": {
            "storage": "preserve_declared_onnx_dtype", "floating_accumulator": "float32",
            "floating_mac": "separate_multiply_add_in_increasing_k_order",
            "nonfinite": "reject", "implicit_precision_conversion": False,
            "onnx_equivalence": "requires_independent_reference_check_with_explicit_tolerances",
            "limitations": ["ONNX does not fix a universal floating reduction order or fused MAC behavior",
                            "Gemm bias lowering materializes a product tensor at the declared storage precision",
                            "finite bounded inputs only; no float-to-integer Cast or quantized operators"],
        },
        "evidence": {"model_imported": True, "independent_numerical_check": False,
                     "native_backend_executed": False, "performance_measured": False},
    }
    return ImportedModel(graph, input_schema, report, source)

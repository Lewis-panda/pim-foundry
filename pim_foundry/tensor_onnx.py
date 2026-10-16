"""Fail-closed ONNX frontend for tensor-graph v2 application execution.

The original ONNX file remains the independent semantic reference. Import does
not certify model quality or accelerator timing. External weights are deliberately
rejected until an application asset manifest can bind all of their byte ranges.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

from .tensor_ir import SCHEMA, validate_graph


class OnnxTensorImportError(ValueError):
    pass


def _dependencies():
    try:
        import onnx
        import numpy as np
    except ImportError as exc:
        raise OnnxTensorImportError("ONNX tensor import requires pim-foundry[onnx]") from exc
    return onnx, np


def load_model(path, input_shapes=None):
    """Load a self-contained ONNX model as a tensor-graph v2 mapping.

    Optional input_shapes binds dynamic dimensions without changing static ones.
    Runtime input names are exposed in metadata; prepare_inputs handles legal
    ONNX names that need renaming and checked int64-index to int32 conversion.
    """
    onnx, np = _dependencies()
    path = Path(path)
    if not path.is_file():
        raise OnnxTensorImportError("ONNX model file does not exist")
    source = path.read_bytes()
    try:
        model = onnx.load_model_from_string(source)
    except Exception as exc:
        raise OnnxTensorImportError(f"cannot parse ONNX model: {exc}") from exc
    if model.functions or model.training_info or model.graph.sparse_initializer:
        raise OnnxTensorImportError("local functions, training graphs, and sparse initializers are unsupported")
    versions = {item.domain: item.version for item in model.opset_import}
    if len(versions) != len(model.opset_import) or set(versions) != {""} or not 13 <= versions[""] <= 23:
        raise OnnxTensorImportError("one standard ONNX opset in [13, 23] is required")
    if any(node.domain not in ("", "ai.onnx") for node in model.graph.node):
        raise OnnxTensorImportError("custom-domain operations are unsupported")
    for initializer in model.graph.initializer:
        if initializer.external_data or initializer.data_location == onnx.TensorProto.EXTERNAL:
            raise OnnxTensorImportError(f"external initializer {initializer.name!r} is not a bound application asset")
    bindings = {} if input_shapes is None else input_shapes
    if not isinstance(bindings, dict) or bindings.keys() - {v.name for v in model.graph.input}:
        raise OnnxTensorImportError("input_shapes must map known model inputs to complete shapes")
    bound_symbols = {}
    for value in model.graph.input:
        if not value.type.HasField("tensor_type") or not value.type.tensor_type.HasField("shape"):
            raise OnnxTensorImportError(f"{value.name}: only tensor inputs with declared rank are supported")
        if value.name not in bindings:
            continue
        given, dims = bindings[value.name], value.type.tensor_type.shape.dim
        if not isinstance(given, (list, tuple)) or len(given) != len(dims) or any(type(d) is not int or d < 0 for d in given):
            raise OnnxTensorImportError(f"{value.name}: shape binding must match rank and contain nonnegative integers")
        for dim, size in zip(dims, given):
            if dim.HasField("dim_value") and dim.dim_value != size:
                raise OnnxTensorImportError(f"{value.name}: shape binding changes a static dimension")
            if dim.dim_param:
                if dim.dim_param in bound_symbols and bound_symbols[dim.dim_param] != size:
                    raise OnnxTensorImportError("conflicting shared symbolic dimension bindings")
                bound_symbols[dim.dim_param] = size
            dim.ClearField("dim_param")
            dim.dim_value = size
    # Bind the same symbolic dimension throughout the graph before inference.
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.type.HasField("tensor_type"):
            for dim in value.type.tensor_type.shape.dim:
                if dim.dim_param in bound_symbols:
                    size = bound_symbols[dim.dim_param]
                    dim.ClearField("dim_param")
                    dim.dim_value = size
    try:
        onnx.checker.check_model(model)
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:
        raise OnnxTensorImportError(f"ONNX validation or shape inference failed: {exc}") from exc
    dtype_codes = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.FLOAT16: "float16",
                   onnx.TensorProto.INT8: "int8", onnx.TensorProto.INT32: "int32",
                   onnx.TensorProto.INT64: "int64"}
    specs, constants, values, nodes, initializers, aliases = {}, {}, {}, [], {}, {}
    used_names, symbol_names = set(), {}
    report = {"source_sha256": hashlib.sha256(source).hexdigest(), "opset": versions[""],
              "onnx_version": onnx.__version__, "input_names": {}, "output_names": {},
              "input_dtypes": {}, "narrowed_int64_values": [], "lowerings": [],
              "integer_policy": "int64 permitted only as checked int32 indices or compile-time shape/control data",
              "performance_evidence": False}

    def identifier(name):
        if name in aliases:
            return aliases[name]
        base = name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:/-]{0,255}", name or "") else "value"
        candidate, suffix = base, 0
        while candidate in used_names:
            suffix += 1
            candidate = f"{base}_{suffix}"
        aliases[name] = candidate
        used_names.add(candidate)
        return candidate

    def fresh(label):
        return identifier(f"__lowered_{label}_{len(values)}_{len(nodes)}")

    def symbol(name):
        if name not in symbol_names:
            symbol_names[name] = f"dim_{len(symbol_names)}"
        return symbol_names[name]

    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if not value.type.HasField("tensor_type") or not value.type.tensor_type.HasField("shape"):
            raise OnnxTensorImportError(f"{value.name}: tensor type and rank must be inferable")
        tensor = value.type.tensor_type
        if tensor.elem_type not in dtype_codes:
            raise OnnxTensorImportError(f"{value.name}: unsupported tensor dtype {tensor.elem_type}")
        dims = []
        for index, dim in enumerate(tensor.shape.dim):
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            else:
                dims.append(symbol(dim.dim_param or f"{value.name}_axis_{index}"))
        specs[value.name] = {"shape": dims, "dtype": dtype_codes[tensor.elem_type]}
    for tensor in model.graph.initializer:
        array = onnx.numpy_helper.to_array(tensor)
        if array.dtype.name not in set(dtype_codes.values()) or not np.all(np.isfinite(array)):
            raise OnnxTensorImportError(f"{tensor.name}: unsupported or nonfinite initializer")
        constants[tensor.name] = array
        specs[tensor.name] = {"shape": list(array.shape), "dtype": array.dtype.name}
    initializer_names = set(constants)
    if initializer_names & {v.name for v in model.graph.input}:
        raise OnnxTensorImportError("overridable initializer inputs are unsupported; freeze them explicitly")

    def spec(name):
        if name not in specs:
            raise OnnxTensorImportError(f"{name}: output shape/type cannot be inferred; bind input_shapes or export value_info")
        return specs[name]

    def add_value(name, *, index=False):
        target = identifier(name)
        if target in values:
            return target
        declared = dict(spec(name))
        if declared["dtype"] == "int64":
            if not index:
                raise OnnxTensorImportError(f"{name}: int64 data arithmetic is outside the int32 runtime contract")
            declared["dtype"] = "int32"
            report["narrowed_int64_values"].append(name)
        values[target] = {"shape": list(declared["shape"]), "dtype": declared["dtype"]}
        if name in constants:
            array = constants[name]
            if declared["dtype"] == "int32" and array.dtype.name == "int64":
                if np.any(array < -(2 ** 31)) or np.any(array >= 2 ** 31):
                    raise OnnxTensorImportError(f"{name}: int64 index cannot be represented as int32")
                array = array.astype(np.int32)
            initializers[target] = array.copy()
        return target

    def constant(name, label, *, integer=False):
        if not name or name not in constants:
            raise OnnxTensorImportError(f"{label} requires compile-time constant input; bind input_shapes for shape expressions")
        array = constants[name]
        if integer and array.dtype.kind not in "iu":
            raise OnnxTensorImportError(f"{label} requires integer control data")
        return array

    def emit(op, ins, out, attrs=None, output_spec=None):
        if output_spec is not None:
            values[out] = {"shape": list(output_spec["shape"]), "dtype": output_spec["dtype"]}
        nodes.append({"name": f"node_{len(nodes)}", "op": op, "inputs": ins, "output": out, "attrs": attrs or {}})

    def only(attrs, allowed, label):
        if attrs.keys() - set(allowed):
            raise OnnxTensorImportError(f"{label}: unsupported attributes {sorted(attrs.keys() - set(allowed))}")

    input_names = []
    for source_input in model.graph.input:
        name = source_input.name
        dtype = spec(name)["dtype"]
        # Runtime int64 is accepted only when every use is an indexing operand.
        if dtype == "int64":
            uses = [(n.op_type, i) for n in model.graph.node for i, n_in in enumerate(n.input) if n_in == name]
            if not uses or any(op != "Gather" or index != 1 for op, index in uses):
                raise OnnxTensorImportError(f"{name}: int64 runtime inputs are supported only as Gather indices")
        target = add_value(name, index=dtype == "int64")
        input_names.append(target)
        report["input_names"][name], report["input_dtypes"][name] = target, dtype
    for ordinal, source_node in enumerate(model.graph.node):
        op, ins, outs = source_node.op_type, list(source_node.input), list(source_node.output)
        if op in ("Gemm", "Conv", "LayerNormalization"):
            while ins and not ins[-1]:
                ins.pop()
        label = source_node.name or f"{op}_{ordinal}"
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in source_node.attribute}
        if len(attrs) != len(source_node.attribute):
            raise OnnxTensorImportError(f"{label}: duplicate attributes")
        if len(outs) != 1 or not outs[0]:
            raise OnnxTensorImportError(f"{label}: one-output operators are required")
        output = outs[0]
        if op == "Constant":
            only(attrs, {"value", "value_int", "value_ints", "value_float", "value_floats"}, label)
            if len(attrs) != 1:
                raise OnnxTensorImportError(f"{label}: unsupported Constant encoding")
            key, value = next(iter(attrs.items()))
            if key == "value":
                if value.external_data or value.data_location == onnx.TensorProto.EXTERNAL:
                    raise OnnxTensorImportError("external Constant tensors are unsupported")
                array = onnx.numpy_helper.to_array(value)
            else:
                array = np.asarray(value, dtype=np.int64 if "int" in key else np.float32)
            if array.dtype.name not in set(dtype_codes.values()) or not np.all(np.isfinite(array)):
                raise OnnxTensorImportError(f"{label}: unsupported or nonfinite Constant")
            constants[output] = array
            specs[output] = {"shape": list(array.shape), "dtype": array.dtype.name}
            report["lowerings"].append({"source_op": op, "action": "constant"})
            continue
        if op == "Shape":
            only(attrs, {"start", "end"}, label)
            shape = spec(ins[0])["shape"]
            if any(type(d) is not int for d in shape):
                raise OnnxTensorImportError(f"{label}: Shape needs concrete input_shapes")
            constants[output] = np.asarray(shape, dtype=np.int64)[attrs.get("start", 0):attrs.get("end", len(shape))]
            specs[output] = {"shape": list(constants[output].shape), "dtype": "int64"}
            continue
        # Fold shape/control subgraphs only; floating nodes stay executable.
        if ins and all(name in constants and constants[name].dtype.kind in "iu" for name in ins) and op in ("Gather", "Concat", "Unsqueeze", "Squeeze", "Add", "Sub", "Mul", "Div", "Cast"):
            a = constants[ins[0]]
            if op == "Gather":
                only(attrs, {"axis"}, label)
                folded = np.take(a, constants[ins[1]], axis=attrs.get("axis", 0))
            elif op == "Concat":
                only(attrs, {"axis"}, label)
                folded = np.concatenate([constants[n] for n in ins], axis=attrs.get("axis", 0))
            elif op in ("Unsqueeze", "Squeeze"):
                only(attrs, set(), label)
                axes = tuple(int(x) for x in constant(ins[1], label, integer=True).reshape(-1)) if len(ins) > 1 else tuple(i for i, size in enumerate(a.shape) if size == 1)
                folded = np.expand_dims(a, axes) if op == "Unsqueeze" else np.squeeze(a, axes)
            elif op == "Cast":
                only(attrs, {"to", "saturate"}, label)
                if attrs.get("to") not in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
                    raise OnnxTensorImportError(f"{label}: integer shape Cast must preserve integer type")
                folded = a.astype(np.int64 if attrs["to"] == onnx.TensorProto.INT64 else np.int32)
                if not np.array_equal(folded.astype(np.int64), a.astype(np.int64)):
                    raise OnnxTensorImportError(f"{label}: shape Cast overflows")
            else:
                only(attrs, set(), label)
                b = constants[ins[1]]
                if op == "Div":
                    if np.any(b == 0):
                        raise OnnxTensorImportError(f"{label}: zero in shape division")
                    quotients = []
                    for x, y in zip(*[z.reshape(-1) for z in np.broadcast_arrays(a, b)]):
                        x, y = int(x), int(y)
                        quotient = (abs(x) // abs(y)) * (-1 if (x < 0) != (y < 0) else 1)
                        if not -(2 ** 63) <= quotient < 2 ** 63:
                            raise OnnxTensorImportError(f"{label}: shape integer division overflows")
                        quotients.append(quotient)
                    folded = np.asarray(quotients, dtype=np.int64).reshape(np.broadcast_shapes(a.shape, b.shape))
                else:
                    object_result = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply}[op](a.astype(object), b.astype(object))
                    if np.any(object_result < -(2 ** 63)) or np.any(object_result >= 2 ** 63):
                        raise OnnxTensorImportError(f"{label}: shape integer overflow")
                    folded = np.asarray(object_result, dtype=np.int64)
            constants[output] = folded
            inferred_dtype = specs.get(output, {}).get("dtype")
            if inferred_dtype == "int32":
                if np.any(folded < -(2 ** 31)) or np.any(folded >= 2 ** 31):
                    raise OnnxTensorImportError(f"{label}: shape int32 overflow")
                constants[output] = folded = folded.astype(np.int32)
            specs[output] = {"shape": list(folded.shape), "dtype": folded.dtype.name}
            continue
        if any(name and spec(name)["dtype"] == "int64" and not (op == "Gather" and i == 1) and not (op in ("Reshape", "Slice", "Squeeze", "Unsqueeze", "ReduceMean") and i > 0) for i, name in enumerate(ins)):
            raise OnnxTensorImportError(f"{label}: int64 arithmetic/data is unsupported")
        out = add_value(output, index=op == "ArgMax")
        supported_binary = {"MatMul": "matmul", "Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div"}
        if op in supported_binary:
            only(attrs, set(), label)
            emit(supported_binary[op], [add_value(n) for n in ins], out)
        elif op in ("Relu", "Identity", "Softmax", "Gelu"):
            allowed = {"Softmax": {"axis"}, "Gelu": {"approximate"}}.get(op, set())
            only(attrs, allowed, label)
            mapped = {"Relu": "relu", "Identity": "identity", "Softmax": "softmax", "Gelu": "gelu"}[op]
            lowered_attrs = dict(attrs)
            if "approximate" in lowered_attrs:
                lowered_attrs["approximate"] = lowered_attrs["approximate"].decode("ascii")
            emit(mapped, [add_value(ins[0])], out, lowered_attrs)
        elif op == "Gemm":
            only(attrs, {"alpha", "beta", "transA", "transB"}, label)
            if len(ins) not in (2, 3) or any(len(spec(n)["shape"]) != 2 for n in ins[:2]):
                raise OnnxTensorImportError(f"{label}: Gemm requires rank-two matrix operands")
            alpha, beta = float(attrs.get("alpha", 1)), float(attrs.get("beta", 1))
            if any(attrs.get(key, 0) not in (0, 1) for key in ("transA", "transB")):
                raise OnnxTensorImportError(f"{label}: Gemm transpose attributes must be zero or one")
            if not math.isfinite(alpha) or not math.isfinite(beta):
                raise OnnxTensorImportError(f"{label}: nonfinite Gemm coefficient")
            mm = out if alpha == 1 and len(ins) == 2 else fresh("gemm_matmul")
            emit("matmul", [add_value(n) for n in ins[:2]], mm,
                 {"transpose_a": bool(attrs.get("transA", 0)), "transpose_b": bool(attrs.get("transB", 0))}, values[out])
            current = mm
            def scaled(value, coefficient, target, target_spec):
                scalar = fresh("scale")
                values[scalar] = {"shape": [], "dtype": target_spec["dtype"]}
                initializers[scalar] = np.asarray(coefficient, dtype=target_spec["dtype"])
                emit("mul", [value, scalar], target, output_spec=target_spec)
            if alpha != 1:
                current = out if len(ins) == 2 else fresh("gemm_scaled")
                scaled(mm, alpha, current, values[out])
            if len(ins) == 3:
                bias = add_value(ins[2])
                if beta != 1:
                    scaled_bias = fresh("gemm_bias")
                    scaled(bias, beta, scaled_bias, values[bias])
                    bias = scaled_bias
                emit("add", [current, bias], out)
        elif op == "Transpose":
            only(attrs, {"perm"}, label)
            emit("transpose", [add_value(ins[0])], out, attrs)
        elif op in ("Reshape", "Flatten", "Squeeze", "Unsqueeze"):
            input_shape = spec(ins[0])["shape"]
            if op == "Reshape":
                only(attrs, {"allowzero"}, label)
                target = constant(ins[1], label, integer=True)
                if target.ndim != 1:
                    raise OnnxTensorImportError(f"{label}: reshape shape must be rank one")
                target = [int(d) for d in target]
                if not attrs.get("allowzero", 0):
                    if any(d == 0 and i >= len(input_shape) for i, d in enumerate(target)):
                        raise OnnxTensorImportError(f"{label}: reshape zero cannot copy an absent input dimension")
                    target = [input_shape[i] if d == 0 else d for i, d in enumerate(target)]
            elif op == "Flatten":
                only(attrs, {"axis"}, label)
                axis = int(attrs.get("axis", 1))
                if not -len(input_shape) <= axis <= len(input_shape):
                    raise OnnxTensorImportError(f"{label}: invalid Flatten axis")
                axis = axis if axis >= 0 else axis + len(input_shape)
                def product(dims):
                    if all(type(d) is int for d in dims):
                        return math.prod(dims)
                    if len(dims) == 1:
                        return dims[0]
                    raise OnnxTensorImportError(f"{label}: bind input_shapes for products of dynamic dimensions")
                target = [product(input_shape[:axis]), product(input_shape[axis:])]
            else:
                only(attrs, set(), label)
                if len(ins) < 2:
                    if any(type(d) is not int for d in input_shape):
                        raise OnnxTensorImportError(f"{label}: implicit squeeze needs concrete input shape")
                    axes = [i for i, d in enumerate(input_shape) if d == 1]
                else:
                    axes = [int(a) for a in constant(ins[1], label, integer=True).reshape(-1)]
                rank = len(input_shape) + len(axes) if op == "Unsqueeze" else len(input_shape)
                if any(not -rank <= a < rank for a in axes):
                    raise OnnxTensorImportError(f"{label}: invalid squeeze/unsqueeze axis")
                axes = [a % rank for a in axes]
                if len(set(axes)) != len(axes):
                    raise OnnxTensorImportError(f"{label}: repeated axis")
                if op == "Squeeze":
                    if any(input_shape[a] != 1 for a in axes):
                        raise OnnxTensorImportError(f"{label}: squeezed dimensions must be statically one")
                    target = [d for i, d in enumerate(input_shape) if i not in axes]
                else:
                    source_dims = iter(input_shape)
                    target = [1 if i in axes else next(source_dims) for i in range(rank)]
            emit("reshape", [add_value(ins[0])], out, {"shape": target})
        elif op == "Gather":
            only(attrs, {"axis"}, label)
            emit("gather", [add_value(ins[0]), add_value(ins[1], index=True)], out, attrs)
        elif op == "Concat":
            only(attrs, {"axis"}, label)
            emit("concat", [add_value(n) for n in ins], out, attrs)
        elif op == "Slice":
            only(attrs, set(), label)
            lowered = {key: [int(v) for v in constant(n, label, integer=True).reshape(-1)] for key, n in zip(("starts", "ends", "axes", "steps"), ins[1:]) if n}
            emit("slice", [add_value(ins[0])], out, lowered)
        elif op == "Cast":
            only(attrs, {"to", "saturate"}, label)
            # ONNX float-to-integer truncates, while the IR rounds to even.
            if spec(ins[0])["dtype"].startswith("float") and spec(output)["dtype"].startswith("int"):
                raise OnnxTensorImportError(f"{label}: ONNX float-to-integer Cast truncation differs from tensor v2 rounding")
            emit("cast", [add_value(ins[0])], out, {"dtype": values[out]["dtype"]})
        elif op == "LayerNormalization":
            only(attrs, {"axis", "epsilon", "stash_type"}, label)
            if attrs.get("stash_type", onnx.TensorProto.FLOAT) != onnx.TensorProto.FLOAT:
                raise OnnxTensorImportError(f"{label}: LayerNormalization requires float32 accumulation")
            emit("layer_norm", [add_value(n) for n in ins], out,
                 {"axis": attrs.get("axis", -1), "epsilon": attrs.get("epsilon", 1e-5)})
        elif op in ("ReduceMean", "GlobalAveragePool"):
            only(attrs, {"axes", "keepdims", "noop_with_empty_axes"} if op == "ReduceMean" else set(), label)
            axes = attrs.get("axes", list(range(len(spec(ins[0])["shape"]))))
            if len(ins) > 1 and ins[1]:
                axes = [int(v) for v in constant(ins[1], label, integer=True).reshape(-1)]
            if op == "GlobalAveragePool":
                axes = list(range(2, len(spec(ins[0])["shape"])))
            if not axes and not attrs.get("noop_with_empty_axes", 0):
                axes = list(range(len(spec(ins[0])["shape"])))
            emit("reduce_mean", [add_value(ins[0])], out, {"axes": axes, "keepdims": bool(attrs.get("keepdims", 1))})
        elif op == "ArgMax":
            only(attrs, {"axis", "keepdims", "select_last_index"}, label)
            if attrs.get("select_last_index", 0):
                raise OnnxTensorImportError(f"{label}: last-index tie selection unsupported")
            emit("argmax", [add_value(ins[0])], out, {"axis": attrs.get("axis", 0), "keepdims": bool(attrs.get("keepdims", 1))})
        elif op in ("Conv", "AveragePool", "MaxPool"):
            allowed = {"auto_pad", "pads", "strides", "kernel_shape", "dilations", "group"} if op == "Conv" else {"auto_pad", "pads", "strides", "kernel_shape", "ceil_mode", "count_include_pad", "storage_order", "dilations"}
            only(attrs, allowed, label)
            auto_pad = attrs.get("auto_pad", b"NOTSET")
            if auto_pad not in (b"NOTSET", b"VALID") or attrs.get("ceil_mode", 0) or attrs.get("storage_order", 0):
                raise OnnxTensorImportError(f"{label}: auto padding, ceil pooling, or alternate storage order unsupported")
            lowered = {"strides": attrs.get("strides", [1, 1]), "pads": [0, 0, 0, 0] if auto_pad == b"VALID" else attrs.get("pads", [0, 0, 0, 0])}
            if op == "Conv":
                lowered.update(groups=attrs.get("group", 1), dilations=attrs.get("dilations", [1, 1]))
                if "kernel_shape" in attrs and attrs["kernel_shape"] != spec(ins[1])["shape"][2:]:
                    raise OnnxTensorImportError(f"{label}: kernel_shape disagrees with weights")
                mapped = "conv2d"
            else:
                if attrs.get("dilations", [1, 1]) != [1, 1]:
                    raise OnnxTensorImportError(f"{label}: dilated pooling unsupported")
                lowered["kernel"] = attrs.get("kernel_shape")
                mapped = "avg_pool2d" if op == "AveragePool" else "max_pool2d"
                if op == "AveragePool":
                    lowered["count_include_pad"] = bool(attrs.get("count_include_pad", 0))
            emit(mapped, [add_value(n) for n in ins], out, lowered)
        else:
            raise OnnxTensorImportError(f"{label}: unsupported ONNX operator {op}")
        report["lowerings"].append({"source_op": op, "source_node": label, "output": out})
    output_names = []
    for source_output in model.graph.output:
        name = source_output.name
        target = add_value(name, index=spec(name)["dtype"] == "int64")
        if spec(name)["dtype"] == "int64" and target in initializers:
            raise OnnxTensorImportError(f"{name}: integer control outputs are outside this frontend's application-data contract")
        output_names.append(target)
        report["output_names"][name] = target
    # Keep only materialized constants; compile-time shape constants are not graph values.
    return validate_graph({"schema": SCHEMA, "values": values, "inputs": input_names,
                           "outputs": output_names, "initializers": initializers,
                           "nodes": nodes, "metadata": {"onnx": report}})


def prepare_inputs(graph, inputs):
    """Convert source-named feeds to checked, declared v2 storage arrays."""
    _, np = _dependencies()
    metadata = graph.get("metadata", {}).get("onnx")
    if metadata is None or not isinstance(inputs, dict) or set(inputs) != set(metadata["input_names"]):
        raise OnnxTensorImportError("feeds must match original ONNX input names exactly")
    prepared = {}
    for original, target in metadata["input_names"].items():
        raw = np.asarray(inputs[original])
        declared = graph["values"][target]["dtype"]
        source_dtype = metadata["input_dtypes"][original]
        if raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw)):
            raise OnnxTensorImportError(f"{original}: feed must contain finite numeric values")
        if isinstance(inputs[original], np.ndarray) and raw.dtype.name != source_dtype:
            raise OnnxTensorImportError(f"{original}: expected original ONNX dtype {source_dtype}, got {raw.dtype.name}")
        if declared.startswith("int"):
            info = np.iinfo(declared)
            if np.any(raw < info.min) or np.any(raw > info.max) or np.any(raw != np.trunc(raw)):
                raise OnnxTensorImportError(f"{original}: indices exceed exact {declared} range")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.array(raw, dtype=declared, copy=True)
        if not np.all(np.isfinite(array)):
            raise OnnxTensorImportError(f"{original}: input conversion overflows {declared}")
        prepared[target] = array
    return prepared


def compare_reference(path, inputs, *, input_shapes=None, atol=1e-5, rtol=1e-5, actual_outputs=None):
    """Compare original ONNX ReferenceEvaluator against v2/backend outputs.

    actual_outputs optionally supplies actual platform outputs in IR value names;
    when omitted, the common NumPy executor supplies them. Values and shape must
    match before tolerance comparison. Reports are about these inputs only.
    """
    onnx, np = _dependencies()
    from onnx.reference import ReferenceEvaluator
    from .tensor_executor import execute
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise OnnxTensorImportError("comparison tolerances must be finite and nonnegative")
    graph = load_model(path, input_shapes)
    feeds = prepare_inputs(graph, inputs)
    metadata = graph["metadata"]["onnx"]
    actual = execute(graph, feeds)["outputs"] if actual_outputs is None else actual_outputs
    if not isinstance(actual, dict) or set(actual) != set(graph["outputs"]):
        raise OnnxTensorImportError("actual outputs must exactly match imported graph outputs")
    original_feeds = {original: feeds[target].astype(metadata["input_dtypes"][original]) for original, target in metadata["input_names"].items()}
    reference_source = Path(path).read_bytes()
    if hashlib.sha256(reference_source).hexdigest() != metadata["source_sha256"]:
        raise OnnxTensorImportError("ONNX source changed between import and reference execution")
    evaluator = ReferenceEvaluator(onnx.load_model_from_string(reference_source))
    expected = evaluator.run(None, original_feeds)
    per_output = {}
    for original, reference in zip(evaluator.output_names, expected):
        target = metadata["output_names"][original]
        observed = np.asarray(actual[target])
        finite = np.all(np.isfinite(reference)) and np.all(np.isfinite(observed))
        shape_ok = reference.shape == observed.shape
        dtype_ok = observed.dtype.name == graph["values"][target]["dtype"]
        error = float(np.max(np.abs(reference.astype(np.float64) - observed.astype(np.float64)))) if finite and shape_ok and reference.size else (0.0 if finite and shape_ok else None)
        exact_integer = reference.dtype.kind in "iu"
        passed = bool(finite and shape_ok and dtype_ok and (np.array_equal(reference, observed) if exact_integer else np.allclose(observed, reference, atol=atol, rtol=rtol)))
        per_output[original] = {"passed": passed, "max_abs_error": error, "expected_shape": list(reference.shape),
                                "actual_shape": list(observed.shape), "reference_dtype": reference.dtype.name,
                                "actual_dtype": observed.dtype.name, "declared_dtype_matched": dtype_ok,
                                "comparison": "exact_integer" if exact_integer else "absolute_plus_relative_tolerance"}
    errors = [v["max_abs_error"] for v in per_output.values() if v["max_abs_error"] is not None]
    return {"passed": all(v["passed"] for v in per_output.values()), "max_abs_error": max(errors, default=None),
            "outputs": per_output, "atol": atol, "rtol": rtol, "reference": "onnx.reference.ReferenceEvaluator",
            "source_sha256": metadata["source_sha256"], "onnx_version": onnx.__version__, "numpy_version": np.__version__,
            "scope": "provided_model_and_input_values_only", "hardware_performance_validated": False}

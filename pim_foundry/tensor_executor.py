"""Finite, checked CPU reference execution for tensor-graph v2.

NumPy execution is a functional reference, never a hardware performance model.
Float16 storage uses float32 node arithmetic followed by one output conversion.
Matmul reduction order follows the installed NumPy/BLAS implementation; this is
not the sequential non-fused arithmetic contract of typed-graph v1.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from .tensor_ir import validate_graph, OPS


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("tensor-graph v2 requires NumPy; install pim-foundry[tensor]") from exc
    return np


def _axis(axis, rank, label="axis"):
    if type(axis) is not int or not -rank <= axis < rank:
        raise ValueError(f"{label} must address a tensor axis")
    return axis % rank


def _integer(value, label, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{label} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return value


def _boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _resolve(value, symbols, label):
    if isinstance(value, str):
        if value not in symbols:
            raise ValueError(f"{label}: unbound shape symbol {value}")
        return symbols[value]
    return _integer(value, label)


def _bind_shape(shape, spec, symbols, contract, label):
    if len(shape) != len(spec["shape"]):
        raise ValueError(f"{label}: rank {len(shape)} does not match declared rank {len(spec['shape'])}")
    for actual, declared in zip(shape, spec["shape"]):
        actual = int(actual)
        if isinstance(declared, str):
            if declared in symbols and symbols[declared] != actual:
                raise ValueError(f"{label}: dimension symbol {declared} expected {symbols[declared]}, got {actual}")
            bounds = contract.get(declared, {})
            if actual < bounds.get("min", 0) or ("max" in bounds and actual > bounds["max"]):
                raise ValueError(f"{label}: dimension {declared} violates shape_contract")
            symbols[declared] = actual
        elif declared != actual:
            raise ValueError(f"{label}: expected shape {spec['shape']}, got {list(shape)}")


def _array(value, dtype, label, *, strict_array=True):
    np = _numpy()
    if isinstance(value, np.ndarray) and strict_array and value.dtype != np.dtype(dtype):
        raise ValueError(f"{label}: ndarray dtype {value.dtype} differs from declared {dtype}; use an explicit cast")
    source = np.asarray(value)
    if source.dtype.kind not in "iuf" or source.dtype.kind == "b":
        raise ValueError(f"{label}: tensor values must be real numeric scalars")
    if not np.all(np.isfinite(source)):
        raise ValueError(f"{label}: nonfinite values are forbidden")
    if dtype in ("int8", "int32"):
        bounds = np.iinfo(dtype)
        if np.any(source < bounds.min) or np.any(source > bounds.max) or np.any(source != np.trunc(source)):
            raise ValueError(f"{label}: values are not representable as {dtype}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(source, dtype=dtype, copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label}: {dtype} overflow")
    result.flags.writeable = False
    return result


def _same_dtype(arrays, label):
    if len({a.dtype.str for a in arrays}) != 1:
        raise ValueError(f"{label}: mixed input dtypes require explicit cast nodes")


def _floating(arrays, label):
    _same_dtype(arrays, label)
    if any(a.dtype.name not in ("float16", "float32") for a in arrays):
        raise ValueError(f"{label}: operation requires floating inputs")


def _indices(array, label):
    if array.dtype.name != "int32":
        raise ValueError(f"{label}: indices must have dtype int32")


def _pair(value, label, minimum=1):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain two integers")
    return tuple(_integer(x, label, minimum) for x in value)


def _padding(value):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("pads must be [top, left, bottom, right]")
    return tuple(_integer(x, "padding", 0) for x in value)


def _plan(node, arrays, output_spec, symbols):
    """Check semantics and infer output shape before allocation or callbacks."""
    np = _numpy()
    op, attrs, name = node["op"], node["attrs"], node["name"]
    x = arrays[0]
    params = {}
    shape = x.shape
    work = max(1, x.size)
    if op == "matmul":
        _same_dtype(arrays, name)
        a, b = arrays
        ta = _boolean(attrs.get("transpose_a", False), "transpose_a")
        tb = _boolean(attrs.get("transpose_b", False), "transpose_b")
        if a.ndim == 0 or b.ndim == 0 or (ta and a.ndim < 2) or (tb and b.ndim < 2):
            raise ValueError(f"{name}: matmul requires vectors or matrices")
        ashape = a.shape[:-2] + (a.shape[-1], a.shape[-2]) if ta else a.shape
        bshape = b.shape[:-2] + (b.shape[-1], b.shape[-2]) if tb else b.shape
        av, bv = len(ashape) == 1, len(bshape) == 1
        aa, bb = ((1,) + ashape if av else ashape), (bshape + (1,) if bv else bshape)
        if aa[-1] != bb[-2]:
            raise ValueError(f"{name}: matmul contracting dimensions differ")
        batch = np.broadcast_shapes(aa[:-2], bb[:-2])
        shape = batch + (() if av else (aa[-2],)) + (() if bv else (bb[-1],))
        params = {"transpose_a": ta, "transpose_b": tb}
        work = 2 * math.prod(shape) * aa[-1]
        if a.dtype.kind == "i" and output_spec["dtype"] != "int32":
            raise ValueError(f"{name}: integer matmul output must be int32")
    elif op in ("add", "sub", "mul", "div"):
        _same_dtype(arrays, name)
        if op == "div":
            _floating(arrays, name)
            if np.any(arrays[1] == 0):
                raise ValueError(f"{name}: division by zero")
        shape = np.broadcast_shapes(*(a.shape for a in arrays))
        work = math.prod(shape)
    elif op == "reshape":
        target = attrs.get("shape")
        if not isinstance(target, list) or len(target) > 16:
            raise ValueError(f"{name}: reshape requires a shape list")
        shape = tuple(_resolve(d, symbols, "reshape") for d in target)
        if any(d < -1 for d in shape) or shape.count(-1) > 1:
            raise ValueError(f"{name}: invalid inferred reshape dimension")
        if -1 in shape:
            known = math.prod(d for d in shape if d != -1)
            if known == 0 or x.size % known:
                raise ValueError(f"{name}: ambiguous or incompatible inferred reshape")
            shape = tuple(x.size // known if d == -1 else d for d in shape)
        if math.prod(shape) != x.size:
            raise ValueError(f"{name}: reshape changes the element count")
        params["shape"] = shape
    elif op == "transpose":
        perm = attrs.get("perm", list(reversed(range(x.ndim))))
        if not isinstance(perm, list) or any(type(a) is not int for a in perm) or sorted(perm) != list(range(x.ndim)):
            raise ValueError(f"{name}: perm must be a permutation of all axes")
        shape, params = tuple(x.shape[a] for a in perm), {"perm": perm}
    elif op in ("gather", "embedding"):
        index = arrays[1]
        _indices(index, name)
        axis = 0 if op == "embedding" else _axis(attrs.get("axis", 0), x.ndim)
        if op == "embedding" and x.ndim != 2:
            raise ValueError(f"{name}: embedding weights must have rank two")
        if np.any(index >= x.shape[axis]) or np.any(index < (0 if op == "embedding" else -x.shape[axis])):
            raise ValueError(f"{name}: index is out of bounds")
        shape = x.shape[:axis] + index.shape + x.shape[axis + 1:]
        params["axis"] = axis
        work = math.prod(shape)
    elif op == "slice":
        starts, ends = attrs.get("starts"), attrs.get("ends")
        if not isinstance(starts, list) or not isinstance(ends, list) or len(starts) != len(ends):
            raise ValueError(f"{name}: slice requires equal-length starts and ends")
        axes, steps = attrs.get("axes", list(range(len(starts)))), attrs.get("steps", [1] * len(starts))
        if not isinstance(axes, list) or not isinstance(steps, list) or len(axes) != len(starts) or len(steps) != len(starts):
            raise ValueError(f"{name}: slice axes and steps must match starts")
        axes = [_axis(a, x.ndim) for a in axes]
        if len(set(axes)) != len(axes):
            raise ValueError(f"{name}: repeated slice axis")
        slices = [slice(None)] * x.ndim
        for start, end, axis, step in zip(starts, ends, axes, steps):
            start, end, step = _resolve(start, symbols, "slice start"), _resolve(end, symbols, "slice end"), _integer(step, "slice step")
            if step == 0:
                raise ValueError(f"{name}: slice step cannot be zero")
            slices[axis] = slice(start, end, step)
        shape = tuple(len(range(*s.indices(d))) for s, d in zip(slices, x.shape))
        params["slices"] = tuple(slices)
    elif op == "concat":
        _same_dtype(arrays, name)
        axis = _axis(attrs.get("axis", 0), x.ndim)
        if any(a.ndim != x.ndim or any(a.shape[i] != x.shape[i] for i in range(x.ndim) if i != axis) for a in arrays):
            raise ValueError(f"{name}: concat non-axis dimensions differ")
        shape = tuple(sum(a.shape[i] for a in arrays) if i == axis else x.shape[i] for i in range(x.ndim))
        params["axis"] = axis
        work = math.prod(shape)
    elif op in ("softmax", "layer_norm", "rms_norm", "reduce_mean", "gelu", "silu"):
        _floating(arrays, name)
        if op == "softmax":
            axis = _axis(attrs.get("axis", -1), x.ndim)
            if x.shape[axis] == 0:
                raise ValueError(f"{name}: softmax over an empty axis")
            params["axis"] = axis
        elif op in ("layer_norm", "rms_norm"):
            axis = _axis(attrs.get("axis", -1), x.ndim)
            epsilon = attrs.get("epsilon", 1e-5 if op == "layer_norm" else 1e-6)
            if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
                raise ValueError(f"{name}: epsilon must be finite and positive")
            if any(a.shape != x.shape[axis:] for a in arrays[1:]) or math.prod(x.shape[axis:]) == 0:
                raise ValueError(f"{name}: normalization parameter shape mismatch or empty normalized dimension")
            params = {"axes": tuple(range(axis, x.ndim)), "epsilon": float(epsilon)}
        elif op == "reduce_mean":
            axes = attrs.get("axes", list(range(x.ndim)))
            if not isinstance(axes, list):
                raise ValueError(f"{name}: axes must be a list")
            axes = tuple(_axis(a, x.ndim) for a in axes)
            if len(set(axes)) != len(axes):
                raise ValueError(f"{name}: duplicate reduction axis")
            keep = _boolean(attrs.get("keepdims", True), "keepdims")
            if any(x.shape[a] == 0 for a in axes):
                raise ValueError(f"{name}: mean over an empty dimension")
            shape = tuple(1 if i in axes else d for i, d in enumerate(x.shape)) if keep else tuple(d for i, d in enumerate(x.shape) if i not in axes)
            params = {"axes": axes, "keepdims": keep}
        elif op == "gelu":
            approximate = attrs.get("approximate", "none")
            if approximate not in ("none", "tanh"):
                raise ValueError(f"{name}: gelu approximate must be none or tanh")
            params["approximate"] = approximate
        work *= 8
    elif op in ("relu", "identity"):
        pass
    elif op == "argmax":
        axis = _axis(attrs.get("axis", -1), x.ndim)
        keep = _boolean(attrs.get("keepdims", False), "keepdims")
        if not x.shape[axis]:
            raise ValueError(f"{name}: argmax over an empty dimension")
        if output_spec["dtype"] != "int32":
            raise ValueError(f"{name}: argmax output must be int32")
        shape = tuple(1 if i == axis else d for i, d in enumerate(x.shape)) if keep else tuple(d for i, d in enumerate(x.shape) if i != axis)
        params = {"axis": axis, "keepdims": keep}
    elif op == "cast":
        if attrs.get("dtype", output_spec["dtype"]) != output_spec["dtype"]:
            raise ValueError(f"{name}: cast dtype differs from output declaration")
    elif op == "causal_attention":
        _floating(arrays, name)
        q, k, v = arrays
        if any(a.ndim != 4 for a in arrays) or q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3] or q.shape[3] != k.shape[3]:
            raise ValueError(f"{name}: attention requires compatible [batch, heads, sequence, channels] tensors")
        if not k.shape[1] or q.shape[1] % k.shape[1] or not k.shape[2] or not q.shape[3]:
            raise ValueError(f"{name}: invalid attention head, key length, or channel dimensions")
        offset = _resolve(attrs.get("query_offset", 0), symbols, "query_offset")
        if offset < 0 or offset + q.shape[2] > k.shape[2]:
            raise ValueError(f"{name}: query positions exceed the key cache")
        scale = attrs.get("scale", 1.0 / math.sqrt(q.shape[3]))
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"{name}: attention scale must be finite and positive")
        shape = q.shape[:3] + (v.shape[3],)
        params = {"offset": offset, "scale": float(scale), "repeat": q.shape[1] // k.shape[1],
                  "scratch_elements": q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2]}
        work = 2 * params["scratch_elements"] * (q.shape[3] + v.shape[3])
    elif op == "rotary":
        _floating(arrays, name)
        if not x.ndim:
            raise ValueError(f"{name}: rotary requires a channel dimension")
        dim = _integer(attrs.get("rotary_dim", x.shape[-1]), "rotary_dim", 2)
        interleaved = _boolean(attrs.get("interleaved", False), "interleaved")
        if dim % 2 or dim > x.shape[-1] or arrays[1].shape != arrays[2].shape:
            raise ValueError(f"{name}: invalid rotary dimensions")
        target = x.shape[:-1] + (dim // 2,)
        if np.broadcast_shapes(target, arrays[1].shape) != target:
            raise ValueError(f"{name}: cos/sin do not broadcast to rotated half channels")
        params = {"dim": dim, "interleaved": interleaved}
    elif op in ("conv2d", "max_pool2d", "avg_pool2d"):
        _floating(arrays, name)
        if x.ndim != 4:
            raise ValueError(f"{name}: convolution and pooling use NCHW tensors")
        stride = _pair(attrs.get("strides", [1, 1]), "strides")
        pads = _padding(attrs.get("pads", [0, 0, 0, 0]))
        dilation = (1, 1)
        if op == "conv2d":
            w = arrays[1]
            groups = _integer(attrs.get("groups", 1), "groups", 1)
            dilation = _pair(attrs.get("dilations", [1, 1]), "dilations")
            if w.ndim != 4 or x.shape[1] % groups or w.shape[0] % groups or w.shape[1] != x.shape[1] // groups:
                raise ValueError(f"{name}: invalid grouped convolution channels")
            if len(arrays) == 3 and arrays[2].shape != (w.shape[0],):
                raise ValueError(f"{name}: convolution bias shape mismatch")
            kernel, channels = w.shape[2:], w.shape[0]
            if any(k == 0 for k in kernel):
                raise ValueError(f"{name}: convolution kernel cannot be empty")
            params["groups"] = groups
        else:
            kernel, channels = _pair(attrs.get("kernel"), "kernel"), x.shape[1]
            if op == "avg_pool2d":
                params["count_include_pad"] = _boolean(attrs.get("count_include_pad", False), "count_include_pad")
        height = (x.shape[2] + pads[0] + pads[2] - dilation[0] * (kernel[0] - 1) - 1) // stride[0] + 1
        width = (x.shape[3] + pads[1] + pads[3] - dilation[1] * (kernel[1] - 1) - 1) // stride[1] + 1
        if min(height, width) <= 0:
            raise ValueError(f"{name}: kernel does not fit padded input")
        shape = (x.shape[0], channels, height, width)
        params.update(stride=stride, pads=pads, dilation=dilation, kernel=kernel)
        params["scratch_elements"] = x.shape[0] * x.shape[1] * (x.shape[2] + pads[0] + pads[2]) * (x.shape[3] + pads[1] + pads[3])
        work = math.prod(shape) * math.prod(kernel) * (2 * arrays[1].shape[1] if op == "conv2d" else 1)
    else:  # validate_graph already rejects this, retained against dispatch drift.
        raise ValueError(f"unsupported operation: {op}")
    dtype = output_spec["dtype"]
    shape_ops = {"reshape", "transpose", "gather", "embedding", "slice", "concat", "identity"}
    if op in shape_ops and dtype != x.dtype.name:
        raise ValueError(f"{name}: data movement cannot change dtype")
    if op not in shape_ops | {"cast", "argmax"}:
        if x.dtype.kind == "f" and dtype not in ("float16", "float32"):
            raise ValueError(f"{name}: floating arithmetic requires a floating output")
        if x.dtype.kind == "i" and dtype not in ("int8", "int32"):
            raise ValueError(f"{name}: integer arithmetic requires an integer output")
    return tuple(shape), params, int(work)


def _softmax(x, axis):
    np = _numpy()
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True, dtype=np.float32)


def _compute(node, arrays, params):
    np = _numpy()
    op = node["op"]
    arithmetic = op not in {"reshape", "transpose", "gather", "embedding", "slice", "concat", "identity", "cast"}
    args = [a.astype(np.float32, copy=False) if a.dtype.kind == "f" else a.astype(np.int64, copy=False) for a in arrays] if arithmetic else arrays
    x = args[0]
    if op == "matmul":
        a, b = args
        if params["transpose_a"]:
            a = np.swapaxes(a, -1, -2)
        if params["transpose_b"]:
            b = np.swapaxes(b, -1, -2)
        if a.dtype.kind == "i":
            bound = max((abs(int(a.min())), abs(int(a.max()))), default=0) if a.size else 0
            bound *= max((abs(int(b.min())), abs(int(b.max()))), default=0) if b.size else 0
            bound *= a.shape[-1]
            if bound > np.iinfo(np.int64).max:
                raise ValueError(f"{node['name']}: integer matmul cannot guarantee exact int64 accumulation")
        return np.matmul(a, b)
    if op in ("add", "sub", "mul", "div"):
        return {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}[op](*args)
    if op == "reshape":
        return np.reshape(x, params["shape"])
    if op == "transpose":
        return np.transpose(x, params["perm"])
    if op in ("gather", "embedding"):
        return np.take(x, args[1], axis=params["axis"])
    if op == "slice":
        return x[params["slices"]]
    if op == "concat":
        return np.concatenate(args, axis=params["axis"])
    if op == "softmax":
        return _softmax(x, params["axis"])
    if op in ("layer_norm", "rms_norm"):
        if op == "layer_norm":
            centered = x - np.mean(x, axis=params["axes"], keepdims=True, dtype=np.float32)
        else:
            centered = x
        variance = np.mean(centered * centered, axis=params["axes"], keepdims=True, dtype=np.float32)
        result = centered / np.sqrt(variance + np.float32(params["epsilon"]))
        if len(args) > 1:
            result = result * args[1]
        if len(args) > 2:
            result = result + args[2]
        return result
    if op == "gelu":
        if params["approximate"] == "tanh":
            return np.float32(0.5) * x * (1 + np.tanh(np.float32(math.sqrt(2 / math.pi)) * (x + np.float32(0.044715) * x * x * x)))
        erf = np.asarray(np.frompyfunc(math.erf, 1, 1)(x.astype(np.float64) / math.sqrt(2)), dtype=np.float32)
        return np.float32(0.5) * x * (1 + erf)
    if op == "silu":
        result = np.empty_like(x)
        positive = x >= 0
        result[positive] = x[positive] / (1 + np.exp(-x[positive]))
        exp = np.exp(x[~positive])
        result[~positive] = x[~positive] * exp / (1 + exp)
        return result
    if op == "relu":
        return np.maximum(x, 0)
    if op == "argmax":
        return np.argmax(x, axis=params["axis"], keepdims=params["keepdims"])
    if op in ("cast", "identity"):
        if op == "cast" and node["output_dtype"] in ("int8", "int32") and x.dtype.kind == "f":
            return np.rint(x)
        return x
    if op == "reduce_mean":
        return np.mean(x, axis=params["axes"], keepdims=params["keepdims"], dtype=np.float32)
    if op == "causal_attention":
        q, k, v = args
        if params["repeat"] != 1:
            k, v = np.repeat(k, params["repeat"], axis=1), np.repeat(v, params["repeat"], axis=1)
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * np.float32(params["scale"])
        allowed = np.arange(k.shape[2])[None, :] <= (params["offset"] + np.arange(q.shape[2]))[:, None]
        scores = np.where(allowed, scores, -np.inf)
        return np.matmul(_softmax(scores, -1), v)
    if op == "rotary":
        cos, sin, dim = args[1], args[2], params["dim"]
        result = x.copy()
        if params["interleaved"]:
            first, second = x[..., :dim:2], x[..., 1:dim:2]
            result[..., :dim:2], result[..., 1:dim:2] = first * cos - second * sin, second * cos + first * sin
        else:
            first, second = x[..., :dim // 2], x[..., dim // 2:dim]
            result[..., :dim // 2], result[..., dim // 2:dim] = first * cos - second * sin, second * cos + first * sin
        return result
    if op in ("conv2d", "max_pool2d", "avg_pool2d"):
        return _spatial(op, args, params)
    raise ValueError(f"unsupported operation: {op}")


def _spatial(op, arrays, p):
    np = _numpy()
    x = arrays[0]
    pt, pl, pb, pr = p["pads"]
    sh, sw = p["stride"]
    dh, dw = p["dilation"]
    kh, kw = p["kernel"]
    h = (x.shape[2] + pt + pb - dh * (kh - 1) - 1) // sh + 1
    w = (x.shape[3] + pl + pr - dw * (kw - 1) - 1) // sw + 1
    padded = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), constant_values=(-np.inf if op == "max_pool2d" else 0))
    channels = arrays[1].shape[0] if op == "conv2d" else x.shape[1]
    result = np.empty((x.shape[0], channels, h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            patch = padded[:, :, i * sh:i * sh + dh * (kh - 1) + 1:dh, j * sw:j * sw + dw * (kw - 1) + 1:dw]
            if op == "conv2d":
                groups = p["groups"]
                ci, co = x.shape[1] // groups, channels // groups
                for group in range(groups):
                    result[:, group * co:(group + 1) * co, i, j] = np.einsum("nchw,ochw->no", patch[:, group * ci:(group + 1) * ci], arrays[1][group * co:(group + 1) * co], optimize=False)
            elif op == "max_pool2d":
                result[:, :, i, j] = np.max(patch, axis=(2, 3))
            else:
                count = kh * kw if p["count_include_pad"] else max(0, min(i * sh - pt + kh, x.shape[2]) - max(i * sh - pt, 0)) * max(0, min(j * sw - pl + kw, x.shape[3]) - max(j * sw - pl, 0))
                if not count:
                    raise ValueError("average pool window contains only padding")
                result[:, :, i, j] = np.sum(patch, axis=(2, 3), dtype=np.float32) / count
    if op == "conv2d" and len(arrays) > 2:
        result += arrays[2][None, :, None, None]
    return result


def execute(graph, inputs, *, overrides=None, node_callback=None):
    """Run a graph and return actual outputs, intermediates, and a scoped trace.

    A callback receives (node, input_arrays, output_spec, context). Return None to
    use the reference, an ndarray, or {"value": ndarray, "metadata": JSON object}.
    Callback output dtype and shape are mandatory and checked. Inputs are read-only
    owned copies. A callback must not mutate them or any external application state.
    Per-op overrides take precedence over the general callback. No execution trace
    field is interpreted as measured hardware performance by this executor.
    """
    import json
    np = _numpy()
    graph = validate_graph(graph)
    if not isinstance(inputs, Mapping) or set(inputs) != set(graph["inputs"]):
        raise ValueError("runtime inputs must exactly match graph inputs")
    if overrides is not None and (not isinstance(overrides, Mapping) or overrides.keys() - OPS.keys() or any(not callable(v) for v in overrides.values())):
        raise ValueError("overrides must map operation names to callables")
    if node_callback is not None and not callable(node_callback):
        raise ValueError("node_callback must be callable")
    values, symbols, trace = {}, {}, []
    total_elements, total_work = 0, 0
    limits = graph["limits"]

    def accept(name, raw, *, strict=True):
        nonlocal total_elements
        spec = graph["values"][name]
        # Inspect allocation sizes before copying user-owned buffers.
        shape = np.shape(raw)
        count = math.prod(shape)
        if count > limits["max_tensor_elements"] or total_elements + count > limits["max_total_elements"]:
            raise ValueError(f"{name}: tensor allocation exceeds configured element budget")
        _bind_shape(shape, spec, symbols, graph["shape_contract"], name)
        array = _array(raw, spec["dtype"], name, strict_array=strict)
        total_elements += array.size
        values[name] = array
        return array

    for name in graph["inputs"]:
        accept(name, inputs[name])
    for name, raw in graph["initializers"].items():
        accept(name, raw)
    for node in graph["nodes"]:
        args = [values[name] for name in node["inputs"]]
        spec = graph["values"][node["output"]]
        shape, params, work = _plan(node, args, spec, symbols)
        count = math.prod(shape)
        scratch = params.get("scratch_elements", 0)
        if max(count, scratch) > limits["max_tensor_elements"] or total_elements + count + scratch > limits["max_total_elements"]:
            raise ValueError(f"{node['name']}: inferred allocation exceeds configured element budget")
        if total_work + work > limits["max_operations"]:
            raise ValueError(f"{node['name']}: execution exceeds configured operation budget")
        _bind_shape(shape, spec, symbols, graph["shape_contract"], node["name"])
        total_work += work
        metadata = {}
        callback = (overrides or {}).get(node["op"], node_callback)
        result = None
        if callback is not None:
            context = {"symbols": dict(symbols), "output_shape": shape, "operation_count": work,
                       "reference_arithmetic": "float32_node_arithmetic_numpy_blas_or_exact_checked_integer"}
            result = callback(node, args, dict(spec), context)
        execution = "numpy_reference"
        if result is not None:
            execution = "backend_callback"
            if isinstance(result, Mapping):
                if set(result) - {"value", "metadata"} or "value" not in result:
                    raise ValueError(f"{node['name']}: callback must return value and optional metadata")
                metadata, result = result.get("metadata", {}), result["value"]
            if not isinstance(metadata, Mapping):
                raise ValueError(f"{node['name']}: callback metadata must be an object")
            try:
                json.dumps(metadata, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{node['name']}: callback metadata must be finite JSON") from exc
            if not isinstance(result, np.ndarray) or result.dtype != np.dtype(spec["dtype"]):
                raise ValueError(f"{node['name']}: callback must return ndarray of declared output dtype")
        else:
            try:
                with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                    result = _compute(dict(node, output_dtype=spec["dtype"]), args, params)
            except FloatingPointError as exc:
                raise ValueError(f"{node['name']}: nonfinite arithmetic or overflow") from exc
        output = accept(node["output"], result, strict=execution == "backend_callback")
        trace.append({"node": node["name"], "op": node["op"], "inputs": list(node["inputs"]),
                      "output": node["output"], "input_shapes": [list(a.shape) for a in args],
                      "output_shape": list(output.shape), "dtype": output.dtype.name,
                      "operations": work, "execution": execution, "metadata": dict(metadata)})
    return {"outputs": {name: values[name] for name in graph["outputs"]}, "intermediates": values,
            "trace": trace, "symbols": symbols, "operation_count": total_work,
            "stored_elements": total_elements, "execution_mode": "functional_reference_with_callbacks" if any(t["execution"] == "backend_callback" for t in trace) else "numpy_functional_reference",
            "numpy_version": np.__version__, "hardware_performance_validated": False}

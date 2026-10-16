# Tensor graph v2: executable application semantics

A decoder appends new keys and values to the previous cache, attends to the visible
prefix, and passes the updated cache into its next invocation. Tensor graph v2
represents those arrays as ordinary inputs and outputs. This allows prefill,
decode, attention, normalization, and linear layers to share one execution graph;
state does not disappear into an artifact-specific simulator.

This executor computes actual values on the CPU. Its NumPy reference does not
measure GPU, NPU, SRAM-CIM, or HBM-PIM performance. A platform adapter can replace
individual operations through a checked callback and attach separately scoped
simulation evidence. The callback's output becomes the next node's actual input.
Existing `typed-graph/v1` arithmetic and adapters keep their existing contract.

## Graph contract

`pim_foundry.tensor_ir.validate_graph(graph)` validates and normalizes a mapping.
The schema identifier is `pim-foundry.tensor-graph.v2`:

```json
{
  "schema": "pim-foundry.tensor-graph.v2",
  "values": {
    "cache": {"shape": ["B", "P", 64], "dtype": "float32"},
    "new": {"shape": ["B", "T", 64], "dtype": "float32"},
    "next_cache": {"shape": ["B", "S", 64], "dtype": "float32"}
  },
  "inputs": ["cache", "new"],
  "outputs": ["next_cache"],
  "initializers": {},
  "shape_contract": {"B": {"min": 1}, "S": {"max": 8192}},
  "nodes": [
    {"name": "append", "op": "concat", "inputs": ["cache", "new"],
     "output": "next_cache", "attrs": {"axis": 1}}
  ]
}
```

Values have explicit ranks, dimensions, and storage dtypes. Shapes may contain
nonnegative integers or named symbols. Zero dimensions support an initially empty
KV cache; scalar tensors have shape `[]`. A repeated symbol must have the same
size everywhere. Input shapes bind input symbols; operator shape inference binds
new output symbols, such as `S = P + T` above. An attribute using an unbound symbol
fails. Shape attributes do not execute expressions or Python code.

Every value has one source: a runtime input, an initializer, or one node. Nodes
must be in topological order. Undeclared values, duplicate writers, unused
declarations without producers, unknown fields, unknown operations, and unknown
attributes fail validation. Runtime input names must match exactly. Outputs may
include inputs or initializers, allowing explicit state passthrough.

Initializer shapes must be concrete. Initializers and inputs are disjoint.
Programmatic initializers accept NumPy arrays; JSON initializers use nested lists
whose shape matches the declaration. No pickle, dynamic import, or executable
object is loaded by the graph schema. Applications loading external weights must
validate and bind those assets separately.

## Numerical contract

Supported storage dtypes are `float16`, `float32`, `int8`, and `int32`. NumPy input
and initializer arrays must already have the declared dtype. JSON numeric lists
are converted according to the explicit declaration. Mixed operand dtypes require
cast nodes. Inputs, initializers, and stored outputs must be finite.

Floating arithmetic uses float32 node operands, then rounds once to the declared
output dtype. For example, a float16 matmul accumulates in float32 and stores its
output as float16 or float32. Matmul reduction order follows the installed
NumPy/BLAS implementation. This differs from v1's sequential non-fused scalar MAC
contract. It does not promise bit-identical output across BLAS libraries, CPU
architectures, or NumPy releases. Compare independent implementations with a
stated tolerance and record the environment.

Integer arithmetic uses an exact int64 temporary and checks representability of
the final declared output. Integer matmul requires an int32 output and rejects
inputs whose absolute-product bound cannot guarantee exact int64 accumulation.
It does not implement saturation or wraparound. An explicit cast from float to
integer rounds ties to even, then checks range; it is not a quantization scale or
zero-point operator. Integer division and integer normalization are unsupported.

Softmax subtracts each row's maximum before exponentiation. Layer normalization
uses population variance; RMS normalization uses the mean square. Both reject
nonpositive epsilon. Floating overflow, division by zero, empty normalized axes,
and nonfinite output fail instead of producing a plausible benchmark result.
The causal-attention implementation uses an internal negative-infinity mask;
that mask is not an externally stored graph tensor.

## Operators

All operators have one named output. Axis indices accept the usual negative
indices. Shape-only operations preserve storage dtype.

| Operation | Inputs and attributes |
|---|---|
| `matmul` | Two vectors or matrices, including broadcast batch dimensions; optional boolean `transpose_a`, `transpose_b` on the final two axes. |
| `add`, `sub`, `mul`, `div` | Two inputs with NumPy-style broadcasting. Division requires floating inputs. |
| `reshape` | One input; required `shape` list with concrete dimensions, bound symbols, and at most one `-1`. Zero is a literal zero dimension. |
| `transpose` | One input; `perm` is a complete permutation; default reverses the axes. |
| `embedding` | `[weights, indices]`; rank-two weights and int32 indices. Negative embedding IDs are rejected. |
| `gather` | `[data, indices]`; int32 indices and `axis` (default 0). In-range negative gather indices are accepted. |
| `slice` | One input; equal-length `starts`, `ends`; optional `axes`, `steps`. Uses Python slice endpoints, including negative steps. |
| `concat` | One or more inputs, `axis` (default 0). Other dimensions must agree. |
| `softmax` | One floating input, `axis` (default -1). |
| `layer_norm` | `[x]`, `[x, gamma]`, or `[x, gamma, beta]`; `axis` begins the normalized suffix, default -1; `epsilon` defaults to 1e-5. Parameter shapes must exactly match the suffix. |
| `rms_norm` | `[x]` or `[x, gamma]`; suffix `axis`, default -1; `epsilon` defaults to 1e-6. |
| `gelu` | `approximate` is `none` (erf) or `tanh`; default `none`. |
| `silu`, `relu`, `identity` | One input. |
| `cast` | One input; optional `dtype` must match the declared output dtype. |
| `reduce_mean` | Floating input; `axes` defaults to every axis, `keepdims` defaults to true. |
| `argmax` | One input, `axis` default -1, `keepdims` default false; first index wins ties, output int32. |
| `causal_attention` | `[q, k, v]` in `[batch, heads, sequence, channels]` order; optional `query_offset` (default 0, integer or bound symbol) and positive `scale` (default 1/sqrt(query channels)). Supports grouped-query attention when query heads are divisible by KV heads. |
| `rotary` | `[x, cos, sin]`; cos/sin broadcast to the rotated half-channel shape. Even `rotary_dim` defaults to all channels; `interleaved` chooses paired channels (default false, split-half convention). Unrotated channels pass through. |
| `conv2d` | `[x, weights]` or `[x, weights, bias]`; NCHW/OIHW layout; `strides`, `dilations`, `groups`, `pads` in top/left/bottom/right order. |
| `max_pool2d`, `avg_pool2d` | NCHW input; required `kernel`, optional `strides`, `pads`. Floor output sizing. Average-pool `count_include_pad` defaults to false. |

Attention requires `query_offset + query_length <= key_length`. Query position
`query_offset + t` can see only keys at positions no greater than that position.
For cached decode, append current K/V first and pass the prior cache length as
`query_offset`. The executor does not silently truncate, evict, or reuse a cache.
RoPE positions, tokenization, padding masks, and application stopping rules remain
explicit application responsibilities. The fused attention operation implements a
causal mask, not an arbitrary padding or sliding-window mask; graphs needing
those can express score matmul, finite additive mask, softmax, and value matmul.

## Execution and backend callbacks

```python
from pim_foundry.tensor_executor import execute
result = execute(graph, inputs, node_callback=my_backend)
outputs = result["outputs"]
```

`execute(graph, inputs, *, overrides=None, node_callback=None)` returns actual
NumPy arrays in `outputs` and `intermediates`, plus `symbols`, an operation trace,
operation-count estimates, retained element counts, NumPy version, and an explicit
functional execution-mode label. Arrays are owned, read-only copies. All node
intermediates are retained for numerical auditing.

A callback has the signature:

```python
def my_backend(node, input_arrays, output_spec, context):
    # Return None to use the NumPy implementation for this node.
    # Otherwise return an ndarray with exact output dtype and shape, or:
    return {"value": output_array, "metadata": {"backend": "my_backend"}}
```

`context` includes concrete `output_shape`, bound `symbols`, `operation_count`,
and the reference arithmetic label. Callback outputs must have the exact declared
storage dtype and inferred shape, and must be finite. Metadata must be finite
JSON. An `overrides` mapping selects a callback by operation and takes precedence
over the general callback. Callbacks are trusted Python plugin code, not a
sandbox; they must not mutate inputs or hide application state.

Trace entries identify input/output shapes, storage dtype, estimated arithmetic
work, implementation (`numpy_reference` or `backend_callback`), and callback
metadata. A NumPy wall-clock duration is not emitted as accelerator latency.
`hardware_performance_validated` is always false in this functional executor.
Hardware evidence requires a separate platform evaluator and calibration record.

## Resource limits and verification

Optional graph `limits` override positive integer limits:

- `max_tensor_elements`: 250,000,000 by default.
- `max_total_elements`: 500,000,000 by default.
- `max_operations`: 100,000,000,000 by default.

Input and inferred output sizes are checked before copying or executing a node.
The total includes all retained graph values and explicitly estimated attention
or convolution/pooling workspace. These are conservative logical-element/work
budgets, not an operating-system memory limit or exact BLAS allocation model.
Temporary dtype conversions, NumPy internals, and trusted callback allocations
can require additional memory. Set process-level limits when running untrusted
or large graphs. Users may raise graph limits for real workloads; this contract
does not hide an arbitrary small-model-only limit.

`tests/test_tensor_runtime.py` checks scalar and batched arithmetic, independent
scalar grouped-query attention, future-token isolation, full-sequence versus
incremental-cache equivalence, normalization equations, actual callback output
propagation, and invalid states. Independent ONNX reference tests cover MatMul,
LayerNormalization, grouped/dilated convolution, and padded pooling. Rejection
tests cover dtype mismatch, overflow, division by zero, malformed graph structure,
symbol bounds, invalid indexing, callback failures, and work budgets. Passing
these tests establishes the documented functional contract; it does not calibrate
accelerator timing, validate every possible application, or reproduce a paper's
hardware claims.

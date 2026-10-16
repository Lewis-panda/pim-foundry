# Typed numerical contract

An SRAM kernel can write signed INT32 sums while the next HBM kernel reads
FP16 values. Passing the same bytes between them changes the numbers. The shared
graph therefore declares storage dtype on every tensor and inserts a `cast`
operation where conversion is required. A backend must either implement that
operation under its declared policy or reject it; the scheduler cannot silently
reinterpret the bytes.

`pim_foundry.typed` is an executable numerical contract with a deterministic scalar
reference. It does not establish the arithmetic of a timing-only simulator.
Backend interpretation and numerical evidence must remain separate from native
simulator timing evidence.

## Shapes, layout, and operations

All tensors use dense row-major storage and have one or two positive dimensions.
Their flat initializers must match the full shape. Storage types are `int8`,
`int32`, `float16` (IEEE binary16), and `float32` (IEEE binary32).

| Operation | Inputs | Output shape | Numeric requirements |
| --- | --- | --- | --- |
| `matvec` | X[K], W[K,N] | Y[N] | Both inputs have the same INT8, FP16, or FP32 dtype |
| `matmul` | X[M,K], W[K,N] | Y[M,N] | Same requirement; rows reduce independently |
| `add` | A[S], B[S] | Y[S] | Same shape and dtype; no broadcasting |
| `relu` | X[S] | Y[S] | Negative values and signed zeros become positive zero |
| `cast` | X[S] | Y[S] | Explicit storage conversion under the node policy |

`S` denotes an identical rank-one or rank-two shape. Both weights and activations
may be graph inputs or constants. Each produced tensor has exactly one producer;
node order must already be topological. Fan-out and repeated operands are valid.
There are no implicit mixed-dtype inputs, padding, transpose, bias, or broadcast.
Backend padding must retain these logical shapes and exclude padded results.

## Integer arithmetic

INT8 linear inputs accumulate into signed INT32. Every accumulation applies the
node's `overflow` setting, including intermediate sums:

- `error` raises on a value outside [-2³¹, 2³¹−1]. This is the default.
- `saturate` clamps each intermediate sum to that range.
- `wrap` reduces each intermediate sum modulo 2³² with signed interpretation.

`add` follows the same rule. INT32 linear operands are rejected. An INT32 result
may be stored directly. An INT8 result requires an explicit `Requantization`,
except an INT8-to-INT8 ReLU or identity cast.

`Requantization(multiplier, shift, zero_point)` defines:

```
INT8 = clamp_to_int8(round_ties_even(value * multiplier / 2**shift) + zero_point)
```

The positive multiplier is at most 2³¹−1, the shift is 0 through 62, and the zero
point is an INT8 integer. Multiplication and rational rounding are exact; there is
no hidden floating scaling step. A `cast` from a floating tensor can use the same
policy: the binary floating value is converted to its exact rational value before
scaling and rounding. No implicit FP-to-INT32 rounding occurs before the scale.

This definition does not imply that a particular accelerator implements the
requantization natively. Backend support must be explicit.

## Floating arithmetic

FP16 or FP32 storage values are rounded to their declared dtype on entry, using
IEEE round-to-nearest, ties-to-even. Every floating linear node uses an FP32
accumulator, starting at positive zero. For increasing `k`:

```
product = round_fp32(X[..., k] * W[k, n])
accumulator = round_fp32(accumulator + product)
```

The multiplication and addition are **separate**, not fused. Reduction order is
part of the contract. A backend with an FMA or tree reduction cannot claim exact
agreement without checking the resulting arithmetic. Floating `add` rounds its
sum to FP32. Output storage explicitly rounds to FP16 or FP32.

NaN, infinity, and storage or accumulator overflow raise an error. Subnormal
values and underflow to signed zero follow IEEE conversion; values are not flushed
to zero by policy. FP16-to-FP32 conversion is exact. INT32-to-floating conversion
uses the declared output dtype; FP16 overflow is rejected. Floating-to-INT32 casts
round ties to even, then apply the integer overflow policy.

The standard-library reference uses `struct` for binary16/binary32 conversion and
Python binary64 operations between conversions. Products of two FP32 operands are
exactly representable in binary64 before the explicit FP32 rounding. Tests include
FP16 half-way values, subnormals, signed zero, an example separating fused and
non-fused arithmetic, and a reduction-order example.

## Shared API

```python
from pim_foundry.typed import Arithmetic, Graph, Node, Tensor, execute

graph = Graph(
    tensors=(
        Tensor("x", (2,), "int8"),
        Tensor("w", (2, 2), "int8", (-2, 5, 3, -4)),
        Tensor("sum", (2,), "int32"),
        Tensor("half", (2,), "float16"),
    ),
    nodes=(
        Node("sram_linear", "matvec", ("x", "w"), "sum"),
        Node("convert", "cast", ("sum",), "half", Arithmetic("float32")),
    ),
    inputs=("x",),
    outputs=("half",),
)
assert execute(graph, {"x": (-7, 2)}) == {"half": (20.0, -43.0)}
```

`prepare_inputs(graph, inputs)` validates inputs and returns initial values,
including constants. `execute_node(node, tensor_map, values)` evaluates one
operation. `evaluate` returns all intermediate values; `execute` returns only
graph outputs. These interfaces let a scheduler move typed values between
backends without changing numerical policy.

`graph_to_dict` and `graph_from_dict` implement the strict
`pim-foundry.typed-graph.v1` JSON schema. All fields, including rounding, overflow,
requantization, and layout, are explicit; unknown fields are rejected.
`graph_digest` hashes its canonical representation, including constant values.
External runtime input values must be hashed separately by the experiment layer.

The graph permits at most 256 nodes and one million elements per tensor. Scalar
evaluation is limited to 20 million MACs or element operations in total. These
limits bound reference execution and are not hardware-capacity claims.

## What a numerical pass establishes

`compare_outputs` requires exact integer equality even if a floating tolerance is
provided. Floating comparison uses
`abs(actual - reference) <= atol + rtol * abs(reference)` with explicit finite,
nonnegative tolerances. The actual values must already be representable in their
declared storage dtype. Nonfinite or incorrectly typed values fail instead of
being rounded to hide a mismatch.

A comparison against values produced by an instruction interpreter establishes
agreement of that interpreter and lowering with this contract for the tested
inputs. It does not establish numerical behavior of an upstream simulator that
emits only timing statistics. Hardware validation additionally requires actual
hardware output and its numeric execution policy.

`BackendCapabilities` reports declared operations, input/output storage dtypes,
and accumulator dtypes and rejects incompatible nodes. Shape limits, ISA-specific
restrictions, supported overflow/requantization policies, and provenance remain
mandatory checks in the concrete backend; listing a capability is not execution
evidence.

Run the independent contract checks with:

```sh
python3 -m unittest tests.test_typed -v
```

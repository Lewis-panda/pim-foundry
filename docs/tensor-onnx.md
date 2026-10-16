# ONNX applications on the common tensor graph

A CNN's convolution output must flow through pooling and its classifier before a
classification latency can be reported. The tensor ONNX frontend imports those
operators into the same executable graph, so a platform evaluator can retain the
complete application path and compare its actual outputs with the original ONNX
model. Attention graphs likewise retain score construction, masking, softmax,
value aggregation, residual addition, and normalization.

`pim_foundry.tensor_onnx.load_model(path, input_shapes=None)` returns a
`pim-foundry.tensor-graph.v2` mapping. It requires the optional ONNX/NumPy
dependencies. It runs ONNX validation and shape inference, preserves source
identity, and rejects unsupported semantics instead of deleting operators.
This frontend is a documented subset, not a claim to execute arbitrary ONNX.

```python
from pim_foundry.tensor_onnx import load_model, prepare_inputs, compare_reference
from pim_foundry.tensor_executor import execute

graph = load_model("model.onnx", input_shapes={"image": [1, 3, 224, 224]})
feeds = prepare_inputs(graph, {"image": image_array})
actual = execute(graph, feeds)["outputs"]
check = compare_reference("model.onnx", {"image": image_array},
                          input_shapes={"image": [1, 3, 224, 224]},
                          actual_outputs=actual, atol=1e-5, rtol=1e-5)
```

Runtime input arrays must have their original ONNX dtype. JSON-style lists are
converted under the model's explicit dtype declaration. `prepare_inputs` also
maps names that are legal in ONNX but not in the tensor identifier grammar.
Returned graph metadata contains:

- `metadata.onnx.input_names`: original input name → tensor graph name.
- `metadata.onnx.output_names`: original output name → tensor graph name.
- Original source SHA-256, ONNX opset/version, and recorded operator lowerings.
- Original input dtypes and every int64 value narrowed under the index policy.

`compare_reference` runs `onnx.reference.ReferenceEvaluator` on the original
model. It compares supplied backend outputs when `actual_outputs` is provided,
or executes the common functional graph otherwise. Shapes, finiteness, and
actual graph dtypes are checked before numerical comparison. Integer outputs
require exact equality; floating outputs use the stated absolute and relative
tolerance. Reports are limited to the supplied model and input values and always
mark hardware performance as unvalidated. A source hash check prevents comparing
against a model changed after import.

## Supported conversion contract

Only one standard ONNX opset, versions 13–23, is accepted. Nodes must have one
output. The importer supports these operations under the following restrictions:

| ONNX operations | Conversion and restrictions |
|---|---|
| MatMul, Add, Sub, Mul, Div | Common tensor operators, including supported broadcasting. |
| Gemm | Rank-two matrix operands; transposition, alpha, beta, and broadcast bias are explicitly lowered into matmul/mul/add nodes. |
| Relu, Identity, Softmax, Gelu | Direct mapping; GELU preserves `none`/`tanh` choice. |
| Transpose | Explicit permutation or default reversal. |
| Reshape | Shape must be constant after shape-program folding. ONNX zero-copy dimensions are converted explicitly; `allowzero` is retained through the resulting literal shape. |
| Flatten | Dynamic dimension products require `input_shapes` unless the product is already a single bound symbol. |
| Squeeze, Unsqueeze | Constant axes. Squeezed dimensions must be statically one. |
| Gather, Concat, Slice | Gather indices are int32 or checked int64. Slice controls must be constant. |
| LayerNormalization | Float32 accumulation, one output; optional beta, exact suffix parameter shapes. |
| ReduceMean, GlobalAveragePool | Constant reduction axes, explicit keepdims. |
| Cast | Supported dtype conversions whose semantics agree with tensor v2. Float-to-integer Cast is rejected because ONNX truncation differs from v2 ties-to-even rounding. |
| ArgMax | First-index tie rule; output represented as checked int32. |
| Conv | NCHW/OIHW 2D convolution, explicit padding or VALID, groups and dilation. |
| AveragePool, MaxPool | NCHW 2D pooling, floor output sizing, explicit padding or VALID; no index output or dilated pooling. |
| Constant, Shape | Constants and statically bound shapes. Integer Gather/Concat/Squeeze/Unsqueeze/Add/Sub/Mul/Div/Cast shape programs are folded with range checks. |

Floating input and weight dtypes are float16/float32; integer storage is
int8/int32. **General int64 data arithmetic is not supported.** Runtime int64
inputs are accepted only as Gather indices; values must fit int32 before any
execution. Compile-time integer shape controls may retain int64 values, and
ArgMax results have a bounded int32 index representation. This narrower domain
is recorded in metadata; it is not silently presented as arbitrary int64 support.
An index exceeding the supported range fails.

Symbols may remain dynamic when the common operator can infer their runtime
shape. A `Shape` program that needs actual dimension values requires
`input_shapes`. Bindings cannot change declared static dimensions or assign
conflicting sizes to a shared symbol. Converting ONNX names and symbols does not
change the tensor dimensions or values.

Unsupported domains, local functions, training graphs, sparse initializers,
external weight files, overridable initializers, unsupported operators,
unsupported output types, nonfinite constants, and ambiguous shapes fail with an
error. External weights require future integration with explicit application
asset binding; the importer does not follow unverified paths. Supported
operators also fail when the requested attribute has no matching common-IR
semantics, including last-index ArgMax, automatic SAME padding, or ceil-mode
pooling.

The common graph uses float32 node arithmetic and the installed NumPy/BLAS
reduction order. Original ONNX implementations may round in a different order,
especially with float16. Numerical equivalence must therefore be evaluated for
the actual application and precision; import success alone is not equivalence.

## Acceptance evidence

`tests/test_tensor_onnx.py` compares complete original ONNX graphs against their
imported execution: a CNN with its pooling and classifier, an attention block
with residual normalization, dynamic token embedding, a bound integer shape
program, reshape/transpose, Gemm/ArgMax, and global pooling/GELU. A deliberately
wrong backend output must fail comparison. Negative cases cover unsupported
operators, incompatible Cast semantics, unsafe int64 inputs, nonfinite weights,
external tensors, and unsupported tie selection.

These are compiler/runtime interoperability fixtures with deterministic random
weights. They establish application-path and numerical behavior for the tested
cases. They do not establish trained-model accuracy, paper reproduction, or
hardware timing fidelity.

The sealed acceptance run is `runs/onnx-tensor-v08/application/`, with its frozen
replay in `runs/onnx-tensor-v08/replay/` and audit summary in
`runs/onnx-tensor-v08/summary.json`. Both bundles verified all 49 recorded files;
all eight replay comparisons matched. The complete CNN produced two four-class
probability vectors with maximum absolute difference
`2.9802322387695312e-08` against the original ONNX evaluator. Its three-request
schedule uses a declared, uncalibrated analytical GPU profile. The model is a
random-weight interoperability fixture; this result is not trained classification
accuracy or measured GPU performance.

# ONNX application input

A model file contains tensor names, learned weights and operations. The ONNX
frontend reads those objects into the same typed graph consumed by the SRAM,
HBM and accelerator adapters. An application does not need to name a simulator
or manually construct backend instructions. Platform selection and placement
remain deployment choices, outside the ONNX model.

The frontend is optional. The core package still has no runtime dependencies.
The tested environment is Python 3.12 with:

```text
onnx==1.19.0
numpy==2.2.6
protobuf==6.32.1
typing_extensions==4.15.0
ml_dtypes==0.5.3
```

Create a local environment and install these versions there. The repository's
`.venv-onnx` is the environment used for the recorded validation. No global
packages were installed.

```python
from pim_foundry.onnx_frontend import load_model
from pim_foundry.typed import execute

app = load_model("model.onnx", input_shapes={"features": [2, 16]})
prepared = app.prepare_inputs({"features": [[0.25] * 16, [-0.5] * 16]})
outputs = execute(app.graph, prepared)  # Scalar sanity check; use unified.run for device adapters.
quality = app.compare_reference(
    {"features": [[0.25] * 16, [-0.5] * 16]}, outputs,
    atol=0.015625, rtol=0.002,
)
assert quality["passed"]
```

`load_model(path, input_shapes=...)` returns `ImportedModel.graph`,
`ImportedModel.input_schema` and `ImportedModel.report`. `prepare_inputs` accepts
source-shaped nested values or flat row-major values with the correct element
count. It validates integer ranges before any NumPy dtype conversion.
`reference_outputs` runs ONNX's original serialized model with its independent
`ReferenceEvaluator`; `compare_reference` compares its outputs with actual IR or
backend outputs. Output aliases and boundary views are retained in
`report["output_mapping"]`.

The report binds source bytes and the resulting typed graph by SHA-256. It
records every original node, its lowering, constant transpose/expansion,
input shape bindings and explicit precision conversions. Import success alone
does not set numerical-validation or performance-evidence flags.

## Implemented operator contract

Only standard ONNX opsets 13–22 are accepted. The file must be self-contained,
at most 32 MiB and at most 128 source nodes. Tensors have at most one million
elements. Runtime dimensions must be positive and fully bound. A binding may
resolve a symbolic dimension but cannot alter a static dimension or give
conflicting values to the same symbolic dimension.

| Operator | Supported lowering and restrictions |
|---|---|
| MatMul | FP16/FP32 activation `[K]` or `[M,K]`, constant weights `[K,N]`; rank-1 becomes matvec. |
| Gemm | FP16/FP32, `alpha=beta=1`, `transA=0`; constant `transB=1` is transposed at import; optional constant bias broadcast is materialized and charged as stored data. |
| Add | Equal runtime shapes or broadcast initializer; FP16/FP32 and INT32 wrapping. INT8 Add is rejected because its wrapping semantics differ from IR requantization. |
| Relu | Storage dtype retained, subject to the ONNX version's type constraints. |
| Cast | Explicit float16↔float32, integer→float, int8→int32 or same-type conversion. Float→integer and narrowing integer casts are rejected because their truncation/wrapping differs from the IR policy. |
| Flatten | Constant reshape; identity shape; or a direct unshared application input view, preserving row-major bytes and original input schema. |
| Identity | Alias with an explicit output mapping. |

Other operators, non-default unsupported attributes, custom domains, external
weight files, local functions, sparse initializers, overridable initializers,
unbound dimensions and incompatible ranks are rejected. Errors identify the
source node where applicable. No operator is silently removed; view operations
are recorded explicitly. Float-to-INT8 quantization is not inferred to make a
model fit SRAM.

## Numerical and evaluation boundary

The shared IR uses FP32 separate multiply and add in increasing K order, then
rounds each stored result to its declared dtype. ONNX does not impose one
universal floating reduction order. Gemm lowering also materializes its matrix
product before bias addition. A backend using a different reduction or fused
Gemm may differ; callers must specify a quality threshold and compare the actual
outputs on their evaluation inputs. Passing these finite-input checks is not a
proof for every input, a trained-model accuracy score, or hardware validation.
Nonfinite values and floating overflow are rejected by the existing IR contract.
See the official [Gemm operator](https://onnx.ai/onnx/operators/onnx__Gemm.html)
and [Cast operator](https://onnx.ai/onnx/operators/onnx__Cast.html) definitions.

To repeat the recorded interoperability experiment:

```sh
.venv-onnx/bin/python -m unittest discover -s tests -p test_onnx_frontend.py -v
.venv-onnx/bin/python scripts/evaluate_onnx_frontend.py --output runs/onnx-new-evaluation
```

The evaluation emits four actual ONNX files: FP32 MLP, FP16 MLP, an INT8-input
MLP with an explicit FP16 Cast, and a Flatten-input MLP. Each is tested with
three batch sizes and eight input seeds (96 input sets total). They are
interoperability fixtures with generated weights, **not trained paper models**.
Reports, exact input values, independent outputs, import diagnostics and pinned
dependency versions are kept beside each model. Use the paper-application
experiment reports separately when assessing reproduction of a published paper.

Future application support should add a frontend operator only when its shape,
dtype, overflow, rounding and view semantics can be represented, then validate
it against an independent implementation. Backend capability checks select
compatible devices; a cost model may choose among them without changing the
application's numerical contract.

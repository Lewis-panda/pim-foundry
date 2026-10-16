> Paths under `runs/` or `external/` refer to locally generated or separately obtained artifacts, not bundled files. Start with [the public quickstart](getting-started.md); see [validation scope](../VALIDATION.md) for historical observations.

# Shared tensor execution and resource scheduling

A matmul placed on HBM consumes the actual bytes produced by its predecessor. Its instruction interpreter produces the tensor passed to the next node. Device transfers are represented as dependency events with exclusive link and endpoint-DMA reservations. The simulator's wall-clock execution duration is never a device metric.

`tensor_platform.run(graph, inputs, config, placement=None, output=None)` executes the v2 tensor graph. The platform retains the v1 explicit `schema_version`, `devices`, and `links` contract. Results contain outputs, intermediate arrays, an operator trace, and `platform_execution` with a reusable profile and event schedule.

## Backend contracts

| Backend | Actual numerical execution | Timing evidence |
|---|---|---|
| `analytical` | Shared NumPy tensor operators | Supplied rate, bandwidth and launch coefficients; uncalibrated by default |
| `cimflow-sram` | INT8 × INT8 → INT32 through the CIMFlow instruction/memory interpreter; checked against scalar arithmetic | Native CIMFlow simulator plus explicit packing |
| `hbm-simd16` | FP16 × FP16 → FP16, sequential separate FP32 multiply/add through the Foundry SIMD16 interpreter; independently checked | Patched DCC request timing projection plus explicit packing |

Device `kind` names CPU, GPU, NPU, HBM or SRAM; a kind alone does not establish physical execution. Native backends accept explicitly placed matmuls. Unsupported operators or dtypes fail; no implicit conversion or silent fallback occurs. Casts are ordinary graph nodes. Native binary/descriptor hashes participate in platform identity; detailed native results preserve their runtime verification.

Broadcast batches are expanded into native matrix calls. SRAM tiles M≤16 and N≤128; K>128 is rejected because cross-tile reduction would require an explicit additional graph operation and accounting. A conservative reduction bound prevents intermediate overflow. HBM uses bounded row/column tiles without splitting K, preserving the declared accumulation order. These are research lowerers, not physical GPU/NPU drivers or a Samsung instruction encoding.

Analytical matmul, convolution and attention costs count contraction MACs; fused attention also counts its quadratic score operations and workspace. Other operators use a documented tensor-element proxy. An exp or normalization element has no automatically calibrated physical cost. Throughput, bandwidth, packing, energy and idle-power assumptions remain explicit. A zero coefficient is a model assumption, not a zero-power observation.

## Scheduling and capacity

`schedule(profiles, config, arrivals=None, max_inflight=1, state='cold')` accepts execution profiles. Nodes wait for operand copies and the destination compute resource. Transfers reserve the directed link and both endpoints' DMA resources; independent compute and transfer can overlap. Deterministic request/node order inserts work into earliest free resource intervals. This is a feasible list scheduler, not an optimal scheduler.

Inputs, weight copies, intermediates, output buffers and native packing workspace contribute lifetime intervals. Capacity is checked across all concurrent requests. No eviction is performed. Multiple stored copies are charged conservatively. Host arrays retained for diagnostics are separate from modeled device residency. Every issued node contributes to request completion, including unused outputs. Makespan excludes time before the first arrival; absolute observation start/end are reported separately.

Cold state reloads initializers per request. Warm state loads each byte-identical initializer once per destination and keeps it resident through the study; the first upload is included. This differs from older `suite` warm-preload semantics and has a separate cost contract. Request inputs and KV tensors remain request-owned. The Transformer adapter transfers KV state through host graph boundaries; it does not claim device-persistent KV or continuous batching. Within each graph, tensors use the shared dependency schedule.

Identical requests may reuse one verified operation profile. This predicts a workload stream and does not create independent empirical samples. `comparison` requires separate verified runs for repetition evidence.

## Commands and evidence

```sh
python3 -m pim_foundry tensor-run --graph graph.json --inputs inputs.json \
  --platform platform.json --placement placement.json --out runs/custom-tensors
python3 -m pim_foundry tensor-schedule runs/custom-tensors/platform-execution.json \
  --platform platform.json --max-inflight 2 --state warm --out runs/schedule.json
```

For sealed bundles, use `tensor-graph/v2` or `onnx-tensor/v2` through `app-run`. These adapters require a content-bound external reference array asset. Numerical agreement and task-quality evaluation remain distinct.

The four-platform acceptance (local artifact: `../runs/tensor-platform-v08-release/summary.json`) executes batched SRAM→cast→HBM→cast→softmax, HBM+GPU, HBM+NPU and CPU profiles with actual intermediate tensors. Bundles are verified and replayed; a deliberately wrong reference produces `quality_failed`. Tests reject missing routes, malformed timings, wrong placement, implicit precision and insufficient capacity, and verify exclusive resource intervals and timestamp-translation invariance.

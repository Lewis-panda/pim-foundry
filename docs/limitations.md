# Current limits

A tensor can match its reference while the supplied bandwidth is wrong. PIM Foundry therefore keeps functional checks, predicted cost, paper reproduction, and hardware measurements separate. The following limits apply to the public 0.8 implementation.

## Hardware and cost models

- GPU/NPU configurations use analytical costs and CPU functional execution. There are no bundled physical GPU/NPU execution drivers.
- The HBM instruction interpreter is a Foundry numerical model. Its DCC request projection is not a vendor instruction encoding or a measured HBM-PIM device. The patched runtime's explicit completion assumptions remain part of the model.
- CIMFlow integration separates instruction-level arithmetic from native timing. This does not establish that the original timing simulator computes tensor results.
- Device throughput, bandwidth, launch, packing, idle power, and energy coefficients need independent calibration for a physical prediction. Example coefficients are synthetic assumptions. Unknown energy is retained as unknown.
- Ordinary tensor-element cost proxies do not establish calibrated costs for exponentials, normalization, or every supported operation.

## Application coverage

The graph implements the operations listed in [tensor-runtime.md](tensor-runtime.md), with FP16, FP32, INT8, and INT32 storage. Unsupported operations or precision are errors. There is no general quantization optimizer, training engine, arbitrary-precision compiler, or automatic coverage of every model architecture.

The [ONNX frontend](tensor-onnx.md) supports a stated subset of standard opsets 13–23. External tensor files, custom domains, control flow, arbitrary int64 data arithmetic, and several operator attributes are unsupported. Import success must be followed by numerical evaluation on the actual model and input domain.

The complete Transformer adapter implements the GPT-2 architecture and equal-length batches. KV persists as explicit host arrays across graph calls; transfers across those boundaries remain in the cost scope. Device-persistent KV, ragged/continuous batching, speculative decoding, and general serving systems are outside this implementation.

Native backends have narrower arithmetic and geometry limits than the common graph. SRAM K greater than 128 requires explicit graph-level reduction; it is rejected by the current lowerer. Native HBM execution is bounded and uses explicit FP16 conversion where selected. A cast alone is not a quantization scale/zero-point scheme.

## Scheduling and numerical reproducibility

The scheduler builds a deterministic feasible schedule with shared compute, DMA, and directed links. It is not an optimal scheduler. Capacity accounting has no eviction or demand paging, and conservative residency assumptions can overestimate memory use. Host diagnostic arrays are separate from modeled device residency.

NumPy/BLAS reductions may differ across environments; numerical tolerances must be explicit. An exact deterministic replay can fail when dependency or arithmetic behavior changes. Such a failure must remain visible rather than being averaged away or relabeled as a successful reproduction.

Profile reuse predicts repeated identical requests. It does not generate independent timing samples, and per-request metrics from one run are not automatically independent repetitions.

## What existing evaluation establishes

Historical DistilGPT-2 checks cover a complete six-layer graph, cached decoding, and 128 next-token targets from a fixed WikiText prefix. That is an implementation/quality-preservation fixture, not full WikiText perplexity or broad application quality. Tiny GPT-2 exercises the complete native HBM integration path; its small checkpoint and fixture do not establish useful language quality or production-scale performance.

Paper adapters cover the explicitly documented workloads and simulator paths. Several upstream issues were observed, including a CIMFlow timing reproducibility failure. An intact bundle cannot make a failed study eligible for a supported performance conclusion. IANUS remains a source/claim audit without an authenticated original author executable in the evaluated material. See [research evidence](research-evidence.md).

## Distribution and trust

The repository does not include historical private-workspace run bundles, downloaded paper PDFs, upstream source trees, model weights, or dataset shards. Optional studies require their own assets, licenses, and build prerequisites. Public checkout validation and skipped optional tests are recorded separately in [VALIDATION.md](../VALIDATION.md).

File hashes establish integrity, not external authentication or community acceptance. The comparison protocol cannot discover deliberately withheld studies or validate the honesty of arbitrary supplied measurements. Calibration fixtures in tests exercise validation logic; they are not physical-device calibration evidence.

# NeuPIMs artifact audit and execution

NeuPIMs (ASPLOS 2024, DOI [10.1145/3620666.3651380](https://doi.org/10.1145/3620666.3651380)) provides an independent NPU/HBM-PIM simulator lineage: an NPU cycle model connected to an in-house DRAMsim3-derived PIM model. The [author repository](https://github.com/casys-kaist/NeuPIMs) publishes separate `main`, `npu-only`, and `npu+pim` branches. The [official ASPLOS program](https://www.asplos-conference.org/asplos2024/main-program/abstracts/index.html) confirms publication. The [paper](https://arxiv.org/abs/2403.00579) describes request-length sampling from ShareGPT/Alpaca, dual row buffers, sub-batch interleaving, and channel load balancing.

This audit is recorded in `runs/neupims-validation/source-audit.json`. Execution status and raw logs are retained alongside it; a source audit alone is not a successful simulation or paper-figure reproduction.

## Source and build

- NeuPIMs main: `f299af3fc8f20077e816f2e48313294cddb4bd7c`.
- NPU-only: `57617ee0d8abe8aa87a3551d42ee98f999fecd71`.
- NPU+PIM: `e51a4af57b5ca07df98433e250349ccff16f4ea1`.
- Booksim submodule: `f0245986e9a671fc72cb76f62b366a947f33d655`.
- FlameGraph submodule: `cd9ee4c4449775a2f867acf31c84b7fe4b132ad5`.

All original tracked source is retained unchanged. Dependencies live under `external/neupims/.conan-local` and the Python build tool under `.venv-neupims`. The original Conan 1.57 installer fails under Python 3.12 while building its PyYAML dependency; Conan 1.66 is used as a build frontend with the original C++ dependency versions (Boost 1.79.0, robin-hood-hashing 3.11.5, spdlog 1.11.0, nlohmann_json 3.11.2). GCC 13 needs explicit forced includes of `cstdint` and `limits` for missing direct upstream header includes. The original source forces `-O0`; an optimized build adds `-O3` after that flag while retaining assertions. These are build portability changes, not simulator algorithm changes. Build logs and the generated Conan dependency description record the resolved transitive dependencies.

## Effective workload and measurement scope

The released GPT3-7B shape configuration uses embedding 4096, 32 heads, tensor parallelism 4, and one modeled layer. `src/client/Client.cc` forces one generated token per request. The main staged execution path builds QKV generation, multi-head attention, projection, and FFN timing for layer zero. It does not load pretrained model tensors, tokenize prompts, produce tokens, or evaluate perplexity/task accuracy. An application result must therefore say **trace-driven single-layer decode timing**, not full GPT-3 inference.

The provided CLB example contains all 512 requests. `src/scheduler/Scheduler.cc` hardcodes the maximum batch and active request counts to 1024, ignoring the JSON values of 128/130. A wrapper must inspect effective source behavior rather than infer batch size solely from the JSON filename or fields.

## Default baselines are not automatically comparable

The author example commands vary several factors simultaneously:

| Factor | NeuPIMs main | NPU-only | NPU+PIM |
|---|---|---|---|
| Source branch | main | npu-only | npu+pim |
| Memory backend | dual-row-buffer PIM | DRAM | single-row-buffer PIM |
| JSON HBM_size | 32 | 8 | 32 |
| JSON maximum batch | 128 | 512 | 128 |
| Example request assignment | CLB | RR | RR |

Most significantly, the similarly named CLB and RR `*-0.csv` files contain different sequence-length multisets. CLB has 153,548 total tokens and a minimum length of 1; RR has 151,967 total tokens and a minimum length of 5. Both have 512 requests and maximum length 2048. Thus the filenames do not establish identical semantic workloads.

For controlled application validation, request lengths must be a semantic input asset. Channel assignments must be a separate deployment/method asset, derived from those same request lengths. Every variant must retain the actual effective model/scheduler settings, source revision, compilation settings, and metric boundary. Different source branches and hardware profiles must not be compared by dropping existing provenance or cost-contract checks. A successful wrapper replay establishes integration fidelity, not calibrated physical performance.

## Additional primary-source candidates

- [CENT, ASPLOS 2025](https://github.com/Yufeng98/CENT), [paper](https://arxiv.org/abs/2502.07578): CXL-connected GDDR6-PIM and near-memory processing for Llama2. It expands topology and communication coverage beyond the current HBM model. The author estimates 12–24 hours and roughly 100 GB for the full token-by-token sweep; even its sampled quick suite takes hours. Candidate only unless a corresponding execution report is present.
- [AttAcc, ASPLOS 2024](https://github.com/scale-snu/attacc_simulator): useful for reproducing the predecessor's baseline and system model. STARC already builds on this lineage, so it is not counted as an independent simulator implementation.
- [IANUS, ASPLOS 2024](https://arxiv.org/abs/2410.15008): a useful specification for NPU/PIM shared-memory contention and end-to-end transformer scheduling. This audit did not establish an author-published executable artifact; it is not counted as runnable or validated.

## Executed results

The original author CLB file and the original GPT3-7B shape configuration completed on the main branch. The NPU-only branch was then executed with exactly the same ordered request lengths and trace, instead of its different default RR trace. Both paths were rerun through `neupims-artifact/v1` with separate runtime descriptors. The adapter checks the semantic length sequence against the channel-assignment CSV before launching a subprocess, so a similarly named file with different lengths cannot silently substitute a different task.

| Variant | DRAM cycles at 1 GHz | Native modeled latency | Scope |
|---|---:|---:|---|
| NeuPIMs, author CLB assignment | 619,233 | 619,233 ns | 512-request, one-layer, one-token decode batch including simulator drain |
| NeuPIMs, regenerated RR assignment | 650,229 | 650,229 ns | Identical task and platform; only channel assignment changes |
| NPU-only, matched request lengths | 5,937,045 | 5,937,045 ns | Same task; different original source branch and platform configuration |

The two original direct runs and their application executions agree exactly on DRAM counters (and on all reported main-branch stage counters). For main CLB, stage cycles sum to 619,231 while the core's counter is 619,227; neither is silently substituted for the DRAM total of 619,233. The exported latency explicitly uses the 1 GHz DRAM clock and the total DRAM counter, including drain. Wall-clock runtime of the CPU simulation is not a target-system latency measurement.

The regenerated RR trace preserves the ordered request lengths exactly and assigns request `i` to channel `i % 32`. It is a controlled placement ablation, not the author's different default RR workload. CLB/RR is 0.952330640435908 for this one modeled batch (4.7669% lower latency with CLB). The framework accepts this **conditional modeled comparison**. Task quality remains unmeasured and physical-hardware performance remains unvalidated.

The NPU-only comparison is rejected by the common comparison contract because its platform and source-bound cost contract differ. Its timing is retained as an independently executed original baseline, not promoted to an automatically fair speedup. Reproducing the paper's reported whole-system throughput ratios requires an explicit cross-platform experiment protocol, audited effective settings and the complete paper sweep.

The first original `-O0` 512-request attempt timed out after 180 seconds. The optimized full workload completed. A separate eight-request prefix test timed out after 120 seconds while executing the second FFN linear operation; its terminal logs are preserved. This is an observed unsupported/time-limited case, not evidence that all smaller batches work or proof of a specific upstream bug. No result from either timed-out attempt is included in the table.

All three application variants and their three frozen-engine replays passed, and all six sealed bundles verified. A negative control using the author's original RR trace was rejected before execution because its semantic request lengths differ. Application manifests, source/runtime hashes, native logs, and frozen replay results are under `runs/neupims-application-v07`. Reproduction command after building the pinned runtime:

```sh
python3 scripts/evaluate_neupims_application.py --out runs/neupims-application-repeat
```

This is an application-adapter and controlled-ablation validation of the released simulator scope. It does not reproduce the complete ASPLOS paper, port its transformer into the common typed IR, measure model quality, or independently calibrate its hardware timing model.

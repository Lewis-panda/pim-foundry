# CIMFlow application artifact validation

The released ResNet-18 ONNX graph is compiled through the original CIMFlow graph
partitioner and LLVM/MLIR operator compiler, then executed by the original
SystemC simulator. The experiment compares the authors' `dp` and `baseline1`
mapping strategies under their supplied 64-core, T8/K16, 8-byte NoC configuration,
with batch size 8. It extends the existing Foundry SRAM kernel tests to native
multi-core convolution, residual operations, communication, and synchronization.

The application scope is explicitly **the artifact's compiled CNN backbone**.
The released source excludes the classifier/FC block and has pooling generation
commented out. Its cost simulation therefore cannot establish complete ONNX
semantics, classification accuracy, or an end-to-end application speedup. This
is an experimentally audited limitation, not a hypothetical warning.

## Primary sources and pinned artifacts

- [CIMFlow paper, DAC 2025](https://arxiv.org/abs/2505.01107),
  [published DOI](https://doi.org/10.1109/DAC63849.2025.11133270).
- [Author framework and tutorial](https://github.com/BUAA-CI-LAB/CIMFlow), commit
  `2ddf0c300387de53a6ebe062b57e9e3236060916`.
- [Author compiler](https://github.com/BUAA-CI-LAB/CIMFlow-Compiler), commit
  `67f3da074a6dec76b26046001ebe7c380a2bf9a4`.
- [Author simulator](https://github.com/BUAA-CI-LAB/CIMFlow-Simulator), commit
  `531abcb869b1b3af1f95cf923223f09e08725d96`.
- [Released example models](https://github.com/BUAA-CI-LAB/CIMFlow-Compiler/releases/tag/v0.1.0):
  archive SHA-256 `8eb712aa57dc45d3355b18c9e5e6ebc3bb77f4cbb7ed218baaa2c687de161c4e`.

The compiler binary and generated ANTLR classes were extracted from the authors'
public `ghcr.io/buaa-ci-lab/cimflow-tutorial` container layer, verified against
registry SHA-256 `0e935a68451b845ce5d210e4c1af3f5a0ee995e7a21e8064937925a98ea3aa42`.
The binary SHA-256 is
`40482c362b18a9b0c0212dd1b022381252919cd11bc9365c614daa8aefd7a173`.
The compiler binary was not rebuilt locally; its provenance is the authors'
published image, not a demonstrated reproducible compiler build. No privileged
container or global package installation was used. The existing simulator was
built locally from the original pinned source. All Python packages, relevant
source files, native binaries, ANTLR classes, shared libraries, architecture
configuration files, and NoC cost tables are bound in the runtime descriptor.

## What the released network actually exercises

The author ONNX assets both use `[1, 3, 32, 32]` inputs and 100 output classes.
ResNet-18 includes 20 Conv nodes; MobileNetV2 includes 53 Conv nodes, including
depthwise convolutions. The initializers are floating-point ONNX tensors and
the graphs express quantization operations explicitly. The compiler models
INT8 architecture operations. No image dataset or weight quantization accuracy
evaluation is performed by this timing run.

The original graph preprocessing function also forces a 32×32 input shape and
caches the result by model basename. The adapter copies only bound source files
to a new temporary compiler workspace for every execution, so an earlier model
with the same basename cannot supply a stale simplified graph. It preserves
the resulting preprocessed ONNX graph in the output bundle.

`src/cim_compiler/cg/utils/instructions_gen.py` emits convolution, depthwise
convolution, residual Add, and data movement. Its MaxPool and GlobalAveragePool
branches are commented out, and the remaining operators are skipped. The CG
mapper identifies the FC boundary and compiles the convolution backbone before
that boundary. Accordingly, the adapter rejects a whole-model scope and any
request to treat native timing as numerical or task-quality evidence.

For ResNet-18 DP at batch 8, the generated graph-level instruction counts are
600 `conv`, 248 `add`, 2,152 `send`, 2,152 `receive`, 298 `read`, 216 `write`, and
72 `wait_write`. These are compiled instruction instances across the mapped
cores and batches, not counts of original ONNX nodes.

The report's `op_count_` is copied from compiler model metadata, not a dynamic
count of executed arithmetic. The metadata counter includes Gemm/MatMul even
though the CG generator excludes the FC block. The adapter checks that this
metadata survives compilation and simulation, but deliberately exposes only
latency and energy as comparable metrics; it does not qualify the reported
TOPS or TOPS/W as backbone throughput.

## Protocol

The application manifest binds the model bytes, batch size, 32×32 shape,
independent cold batches, and the backbone scope. Deployment and method remain
separate: `cimflow-digital-sram/v1` selects the audited architecture and the
variant selects `dp`, `baseline1`, or `baseline2`. Neither names arbitrary
physical hardware nor embeds a device placement in the application.

The adapter runs original `cim-compiler network`, retains CG IR and native ISA,
then runs `cim-simulator`. The native report's `latency_` is retained in **ms**;
`total_energy_` is retained in **pJ**. Both cover one compiled fixed batch.
Host compilation/simulation wall time is logged separately and never used as
device latency. The modeled global-memory and static/dynamic component costs
are exactly those supplied by the author configurations; this run does not
calibrate them against a physical chip.

The validation script compares the framework's full native report with an
independent direct execution, excluding only `exec_time_` (host wall time), and
requires identical native ISA bytes. Frozen application-bundle replays rerun
compilation and native simulation, with fresh caches, then compare modeled
metrics and identities. Both numerical quality and task quality remain
`not_evaluated`; a conditional backbone model ratio never becomes a
quality-preserving or hardware speedup.

Run the study after obtaining the bound author assets and runtime:

```sh
python3 scripts/evaluate_cimflow_application.py --out runs/cimflow-applications-new --replay
```

The first independent direct runs produced 3.560054 ms for `dp` and 4.874690 ms
for `baseline1`, with 17,082,780,117.398392 pJ and 22,679,911,357.88909 pJ,
respectively. These are original-artifact backbone model observations, not a
reproduction of the paper's full Figure 5 suite or a measured chip result.

## Final validation findings

The overall validation status is **`reproducibility_failed`**, with the following
case-level evidence. No tolerance was relaxed to turn a mismatch into a pass.

| Released backbone / method | Original direct latency | Framework latency | Frozen replay | Validation |
| --- | ---: | ---: | ---: | --- |
| ResNet-18 / DP | 3.560054 ms | 3.560054 ms | 3.560054 ms | Observed direct parity and replay pass |
| ResNet-18 / baseline1 | 4.874690 ms | 4.875508 ms | 4.874690 ms | Reproducibility failed |
| MobileNetV2 / DP | 17.443438 ms | 17.443910 ms | Not run | Reproducibility failed |

All three cases generated byte-identical ISA between independent and framework
compilation. The baseline1 frozen replay preserved task, execution, platform,
cost-contract, and execution-identity hashes, but changed latency and energy.
That is sufficient evidence of a frozen-application replay failure. It does not
prove stochastic behavior under an identical process command and environment:
the fresh temporary paths and execution contexts differ.

Three additional baseline1 native runs all returned 4.874690 ms, including one
that consumed the framework's identical ISA file path. Across all six baseline1
observations, the range was 4.874690–4.875508 ms (spread 0.000818 ms, approximately
0.0168% of the minimum). MobileNetV2's two observations differ by 0.000472 ms
(approximately 0.00271%). These are observed ranges, not statistical confidence
intervals or a quantified hardware-model error bound.

One plausible source mechanism is `Clock::pos_edge_events_`, declared as
`std::set<sc_event*>` in `src/base_component/clock.h`: `Clock::process()` notifies
same-cycle events in pointer-address order. This can make arbitration depend on
heap layout, but a controlled source patch has **not** established causality.
The original simulator is preserved unmodified. The recorded evidence supports
execution-context sensitivity; its exact cause remains unresolved.

The study withholds baseline/DP ratio qualification after this failure. An
individual application bundle's `completed` status means native execution
finished; the study's separate `validation_status` determines whether the
original-artifact parity and replay checks passed. DP's observed pass does not
establish simulator-wide determinism.

The final five sealed framework bundles and study reports are under
`runs/cimflow-applications-v07-validated` (ResNet-18 and both replays) and
`runs/cimflow-mobilenet-v07`. Their summaries retain per-case outcomes and carry
top-level `validation_status: reproducibility_failed`. The evidence includes
11 native simulations: three independent application runs, five framework runs,
and three additional baseline1 checks.

The raw acquisition records, source inventory, model inventory, independent
commands, logs, instructions, and reports are under
`runs/cimflow-application-validation`. Final framework results are recorded by
the evaluation script in its chosen output directory.

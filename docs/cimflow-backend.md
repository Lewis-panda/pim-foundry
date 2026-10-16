# SRAM backend: native CIMFlow timing plus checked instruction lowering

For a `3 × 31` input and `31 × 17` weight matrix, the backend pads the K and N
tails, places signed INT8 values in a binary SRAM image, emits native weight
copies and CIM instructions, and emits INT32 vector additions for K-tile
reduction. An independent instruction interpreter reads those emitted bytes
and instructions; its output must equal the scalar matrix product before the
native timing result is accepted. This catches transpose, padding, signedness,
and reduction errors in lowering.

The public SystemC simulator is built and executed. The pinned revision is
[`531abcb869b1b3af1f95cf923223f09e08725d96`](https://github.com/BUAA-CI-LAB/CIMFlow-Simulator/tree/531abcb869b1b3af1f95cf923223f09e08725d96),
from the [CIMFlow project](https://arxiv.org/abs/2505.01107), associated with
DAC 2025. This is the current public simulator revision tested here, not a
claim that its commit is the original DAC evaluation artifact.

## Executable capability

- Shared operation: signed INT8 GEMV or GEMM, weights `[K,N]`, exact INT32
  accumulation/output; `1 <= M <= 16`, `1 <= K,N <= 128`.
- Geometry: tile K and N independently in `{8,16,32,64}`. Any tensor tail is
  explicitly padded with zeros. Each macro stores a row-major tile of signed
  INT8 words under the declared Foundry functional model.
- Native instructions: general/special register loads, `MEM_CPY`, `CIM_CFG`,
  `CIM_MVM`, `CIM_OUT`, and a configured INT32 SIMD-add instruction. Weight
  programming, input reads, partial output stores, and K-tile sums all appear
  in the native program. M rows execute sequentially and reload weight tiles.
- Every artifact retains operands, memory image, instructions, configuration,
  expected output, hashes, native stdout/stderr, native report, binary/library
  identity, and an execution report with explicit evidence categories.
- Existing artifact directories and execution reports are never overwritten.
  Native execution requires an unmodified checkout at the pinned revision.

## What the native model computes

The upstream model tracks CIM timing/power but does not produce CIM result
values: [`CimUnit::accessAndGetDelay`](https://github.com/BUAA-CI-LAB/CIMFlow-Simulator/blob/531abcb869b1b3af1f95cf923223f09e08725d96/src/core/cim_unit/cim_unit.cpp)
charges SRAM accesses without storing weight values, while
[`CimControlUnit::processOnlyOutput`](https://github.com/BUAA-CI-LAB/CIMFlow-Simulator/blob/531abcb869b1b3af1f95cf923223f09e08725d96/src/core/execute_unit/cim_control_unit.cpp)
issues a memory write with an empty data vector. Its `real_data` mode reads
activation bytes to determine activity; that name does not imply matrix
arithmetic is implemented.

Therefore evidence is labeled `paired_instruction_functional_and_native_timing`:
`lowering_verified=true`, `native_numerical=false`, and
`hardware_calibrated=false`. The instruction interpreter supplies a documented
dense signed-INT8 arithmetic contract that the upstream timing model lacks.
It does not call the matrix reference to generate outputs, and a test changes
weight bytes to demonstrate that its result follows the emitted memory image.
The independent matrix product is used only for comparison.

The supplied profile uses explicit synthetic timing/power coefficients and
one bit-serial SRAM macro. It is not a calibrated paper-chip profile. Native
timing scope is `core_local_prefilled_including_cim_programming_and_k_reduction`:
host-to-staging transfers and final transfers to the host are excluded and
listed in each result. A heterogeneous scheduler must account for those
transfers separately. Comparing these numbers directly with HBM-PIM or GPU
end-to-end latency would violate the scope and calibration contracts.

## Reproduce

From the repository directory:

```sh
python3 scripts/build_cimflow.py --record runs/cimflow-build-new
PIM_FOUNDRY_CIMFLOW_SIMULATOR="$PWD/external/cimflow-simulator/build/cim-simulator" \
  python3 -m unittest tests.test_cimflow -v
python3 runs/cimflow-study/reproduce.py --output runs/cimflow-study-new
```

The build helper clones/checks out the pinned revision if necessary, verifies
submodule commits, builds with at most four jobs, and records compiler,
configuration, source, binary, and dynamic-library identities. It performs no
system installation. The study takes an unused destination directory.

Python API:

```python
from pathlib import Path
from pim_foundry.cimflow import compile_gemv, compile_matmul, execute, interpret

artifact = Path("runs/my-sram-case")
compile_gemv([1, -2, 3], [[1, 2], [3, -4], [5, 6]], artifact)
assert interpret(artifact)["output"] == [[10, 28]]
report = execute(artifact, Path("external/cimflow-simulator/build/cim-simulator"))
assert report["lowering_verified"] and not report["native_numerical"]
```

## Recorded validation

`runs/cimflow-build-final/provenance.json` records the clean build.
`runs/cimflow-study/results/summary.json` records six distinct shapes,
two tile sizes, two deterministic repetitions, 24 numerical lowering checks,
and 48 actual native executions. Each wrapper run is compared against a
second direct invocation of the same binary, config, image, and ISA. All
native report fields except host execution time must match exactly.

| M,K,N | Tile K,N | Native time, ns | Native energy, pJ |
|---|---|---:|---:|
| 1,1,1 | 8,8 | 47 | 22.718 |
| 1,1,1 | 16,16 | 62 | 63.272 |

These two example rows show measured simulator outputs for the synthetic
profile; they are not hardware measurements or benchmark speedup claims.
The full study retains larger and ragged cases with both profiles.

The original upstream `CoreTest` fixture 6 also passes its own timing/energy
assertions: **235 ns and 280400 pJ**. Raw output is retained under
`runs/cimflow-upstream-fixture-*`. To repeat its original check:

```sh
cd external/cimflow-simulator/build
./core-test ../config/test/cim_compute_unit_test_config_base.json \
  ../config/profiler_config.json ../test/data/core/core_test_data_6.json \
  /tmp/cimflow-upstream-core-report.txt
```

The twelve adapter tests cover signed extremes, INT32 accumulation, one-hot
layout tracing, all supported tile sizes, ragged K/N, multiple M rows, zero
inputs, memory corruption, illegal instructions, bounds, artifact integrity,
and native execution. Native execution is opt-in via the environment variable
above; the other eleven tests do not require a simulator installation.

# STARC application artifact audit

The author application constructs the transformer prefill and every decoding step,
runs attention command traces in Ramulator, and combines those costs with its GPU
and communication models. This exercises an application workload with persistent
KV growth, rather than substituting a matrix multiply for the application.

## Publication and immutable inputs

- [STARC, ASPLOS 2026, DOI 10.1145/3779212.3790226](https://doi.org/10.1145/3779212.3790226), published 22 March 2026. ACM lists Artifacts Available and Artifacts Evaluated & Functional badges. These badges do not certify our run or every model assumption.
- [Official conference program](https://www.asplos-conference.org/asplos2026/program/index.html), Session 3A, LLM Attention & KV Cache.
- [Author artifact](https://github.com/EPIC-RPI/STARC), pinned commit `63c8aaa7a02114cd4cf21f32e7621fab5b416957` (11 January 2026).
- The repository gitlink names Ramulator commit `e442c64b2c0db7afd9d23173925d636ea2895a36`, but the author's setup explicitly resets it to `b7c70275f04126c647edb989270cc429776955d1`. Our build follows the setup's explicit commit and applies its supplied patches.
- Build dependencies: yaml-cpp `0579ae3d976091d7d664aa9d2527e0d0cff25763`, spdlog `ad0e89cbfb4d0c1ce4d097e134eb7be67baebb36`, argparse `997da9255618311d1fcb0135ce86022729d1f1cb`. GCC 13.3.0, CMake 3.28.3, C++20. No CUDA execution is needed for this system simulator.
- Python 3.12 in a task-local environment, NumPy 2.5.4, pandas 3.0.6, python-dateutil 2.9.0.post0, six 1.17.0. This differs from the author's recommended Python 3.10; the exact runtime is recorded.

The original source checkout is `external/starc/`. Build logs, executable hashes,
runtime manifests, and native-run results are in `runs/starc-build/`.

## Native application invocation

From an isolated copy of `simulator_starc/main.py`, `src/`, the selected budget
table, and a private `ramulator2/` directory:

```sh
python main.py --system dgx-attacc --gpu H100 --ngpu 8 \
  --model Mistral-7B --lin 2048 --lout 128 --batch 16 --pim bank \
  --powerlimit --ffopt --pipeopt
```

For STARC add:

```sh
--sparsity --kv_budget_table kv_budget_Mistral_STARC.txt
```

The paper artifact README specifies `--lout 32000`. The 128-token runs are an
explicitly shortened decoding window of that configuration, using the same model
dimensions, hardware parameters, input length, batch, and budget-table prefix.
They are application integration experiments, not reproduction of a paper figure.
The simulator creates `lout - 1` decoding steps; the first output is accounted for
by prefill. A table with 32,000 budget values is not a cache of simulator results.

Each run must begin without a `ramulator.out` cache. Its key omits the attention
method, KV budget, clustering flag, and source version. Reusing it across methods
can silently return another experiment's results. Full attention does not enter
the sparse clustering branch, so Full versus STARC requires no method patch.
SparQ and Quest require the author-documented disabling of `--add_cluster`; those
methods have not been executed in this audit.

The author launcher invokes bare `python` for trace generation; the task-local
environment must be first in `PATH`. The author deletes intermediate traces and
YAML, retaining only summary command counts in `ramulator.out`. A framework run
must intercept/preserve native subprocess evidence if it needs trace-level audit.

## Observed original-source application outputs

These numbers retain the original native controller, which has an ASan-reproduced
lifetime defect. They are diagnostic artifact outputs, not validated
physical performance estimates.

| Generated-token argument | Decode steps | Method | Mean decode batch-token latency (ms) | Mean decode batch-token energy (nJ) |
| --- | ---: | --- | ---: | ---: |
| 4 | 3 | Full | 0.9402037822136982 | 562666432.75776 |
| 4 | 3 | STARC | 1.061460655913731 | 571125708.2265599 |
| 128 | 127 | Full | 0.9405744286683656 | 564791586.345942 |
| 128 | 127 | STARC | 0.9385560226692454 | 549221514.9294 |

Original 128-token wall times were 62.16 s for Full and 52.30 s for STARC. These
host execution times are not modeled accelerator latency. All four runs used
fresh work directories, the same executable, and the same environment. Raw CSV,
cache, stdout and stderr are in `runs/starc-full-smoke/`,
`runs/starc-starc-smoke/`, `runs/starc-full-128/`, and `runs/starc-starc-128/`.

The clustering operation is charged only at the final decoding step. Shortening
the window changes its amortization and can reverse the apparent benefit. The
observed three-step versus 127-step change is therefore not a paper speedup claim.

## Metric semantics and limitations

`g_time (ms)` and `g_energy (nJ)` are the author model's averages across
`lout - 1` decoding steps, scaled over all transformer layers, for the entire
batch. They are not per-request end-to-end latency. `s_time` is the separately
modeled prefill time in ms. A modeled batch prefill-plus-decode total can be
derived as `s_time + (lout - 1) * g_time`; this still omits application components
outside the author's transformer model. Do not label that value measured service
latency or hardware throughput.

`kv_budget_Mistral_STARC.txt` supplies author-derived row-access budgets, used
at 16-token row granularity. The run does not load model weights, generate token
values, score LongBench/RULER, or remeasure task quality. Quality remains an
external, unresolved comparison gate.

Two source findings materially constrain interpretation:

1. `hbm3_pim_controller.cpp` uses the first request iterator after removing its
   list element, and also uses it when no first request was found. ASan reproduced
   a heap-use-after-free at line 151. A separate safety variant captures the
   command before removal, guards an absent first request, uses the secondary
   request's own buffer, and clears per-selection PIM state. It preserves the
   author's timing presets, completion semantics, and application model.
2. `config.py` declares `Mistral-7B` with `gqa_size=4`, but `Transformer` never
   consumes that field. Its QKV dimensions use three full hidden dimensions,
   and the CSV hardcodes `gqa_size=0`. This is an execution of the named author
   artifact configuration, not a newly validated model of Mistral's GQA topology.

The repository contains an example CSV row but does not identify its method
unambiguously or provide machine-readable figure reference values. Matching the
adapter's result against an isolated direct run checks integration fidelity;
matching a paper claim requires an identified figure, complete settings and
quality evidence. Those are distinct gates.

## Safety variant and direct rerun

`runs/starc-build/controller-safety.patch` records the complete local patch. The
original source checkout is unchanged by that patch. `safe-source/` contains the
variant, and `release-safe/` contains its executable and shared library. ASan and
UBSan both completed a Full attention trace at length 2049 and a clustered sparse
attention trace at length 2175 without diagnostics; their entire stdout matches
the release build byte for byte. This is bounded sanitizer evidence, not an
exhaustive verification. See `sanitizer-validation.json` and the preserved raw logs.

Both 128-token applications were independently rerun with the safety variant.
For each method, both `output.csv` and all 127 rows of `ramulator.out` are byte
identical to the original release results. This checks that the observed defect
fix did not change these measured artifact outputs; it cannot prove equality for
unexecuted workloads. See `original-safe-parity.json`.

Re-run the author application directly, without importing PIM Foundry:

```sh
python3 runs/starc-build/run_native.py \
  --runtime runs/starc-build/runtime-safe.json \
  --output runs/starc-new-full-128 --method full --lout 128
python3 runs/starc-build/run_native.py \
  --runtime runs/starc-build/runtime-safe.json \
  --output runs/starc-new-starc-128 --method starc --lout 128
```

The runner verifies source and executable hashes, requires a new output directory,
copies the native library and executable into that directory, starts without a
simulator cache, and records the exact command plus stdout, stderr, and CSV. It
checks integration independently of the framework adapter. Runtime descriptors
are `runtime-original.json` and `runtime-safe.json`; the former explicitly records
the sanitizer failure. `source-evidence.json` records native source files, author
setup patches, build versions, local patch identity, and evidence hashes.

## Exact memoization feasibility

A bounded probe generated Full attention traces at lengths 2049, 2050, 2064,
2065 and 32000. The first three were byte identical; length 2065 changed the
trace. A native length-32000 trace (2.98 MB) took 3.825 s on this host. Therefore
content-addressed reuse can remove some repeated simulations, but a complete
32000-output-token application still requires generating every input trace and
running many distinct traces. The probe does not execute that application.
`runs/starc-build/memo-probe/results.json` records hashes and timings. A reusable
cache must bind the trace contents, configuration, executable, and method state;
the author's length-only selection fields are insufficient across methods.

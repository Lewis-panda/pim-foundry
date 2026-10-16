# DCC Foundry fork: request lifetime and explicit completion model

The original controller erases the request's list node and then reads its command to decide whether a second command can issue. This fork saves the command before removal, returns when no first request exists, and uses the second request's own buffer when removing it. The original source and previous experiment results remain intact.

Base: DCC revision `7ffb80e4f33e357177dc8ab148953c0484e1e322`, only `src/Ramulator2/HBMPIM/`. The checked-in patch and source manifest bind every original and resulting source file. This is a distinct simulator profile, `dcc-foundry-explicit-completion-v1`, not an unmodified reproduction of the paper's simulator.

The original subtree's MIT notice is included in [UPSTREAM_LICENSE](UPSTREAM_LICENSE).
See [the scoped third-party notices](../../THIRD_PARTY_NOTICES.md). Author
sources, dependencies and local `runs/` artifacts are not bundled; obtain them
separately before using the advanced build commands below.

## Completion endpoint

An issued operation now stays in an ordered completion queue until its declared endpoint. The termination test and a barrier both wait for this queue. A barrier addresses **one channel**; a whole-memory fence requires a barrier for every used channel.

| Operation | Completion delay after final command issue | Basis |
| --- | --- | --- |
| Read | `nCL + nBL` | Existing source read completion |
| Write | `nCWL + nBL + nWR` | Existing source write-to-precharge constraint; explicitly a **write recovery** endpoint |
| Refresh | `nRFC` / `nRFCSB` | Existing source refresh recovery constraints |
| PIM | `Controller.pim_completion_cycles[DRAM_COMMAND_NAME]` | Required caller-supplied model assumption |

PIM mappings accept integers from 1 to 1,000,000 cycles. Missing mappings fail instead of silently declaring a PIM operation complete at issue. Values are completion delays, not issue intervals. The example build assigns 8 cycles to every PIM command as an explicitly synthetic assumption, **not** as a paper result or hardware measurement. Change the values and retain the full configuration when studying sensitivity.

Callbacks and completion bookkeeping do not impose an extra one-completion-per-cycle port. Completions are ordered by their actual departure cycle; every ready completion drains. Read forwarding keeps its existing one-cycle endpoint. Refresh/maintenance buffers are included in barrier readiness.

## Timing boundary

The patch deliberately preserves the source command issue constraints. The source lacks explicit issue constraints for some PIM commands, including WB, RESET, and RELU. **Completion tracking alone does not resolve PIM dependency hazards between commands.** A conservative timing projection must insert channel barriers at dependent operation boundaries, or after every PIM operation. Such a projection is a separately declared model; it is not the author's original schedule.

This simulator operates on addresses and commands. It does not execute FP16 arithmetic or establish that DCC's original layout generator computes the intended tensor. Numerical validation must come from a separate functional executor and be bound to the projected trace. No runtime descriptor produced here authorizes a hardware-calibrated performance claim.

## Rebuild and regressions

No package or system installation is performed. Dependencies are the previously fetched clean pinned revisions in `runs/paper-dcc-build/deps`. Build from the project root into a **new** output directory:

```sh
python3 scripts/build_dcc_patched.py \
  --output runs/dcc-patched-final \
  --pim-completion-cycles 8 \
  --paper-trace runs/dcc-isca26-study/b1-h32-k128-n512/candidate-837/compiled/kernel.trace
```

The builder creates an isolated copy if needed, verifies 133 source hashes, builds Release and ASan/UBSan with at most four jobs, and saves commands, compiler identity, dependency revisions, raw logs, binary/library hashes, patch hash, and the exact configuration. Libraries stay in their respective build directories so one build cannot overwrite another.

`runtime.json` contains `runtimes.release` and `runtimes.asan`; each records `executable`, `library`, and both SHA-256 hashes. Its `config_template` is a full YAML string with the relative trace path `kernel.trace`. Its `model` records the explicit assumptions and unresolved source timing constraints.

The native regression suite executes empty/idle traces, reads, writes, barriers, mixed row/column requests, a PIM tail, PIM barriers, and the optional actual paper-shape trace. It executes each twice under Release and twice under ASan/UBSan, requires successful completion and identical cycles, and retains raw evidence in `regressions/`. Address and undefined-behavior checks are enabled; leak detection is disabled because the upstream program does not own and free all allocated simulator objects at exit. This is not a claim that every possible trace is free of defects.

`tests/test_dcc_patch.py` also checks that increasing source write recovery by 10 cycles increases a one-write trace by exactly 10; increasing PIM completion by 20 changes one final PIM operation by 20 and two operations separated by a fence by 40. A trace spanning a refresh checks that its final refresh recovery is drained. Missing PIM completion assumptions must fail. Native tests skip when the optional build is absent.

Set `PIM_FOUNDRY_DCC_RUNTIME` to a different build's `runtime.json` when running these tests against a separately rebuilt artifact.

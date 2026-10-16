> Paths under `runs/` or `external/` refer to locally generated or separately obtained artifacts, not bundled files. Start with [the public quickstart](getting-started.md); see [validation scope](../VALIDATION.md) for historical observations.

# Explicit comparison contracts

If one run uses twice the memory capacity, dividing its latency by another run's latency cannot isolate the method's benefit. The comparison protocol records that capacity, fixes a common resource budget when requested, and binds every ratio to the exact workload, metric boundary, quality observation and repetition evidence.

The public API is:

```python
from pim_foundry.comparison import compare, inspect, replay_evidence, evaluate_calibration

report = compare(left_bundle, right_bundle, protocol_path, evidence_paths=[...])
```

`inspect(protocol_path)` validates the protocol and referenced resource files without running a simulator. `compare` first verifies every original and repeated application bundle. Existing `application.compare` is unchanged.

## Tracks and admission rules

| Track | Permitted difference | Required admission condition |
|---|---|---|
| `same_platform` | Mapping or implementation | Same platform identity, same resource values and same cost contract |
| `equal_budget` | Different architectures and device combinations | At least one declared common resource equality or upper bound; every fixed resource axis matches |
| `whole_system` | Different disclosed configurations | Identical semantic workload and metric boundaries; results describe the configured systems together with their resource differences |

Every track preserves the application task identity. That identity contains the model, concrete inputs, preprocessing, scenario, quality policy and semantic parameters. The execution mode must match. Metric **unit, scope and aggregation** must match; changing to `whole_system` does not relax these rules. A failed application or failed quality observation blocks comparison.

A different cost contract is permitted only for `equal_budget` and `whole_system`, with two content-bound `pim-foundry.cost-model-binding.v1` files. Each contains the exact cost contract and metric descriptor from its result, a rationale, and identical nonempty inclusion/exclusion accounting sets. Different metric origins also require valid bindings. These are explicit, inspectable equivalence declarations; source hashes do not prove that a model actually charges every declared cost correctly.

`whole_system` identifies a comparison track, not a promise that host work, loading, preprocessing or compilation were measured. The original metric scope and cost exclusions remain visible. No track establishes a causal isolated scheduling effect automatically.

## Resource disclosure

The protocol pins left/right `pim-foundry.resource-disclosure.v1` files. Each binds its platform hash and reports CPU/GPU/NPU/PIM counts, capacity, compute, external bandwidth, internal bandwidth, power and cost. Units are explicit: `count`, `byte`, `op/s`, `byte/s`, `W`, `USD`. Counts are nonnegative integers. Compute conventions and cost basis are explicit text.

Each quantity distinguishes `model_assumption`, `author_reported`, `observed`, `externally_measured` and `undisclosed` evidence. Unknown capacity, peaks, bandwidth, power or prices use `value: null` and `basis: undisclosed`; unknown values never satisfy a budget. Source files are SHA-256 bound. A numeric quantity is still a supplied declaration, not independently authenticated hardware evidence. Device/core packaging and operation-count conventions must be reviewed before interpreting count or throughput equality.

For shared graph/tensor platforms with an explicit device inventory, disclosed counts and capacity are also checked against the sealed experiment configuration. Rehashing a false resource declaration cannot override that executed configuration. Paper-specific manifests still require review of their architecture-specific resource interpretation.

Budget constraints select a resource and `equal` or `at_most`, one common bound, and an explicit relative tolerance no larger than 10%. `at_most` is a common envelope; it does not imply both systems consume the entire allowance. `fixed_axes` requires exact equality for selected resources. Constraining compute or cost also requires identical conventions. A bandwidth sum alone does not establish an equal bottleneck or contention model.

See the executed equal-memory protocol (local artifact: `../runs/comparison-protocol-v1-delivery/equal_memory_budget/protocol.json`) for a complete example. It fixes total declared modeled capacity to 33,554,432 bytes while allowing a GPU-to-NPU change; unknown power, cost and total compute remain unknown.

## Evidence, failures and qualification

The protocol pins each evidence attachment's SHA-256. Attachments bind exact result-file hashes, a scope, source-file hashes, one of `author_reported`, `observed`, `externally_measured`, and a status. Sources remain available externally; the report records every consumed file's digest. Preserve source files with the report for independent verification.

`replay_evidence(original_bundle, replay_bundle, scope)` verifies both sealed bundles and computes all replay checks itself. During comparison these checks are recomputed, so editing an attachment's `passed` flag cannot turn a failed replay into a successful one. Reusing the same run directory as its own replay is rejected.

`comparable` permits a descriptive modeled ratio under the track's admission conditions. `qualified` additionally requires:

- Numerical quality passed, plus task quality when `quality_requirement: task` is selected.
- Observed correctness and replay evidence for each exact result and metric scope.
- Repeated run samples under the declared deterministic or statistical method.
- For original-paper adapters, observed study reproducibility evidence.

`qualification_scope` states which quality contract passed. Numerical qualification is not dataset accuracy. Neither flag certifies physical performance. `hardware_performance_comparison_eligible` remains false; supplied measurements, calibration and source hashes are reported separately from independent authentication.

Any attached failed evidence blocks comparison. An enclosing study `summary.json` with `status` or `validation_status` equal to `reproducibility_failed` also blocks comparison and is hashed into the report. This is necessary for the actual CIMFlow v0.7 case: each sealed run is intact, but repeated timing differs. Moving bundles away from their study does not preserve unreferenced context; preserve the study evidence attachments when distributing them. No validator can discover an intentionally withheld external failure report.

## Repeated measurements and calibration

Each repetition is a verified bundle whose manifest hash is pinned. Its workload, implementation, platform, cost, status, quality and metric descriptor must match the corresponding base run. The base run must be included and duplicate run paths are rejected. Content integrity cannot prove statistical independence of separately named runs; reports state this limitation.

`summarize_samples` reports count, mean, median, minimum, maximum, p05 and p95. Deterministic execution requires at least two exactly matching samples for qualification and emits no confidence interval. Variation is a reproducibility failure; averaging cannot hide it.

`bootstrap_percentile` uses a fixed local random seed, at least five independent runs and 1,000–100,000 resamples. Its interval describes the **median of independent run-level metrics**, using percentile bootstrap quantiles. Five samples is only an admission guard, not an assurance of adequate statistical power. Request latencies within one run are not independent run samples. No ratio confidence interval or statistical significance is inferred from separate intervals.

`evaluate_calibration` accepts the existing detailed latency dataset format through `calibration.evaluate_validation`, and adds `pim-foundry.metric-calibration.v1` for generic scoped metrics such as energy. Generic datasets pin actual model/platform source files and exact input source files; identity digests are checked against bytes. Every prediction/measurement pair uses the same model, platform, input and metric descriptor. Each measurement has a distinct content-bound raw log, a positive value and an explicit externally-measured declaration. Train and held-out splits require at least two distinct inputs each and cannot overlap. Every held-out relative error must meet the threshold; training errors cannot cancel held-out failures. No parameters are fitted. Supplied measurement contents remain unauthenticated; disjoint hashes do not establish disjoint application families or absence of prior tuning.

## Executed acceptance checks

```sh
.venv-onnx/bin/python scripts/evaluate_comparison_protocol.py --out runs/comparison-protocol-new
python3 -m unittest discover -s tests -p test_comparison.py -v
```

The saved audit (local artifact: `../runs/comparison-protocol-v1-delivery/report.html`) rechecks sealed native v0.6/v0.7 artifacts. It admits the same-platform numerical/replay comparison, the GPU/NPU whole-system comparison, and the equal-capacity comparison. It rejects over-budget and changed-application comparisons, withholds task-quality qualification when no task evaluation exists, and blocks the actual CIMFlow reproduction failure. The GPU/NPU modeled ratio is 1 because their declared analytical parameters coincide; this does not show that physical GPU and NPU hardware perform equally. This audit launches no new hardware or simulator executions and makes no calibration claim.

The complete Transformer comparison (local artifact: `../runs/transformer-comparison-v08-release/report.html`) adds task-quality qualification to the pipeline. Both sealed implementations execute the pretrained tiny GPT-2 graph with the same explicit FP16 linear-node variant, inputs, HBM/GPU platform, cold graph-call policy and host KV transfers. Native HBM mixed mapping takes 193,551.9265 modeled ns; the all-GPU analytical mapping takes 190,874.0625 modeled ns. The mixed/all-GPU ratio is 1.01402948, so this configured model predicts the mixed mapping is about 1.4% slower. Exact replays pass, as do numerical checks and next-token NLL preservation on the fixed 16-scored-token fixture. This is an end-to-end integration and preservation check, not evidence of general model quality or physical speedup. The corrected study binds runtime files in its platform and execution identities.

```sh
.venv-onnx/bin/python scripts/evaluate_comparison_protocol.py \
  --transformer-study runs/transformer-native-hbm-v08-release \
  --out runs/transformer-comparison-new
```

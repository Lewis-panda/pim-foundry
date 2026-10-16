# Research evidence and upstream sources

A paper simulator may finish successfully while omitting arithmetic, using a different workload, or changing resources between baselines. The framework records those boundaries and preserves failures so that a successful wrapper run cannot silently become a validated performance claim.

This page summarizes the documented development studies. **Their historical run bundles are not included in the public repository.** The records are useful for understanding implemented paths and known issues, but a public checkout does not independently reproduce those historical observations by reading this page. Current clean-checkout validation is recorded in [VALIDATION.md](../VALIDATION.md).

## Evidence categories

| Evidence | What it establishes | What still needs checking |
|---|---|---|
| Functional output comparison | Actual outputs agree with a specified independent reference on the tested inputs | Broader input coverage, task quality, and timing accuracy |
| Original artifact execution | A pinned author executable ran under recorded conditions | Whether the path matches the claimed paper experiment and computes values |
| Direct/adapted parity and replay | The adapter preserves recorded observations and can execute them again | Model validity and physical accuracy |
| Resource-controlled model comparison | Ratios meet a declared task, metric, and resource contract | Causal isolation and hardware calibration |
| Hardware calibration | Predictions match supplied measurements within the evaluated domain | Measurement authenticity, generalization, and unseen workloads |
| Paper-number audit | Displayed arithmetic, aggregation, and resource accounting are internally checked | Original raw measurements and executable reproduction |

No category substitutes for the others. Artifact badges and publication venues do not certify a Foundry run.

## Integrated and audited sources

| Source | Implemented or historically exercised path | Boundary or known issue |
|---|---|---|
| [PIMbench / PIMeval](https://github.com/UVA-LavaLab/PIMeval-PIMbench/releases/tag/v1.0.0), [IISWC 2024 paper](https://www.cs.virginia.edu/venkat/papers/iiswc2024.pdf) | Pinned 2024 release; selected applications across three modeled PIM architectures, with independent semantic checks | Bounded geometry and workload subset. BMP/KNN reference semantics and excessive allocation issues are documented; passing an author reference alone was insufficient. [Details](pimbench-validation.md). |
| [CIMFlow framework](https://github.com/BUAA-CI-LAB/CIMFlow), [compiler](https://github.com/BUAA-CI-LAB/CIMFlow-Compiler), [simulator](https://github.com/BUAA-CI-LAB/CIMFlow-Simulator), [paper](https://arxiv.org/abs/2505.01107) | Original CNN backbone compilation/timing plus a separate native SRAM matrix integration | Selected repeated runs produced different timing for the same ISA. The historical study remains `reproducibility_failed`. Backbone timing omits operations needed for a complete classifier result. [Details](cimflow-application-validation.md). |
| [NeuPIMs](https://github.com/casys-kaist/NeuPIMs), [paper](https://arxiv.org/abs/2403.00579) | Original NPU/PIM timing model with a fixed 512-request, single-layer/single-token workload and direct/adapted/replay checks | Timing-only scope; no tensor-level task quality. Original example variants differed in request semantics or platform, so uncontrolled ratios were rejected. [Details](neupims-validation.md). |
| [STARC](https://github.com/EPIC-RPI/STARC), [paper DOI](https://doi.org/10.1145/3779212.3790226) | Original model and a separately identified safety repair, with a bounded 128-token window | The executed window does not reproduce the full long-generation sweep. Timing and cache-model behavior do not establish actual LLM quality. [Details](starc-source-notes.md). |
| [DCC](https://github.com/SPIN-Research-Group/DCC), [paper](https://arxiv.org/abs/2511.15503) | Original runtime defect investigation; a separate patched runtime feeds the Foundry HBM request timing projection | Explicit completion assumptions and separate instruction arithmetic remain visible. This is not a physical vendor ISA implementation or calibrated HBM-PIM measurement. [Details](dcc-runtime-audit.md). |
| [IANUS](https://arxiv.org/abs/2410.15008v1), [paper DOI](https://doi.org/10.1145/3620666.3651324) | Versioned public-paper arithmetic, denominator, resource, and artifact-discovery audit | No authenticated original author executable was obtained in that audit. A third-party [PHSim](https://github.com/Yueq-Zhang/PHSim) command-path test uses a different memory system and does not reproduce IANUS. [Details](ianus-validation.md). |

These sources are linked for provenance. Upstream trees, binaries, model releases, and datasets must be obtained separately under their own terms. Names of paper adapters identify integrations; they do not imply endorsement by the original authors.

## Complete applications versus timing proxies

The GPT-2 adapter executes actual pretrained weights through embeddings, decoder blocks, the vocabulary head, incremental KV state, and greedy token selection. A historical DistilGPT-2 study compared logits, generated tokens, KV tensors, and next-token NLL with an independent CPU Transformers reference. Its evaluation uses **128 targets from a fixed WikiText prefix**, not the full dataset.

A separate tiny GPT-2 fixture exercises selected explicitly FP16 linear nodes through the native HBM path while the remaining nodes use shared CPU functional execution and a GPU analytical profile. It is an integration fixture, not evidence of useful language quality or production-scale performance. Its historical configured model did not show a speedup over the matching analytical baseline. Source identities, numerical tolerance, and scope are recorded in [the Transformer documentation](transformer-application.md).

ONNX fixtures validate import and execution against the original serialized model. Random-weight interoperability fixtures establish a code path; they do not establish trained-model task accuracy. See [ONNX validation](tensor-onnx.md).

## Interpret discrepancies narrowly

The IANUS audit illustrates why denominators and aggregation belong in the protocol. A ratio of summed latencies can differ substantially from an arithmetic or geometric average of per-case speedups. A comparison of 8 GB PIM with 4 GB PIM plus 4 GB DRAM changes available PIM compute as well as allocation policy. Those configurations can be disclosed as whole systems, but do not isolate the policy's effect.

Some reported utilization ratios could not be reconciled with the paper's displayed throughput and peak figures under the audit's stated common-workload/common-aggregation assumptions. Missing raw data, a different denominator, or a different aggregation rule remain possible explanations. The finding requests clarification; it does not establish fabrication. The exact assumptions and paper locations are in [the IANUS audit](ianus-paper-audit.md).

Similarly, the CIMFlow timing discrepancy is a reproducibility finding under recorded inputs and runtime conditions. Its cause and consequences should not be generalized beyond the evidence. Comparison checks retain the failed study status rather than averaging the inconsistency away.

## Add an independently reproducible study

Publish a source revision, explicit application and platform definitions, exact evaluation inputs or their reproducible generation, an independent output oracle, raw observations, repeated-run procedure, and scope-correct baselines. Record preprocessing, precision, memory capacity, transfer costs, quality policy, and excluded activities. Calibration needs a separate held-out set and raw measurement provenance.

Use [the comparison protocol](comparison.md) and [paper-validation protocol](paper-validation-protocol.md) to bind those objects. Include negative controls that exercise reference failures, changed workloads, and resource mismatches. Preserve failures alongside successful runs. Generated bundles belong in a separately licensed experiment release or artifact archive when their size or asset terms make them unsuitable for the source repository.

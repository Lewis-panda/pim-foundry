# PIM Foundry

**One application. Many memory architectures. Evidence you can replay.**

[![CI](https://github.com/Lewis-panda/pim-foundry/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Lewis-panda/pim-foundry/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Lewis-panda/pim-foundry)](https://github.com/Lewis-panda/pim-foundry/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-0f766e.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-2563eb.svg)](pyproject.toml)

[繁體中文](README.zh-TW.md) · [Quickstart](#run-your-first-experiment) · [Paper studies](#published-papers-concrete-results) · [Documentation](#go-further) · [Contribute](CONTRIBUTING.md)

Move an operation from SRAM-CIM to HBM-PIM, add a modeled GPU or NPU, and see how **actual tensor outputs, data movement, resource contention, and cost predictions** change. PIM Foundry keeps the application, precision policy, reference outputs, and comparison boundary explicit.

![PIM Foundry workflow: a fixed application and shared tensor graph, configurable SRAM/HBM and analytical GPU/NPU backends, independent numerical checks, reports and replay.](docs/assets/overview.svg)

- **Build across memory architectures.** A shared tensor graph, ONNX frontend, complete GPT-2 prefill/decode, and explicit backend interfaces.
- **Compare under declared conditions.** Change placement or architecture while tracking precision, capacity, transfers, request arrivals, and metric scope.
- **Inspect every claim.** Independent output checks, original simulator counters, content-bound inputs, and frozen-engine replay accompany the result.

Research preview, v0.8: native paths integrate separately built simulators; GPU/NPU costs are analytical and tensor arithmetic runs on the CPU. APIs may evolve before 1.0. [Execution boundaries](docs/limitations.md).

## How it works—and what you can simulate

An SRAM projection produces 96 bytes of INT32 activations. The GPU consumes them, applies ReLU, and explicitly casts them to 48 bytes of FP16 before HBM executes the next projection. Actual tensor values feed subsequent operations; their transfers enter the recorded schedule.

![Four active device combinations with explicit directed transfers: GPU, HBM plus GPU, HBM plus NPU, and SRAM plus HBM plus GPU.](docs/assets/supported-architectures.svg)

The [architecture gallery](docs/architecture-gallery.md) runs **all four placements**, checks an independent numerical reference, verifies/replays each bundle, and records compute, link, DMA, and peak-memory costs. It keeps the complete platform and workload fixed and uses equal illustrative analytical coefficients.

```sh
# After installing the tensor or ONNX extra:
python examples/architecture_gallery.py --out runs/architecture-gallery
```

<details>
<summary><strong>Follow the execution engine and a real generated schedule</strong></summary>

![Application assets, actual tensor execution, separate resource scheduling, independent checks and reproducible evidence.](docs/assets/system-operation.svg)

![Recorded three-request SRAM/HBM/GPU compute and DMA schedule.](docs/assets/architecture-timeline.png)

[Open the detailed walkthrough](docs/architecture-gallery.md) · [Recorded events and results](validation/architecture-gallery.json) · [Download/open the local interactive explorer](docs/assets/architecture-explorer.html)

The figures show model execution and predicted resource reservations; the portable gallery executes no physical accelerator or native simulator.

</details>

## Results you can inspect

**A complete pretrained model, checked through every decode step.** The DistilGPT-2 development study executes all six layers, embeddings, attention, MLPs, vocabulary projection, and KV updates against an independent Transformers CPU reference.

![Validation results: 493 tests passed and 25 optional integrations skipped; all eight generated tokens match and all twelve KV tensors pass; DistilGPT-2 logit and NLL differences remain within the declared bounds.](docs/assets/validation-results.png)

The software regression results come from the public checkout. The pretrained results are historical local observations on **128 next-token targets** from a fixed WikiText prefix. Model weights and data must be obtained separately; the historical raw run bundles are not distributed. [Machine-readable results](validation/local-v0.8.json) · [Pretrained setup and scope](docs/transformer-application.md) · [CI evidence](VALIDATION.md).

**A controlled paper-model comparison.** In the NeuPIMs study, round-robin (RR) and channel-load-balanced (CLB) placement use the **same ordered 512 requests and the same platform**. CLB lowers modeled latency by **4.77%** in this one-layer, one-token batch, including simulator drain.

![NeuPIMs controlled comparison: round-robin 650.229 microseconds, channel load balancing 619.233 microseconds, on an identical 512-request workload and platform. These are simulator predictions.](docs/assets/neupims-comparison.png)

This is a scoped placement ablation from a historical local study. It measures simulated timing; task quality and physical hardware performance were not evaluated. The different-platform NPU-only run is retained in the evidence and excluded from this same-platform ratio. [Inputs, counters, negative controls, and rerun procedure](docs/neupims-validation.md).

## Published papers, concrete results

Explore **IISWC, DAC, ASPLOS, and ISCA** research through pinned source revisions and explicit study boundaries. The studies below are documented development runs and audits; original upstream assets and historical raw bundles are not included in the public repository. The status applies to the listed experiment, rather than every claim in the paper.

| Paper | What the study demonstrates | Status / scope |
|---|---|---|
| **[PIMbench](https://www.cs.virginia.edu/venkat/papers/iiswc2024.pdf)** · IISWC 2024 | **9 application/architecture cases**, across bit-serial, Fulcrum, and bank-level PIM; 27 direct/adapted/replay invocations. **7,168 triangles** independently verified. | **Selected applications executed.** Bounded geometry; KNN task quality unqualified. [Evidence / rerun](docs/pimbench-validation.md). |
| **[NeuPIMs](https://arxiv.org/abs/2403.00579)** · ASPLOS 2024 | **512 matched requests**, three variants and three replays; direct/adapted timing counters agree. Controlled CLB/RR comparison shown above. | **Timing parity + replay.** One layer, one decode token; no tensor-level quality. [Evidence / rerun](docs/neupims-validation.md). |
| **[STARC](https://doi.org/10.1145/3779212.3790226)** · ASPLOS 2026 | **508 native calls** across Full/STARC runs and replays. CSV/cache outputs match independent direct runs **byte for byte**. | **Parity within a 128-token window.** Separately identified safety repair; full long-generation sweep untested. [Evidence / rerun](docs/starc-application.md). |
| **[CIMFlow](https://arxiv.org/abs/2505.01107)** · DAC 2025 | ResNet-18 DP agrees at **3.560054 ms** across direct/adapted/replay runs; other tested mappings expose timing drift despite identical ISA bytes. | **Reproducibility failure retained.** Backbone timing; no complete classifier-quality claim. [Evidence](docs/cimflow-application-validation.md). |
| **[DCC](https://arxiv.org/abs/2511.15503)** · ISCA 2026 | Reproduced a controller use-after-free with AddressSanitizer; published a separately identified repair and explicit command-completion model. | **Runtime audit + patched integration.** Completion delay is an explicit assumption. [Audit](docs/dcc-runtime-audit.md) · [Patch](patches/dcc/README.md). |
| **[IANUS](https://arxiv.org/abs/2410.15008v1)** · ASPLOS 2024 | **8 arithmetic checks** agree within displayed precision; **4 denominator checks** flag inconsistencies under stated assumptions. | **Paper audit only.** No authenticated author executable obtained. [Evidence](docs/ianus-validation.md). |

**Find problems that a passing reference can miss.** The PIMbench audit catches BMP metadata counted as image pixels and labels leaking into KNN distance, even though the original CPU reference accepts the outputs. Foundry records those outcomes separately from numerical agreement and excludes ineligible results from quality-preserving comparisons. [Read the counterexamples](docs/pimbench-validation.md#independent-failure-evidence).

For source provenance, failure analysis, and the distinction between application execution, paper reproduction, and hardware calibration, see [Research evidence](docs/research-evidence.md). Paper names identify integrations and audits; they do not imply author endorsement.

## Run your first experiment

Python 3.10+ and a CPU are enough. After installation, this example runs **offline**, with generated inputs and no model download or external simulator.

```sh
git clone https://github.com/Lewis-panda/pim-foundry.git
cd pim-foundry
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[onnx]'
python examples/quickstart.py --out runs/quickstart
```

The example executes `matmul → relu → matmul → softmax`, compares an independent scalar reference, schedules an illustrative HBM/GPU configuration, and verifies and replays the result. Open **`runs/quickstart/application/report.html`** and inspect **`runs/quickstart/summary.json`**. Its timing/energy are analytical predictions using example coefficients.

Use a fresh output directory for each run. Install `.[tensor]` if you only need NumPy execution. [Change the graph, mapping, or platform](docs/getting-started.md).

## Make it your experiment

| Your next question | Start here |
|---|---|
| Can my application run through a shared tensor representation? | [Tensor graph and operations](docs/tensor-runtime.md) · [ONNX support](docs/tensor-onnx.md) |
| What changes when I combine HBM-PIM, SRAM-CIM, GPU, or NPU models? | [Platform and scheduling](docs/tensor-platform.md) · [Native SRAM/HBM setup](docs/native-quickstart.md) |
| Does prefill/decode preserve actual model outputs and KV state? | [Complete Transformer application](docs/transformer-application.md) |
| Is this ratio comparing equivalent workloads and resources? | [Same-platform, equal-budget, and whole-system tracks](docs/comparison.md) |
| How do I add my architecture or paper artifact? | [Backend interface](docs/backend-plugin.md) · [Application adapter contract](docs/applications.md) |

Unsupported operators, precision, routes, and capacity fail explicitly. Precision conversions appear in the graph; unknown costs stay identified. Changing architecture does not silently grant a fair-comparison label.

## Go further

[Getting started](docs/getting-started.md) · [Architecture](docs/architecture.md) · [Validation](VALIDATION.md) · [Research evidence](docs/research-evidence.md) · [Limitations](docs/limitations.md) · [Figure sources](docs/assets/README.md)

Bring a model, a backend, or a result to reproduce. [Open an issue](https://github.com/Lewis-panda/pim-foundry/issues) or follow the [contribution guide](CONTRIBUTING.md) to submit a change. [CITATION.cff](CITATION.cff) provides software citation metadata.

First-party code and original figures are [MIT licensed](LICENSE). Referenced papers, simulator sources, models, and datasets retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

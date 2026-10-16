# PIM Foundry

**Run the same application across declared CIM/PIM and accelerator configurations, check its outputs, and preserve the evidence behind each comparison.**

[繁體中文](README.zh-TW.md) · [Getting started](docs/getting-started.md) · [Architecture](docs/architecture.md) · [Limitations](docs/limitations.md) · [Validation](VALIDATION.md)

A matrix operation on HBM consumes the actual tensor produced by its predecessor. Its output flows into the next operation, while a separate resource model accounts for compute, transfers, contention, and memory capacity. Changing the platform or placement keeps the application, inputs, and quality policy explicit.

PIM Foundry is an experimental research framework for compute-in-memory (CIM), processing-in-memory (PIM), and heterogeneous systems. It provides a shared application contract, executable tensor representation, simulator integration, and comparison protocol. It is under active development; APIs are versioned but may change before 1.0.

## Try it locally

Python 3.10+ and a CPU are sufficient for this example. After installation, the quickstart needs no network access, model download, GPU, or external simulator.

```sh
git clone https://github.com/Lewis-panda/pim-foundry.git
cd pim-foundry
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[onnx]'
python examples/quickstart.py --out runs/quickstart
```

The example computes actual arrays, checks an independent numerical reference, evaluates a declared HBM/GPU analytical configuration, and verifies and replays the saved application. Open `runs/quickstart/application/report.html` and inspect `runs/quickstart/summary.json`. Its latency and energy are **model predictions using example coefficients**. It does not run on physical HBM-PIM or GPU hardware. Choose a fresh output directory for each run.

See [Getting started](docs/getting-started.md) to modify the application or placement. Install `.[tensor]` instead if you need NumPy tensor execution without ONNX import.

## What is included

| Capability | Current implementation |
|---|---|
| Shared application entry point | Content-bound model, inputs, scenario, reference outputs, and quality policy; separate platform and placement configuration |
| Executable tensor graph | n-D arrays, explicit precision, symbolic dimensions, broadcasting, attention, normalization, CNN operations, and explicit state inputs/outputs |
| ONNX frontend | A documented operator subset, with numerical comparison against the original ONNX model |
| Complete GPT-2 application | Checkpoint weights, embeddings, all decoder layers, LM head, prefill, incremental decode, KV cache, greedy tokens, and scoped next-token NLL checks |
| SRAM and HBM integration | INT8/INT32 CIMFlow instruction interpretation; FP16 HBM instruction interpretation and a separately identified DCC timing projection |
| Heterogeneous scheduling | Compute, directed links, endpoint DMA, tensor lifetimes, workspace, capacity, request arrivals, and cold/warm weight policies |
| Reproducible experiments | Asset and source snapshots, file hashes, numerical checks, inspectable reports, verification, and replay |
| Comparison protocol | Same-platform, equal-budget, and whole-system tracks; declared resources, quality, metric scope, repetitions, and calibration evidence |

**A device name does not establish hardware execution.** GPU/NPU configurations currently use analytical costs and CPU numerical execution. Native simulator integrations require separately obtained and built upstream runtimes. Hardware calibration and general accuracy across devices have not been established.

## Connect an application or backend

Start with `tensor-graph/v2` for a graph you author, `onnx-tensor/v2` for a supported ONNX model, or `transformer-graph/v1` for the implemented GPT-2 architecture. An application supplies the workload and quality check; an experiment supplies devices, links, placement, and explicit model assumptions.

- [Tensor operations and numerical semantics](docs/tensor-runtime.md)
- [ONNX operator support](docs/tensor-onnx.md)
- [Transformer state and evaluation contract](docs/transformer-application.md)
- [Device execution and scheduling](docs/tensor-platform.md)
- [Application adapter contract](docs/applications.md) and [backend extension interface](docs/backend-plugin.md)

Unsupported operators, precision, routes, or capacity fail explicitly. Precision conversion must appear in the graph. A JSON configuration cannot dynamically import an arbitrary backend; extensions are registered explicitly in Python.

## Compare results with an explicit question

A comparison fixes the application and metric boundary, then states which resources may differ:

- **Same platform:** compare mappings or implementations under the same platform and cost contract.
- **Equal budget:** compare architectures under declared common resource constraints.
- **Whole system:** compare disclosed configurations while retaining their resource differences.

Numerical correctness, task quality, replay, paper reproduction, and hardware calibration are separate evidence. A sealed file manifest establishes content integrity; it does not authenticate measurements. Unknown energy or power stays unknown. See the [comparison protocol](docs/comparison.md) and [research evidence](docs/research-evidence.md).

## Paper integration and evidence

Adapters and audit scripts cover selected paths from PIMbench, CIMFlow, NeuPIMs, STARC, and DCC. IANUS has a source/claim audit; an authenticated author executable was not obtained during that audit. Integration of one workload or simulator component does not establish reproduction of a paper's complete performance claims.

The public repository contains first-party code, tests, configurations, and documentation. Upstream simulator source trees, downloaded checkpoints, datasets, and historical experiment bundles are not redistributed here. [Research evidence](docs/research-evidence.md) identifies the original sources, evaluated scope, and known failures. [VALIDATION.md](VALIDATION.md) separates public-checkout checks from historical local acceptance results.

## Development

```sh
python -m pip install -e '.[onnx]'
python -m unittest discover -s tests -v
```

Tests requiring optional dependencies or external runtimes may skip when those prerequisites are absent; the test report preserves that distinction. See [Contributing](CONTRIBUTING.md) for targeted checks, backend requirements, and reproducible bug reports. Suggestions and contributions are welcome through [issues](https://github.com/Lewis-panda/pim-foundry/issues) and pull requests.

## License

First-party code is licensed under [MIT](LICENSE). Referenced third-party software, papers, models, and datasets retain their own licenses. Their inclusion in an experiment does not change those terms.

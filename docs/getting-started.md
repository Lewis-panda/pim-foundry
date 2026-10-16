# Getting started

The quickstart sends actual arrays through two matrix operations, ReLU, and softmax. It compares the resulting probabilities with independently calculated scalar outputs, then schedules the graph on a declared HBM/GPU model. This gives you a complete, inspectable experiment before you build any external simulator.

## Install and run

Use Python 3.10 or later. These commands use a POSIX shell; on Windows, activate the virtual environment using its `Scripts` directory instead.

```sh
git clone https://github.com/Lewis-panda/pim-foundry.git
cd pim-foundry
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[onnx]'
python examples/quickstart.py --out runs/quickstart
```

Installation retrieves Python dependencies. The example itself is offline and needs only a CPU. It generates its inputs and independent reference locally; it does not download a model, dataset, or simulator. For tensor execution alone, `.[tensor]` is sufficient. The ONNX extra adds the supported model importer and its reference evaluator.

The output directory must not already exist. To repeat the example, choose another name such as `runs/quickstart-second`.

## Inspect what happened

| Output | Purpose |
|---|---|
| `runs/quickstart/summary.json` | Overall checks, replay result, and modeled baseline comparison |
| `runs/quickstart/application/report.html` | Human-readable application result |
| `runs/quickstart/application/result.json` | Scoped metrics, quality, and execution identities |
| `runs/quickstart/fixture/` | Editable source graph, inputs, reference, and application manifest |
| `runs/quickstart/experiment.json` | Declared device models, links, and mixed placement |
| `runs/quickstart/baseline-experiment.json` | Same declared platform, with all nodes placed on the GPU model |
| `runs/quickstart/replay/` | A newly executed replay of the sealed mixed application |
| `runs/quickstart/baseline/` | The executed all-GPU analytical baseline |

Open the HTML file in a browser. The summary should report `passed: true`, and the application should report numerical quality as passed. Task quality is not evaluated: a small probability-array fixture has no trained-model dataset score.

Both device kinds use identical illustrative analytical coefficients. The CPU computes all arrays. A different latency prediction reflects this configured schedule and its transfers; it cannot establish a physical HBM-versus-GPU ranking. A reported zero energy coefficient is an explicit example assumption, not a measurement of zero power.

The scenario contains three modeled requests, two initially arriving together and one arriving later, with at most two in flight and warm initializer reuse. They share one verified operation profile; these are workload predictions, not three independent experimental repetitions.

## Verify and replay yourself

```sh
python -m pim_foundry app-verify runs/quickstart/application
python -m pim_foundry app-replay runs/quickstart/application --out runs/quickstart/manual-replay
python -m pim_foundry app-compare runs/quickstart/application runs/quickstart/baseline --metric latency
```

Verification checks recorded bytes and the complete file manifest. Replay executes the preserved source and inputs, then checks the resulting observations and identities. Neither operation authenticates hardware measurements. Keep sealed bundle contents unchanged; edit the source fixture and create a new bundle for a new experiment.

## Change the placement

Copy the generated experiment and edit its `mapping.placement` object:

```sh
cp runs/quickstart/experiment.json runs/quickstart/my-experiment.json
```

The node names are `projection`, `activation`, `head`, and `softmax`; the declared devices are `hbm` and `gpu`. For example, move `head` to `hbm` while leaving the other entries unchanged. In this example both backends are `analytical`, so this changes modeled placement while retaining the functional numerical path.

```sh
python -m pim_foundry app-inspect --application runs/quickstart/fixture/application.json --experiment runs/quickstart/my-experiment.json
python -m pim_foundry app-run --application runs/quickstart/fixture/application.json --experiment runs/quickstart/my-experiment.json --out runs/my-placement
python -m pim_foundry app-compare runs/quickstart/application runs/my-placement --metric latency
```

Changing device capacity, bandwidth, or other platform assumptions creates a different platform identity. The same-platform comparator will then reject the comparison. Use an explicit [equal-budget or whole-system protocol](comparison.md) when that is the intended research question.

## Supply your own application

Use the generated fixture as a schema example:

1. Describe operations, shapes, storage dtypes, inputs, outputs, and initializer weights in a [v2 tensor graph](tensor-runtime.md), or provide a [supported ONNX model](tensor-onnx.md).
2. Supply inputs and independent reference outputs in NPZ files. Obtain the reference from a separate implementation or a justified analytic oracle. Store no pickled objects.
3. Create an application manifest using `tensor-graph/v2` or `onnx-tensor/v2`, binding every asset's SHA-256, semantic role, format, and path. Declare the scenario and numerical tolerance before comparing outputs. The helper `pim_foundry.application.asset(path, role, format)` creates an asset entry.
4. Supply the experiment's devices, links, and placement; run `app-inspect`, `app-run`, `app-verify`, and `app-replay` as above with your new manifest.

Updating graph or input bytes requires updating their manifest hashes. A reference array must cover every declared graph output with the right shape and dtype. Generic numerical equivalence does not evaluate a downstream task; add a task-specific adapter when the result needs accuracy, NLL, or another application-level measure.

For checkpoint-based GPT-2 generation and KV state, use the [Transformer adapter](transformer-application.md). Preparing its independent reference additionally requires the documented optional reference dependencies and separately obtained model/data assets. Downloads are explicit; they are not part of the quickstart.

## Enable native simulators

A platform's `backend` field selects numerical/timing integration; its `kind` is only a device category. To use `cimflow-sram` or `hbm-simd16`, obtain the appropriate upstream sources, comply with their licenses, build the external runtime, and supply its explicit binary or runtime-descriptor path. See [native setup](native-quickstart.md), [CIMFlow integration](cimflow-backend.md), and [DCC runtime audit](dcc-runtime-audit.md).

The public checkout does not contain prebuilt runtimes or historical local experiments. Native capability checks reject unsupported precision or geometry. Selecting a missing native backend will fail instead of silently substituting the analytical example.

## Common failures

| Failure | What to check |
|---|---|
| Missing NumPy/ONNX | Install the appropriate optional extra in the active virtual environment. |
| Output directory already exists | Use a new directory; runs are not overwritten. |
| Asset hash mismatch | Restore the intended asset or create a new manifest for the changed input. |
| Unsupported operator or dtype | Check the frontend and backend support tables; preserve precision conversion explicitly. |
| No route or insufficient capacity | Inspect directed links, placement, tensor sizes, and workspace. |
| Numerical quality failure | Compare actual outputs with an independent oracle and investigate before changing tolerances. |
| Native runtime missing | Build and bind the external runtime; the analytical quickstart remains available. |

For reproducible bug reports and development checks, see [Contributing](../CONTRIBUTING.md).

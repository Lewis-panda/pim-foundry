# Validation scope

The public checkout is tested without private run directories, downloaded model
weights, author generator files or native simulator builds. Skips are explicit;
they are not counted as passing tests.

## Clean public source, Python 3.12.3

| Profile | Discovered | Passed | Skipped | Failures/errors |
|---|---:|---:|---:|---:|
| Standard-library core | 518 | 398 | 120 | 0 / 0 |
| Tensor + ONNX extras | 518 | 493 | 25 | 0 / 0 |

The ONNX profile's skips comprise 14 tests requiring separately obtained author
DCC generator files, 7 tests requiring the optional DCC runtime descriptor, and
4 native integration/build opt-ins. The core profile additionally skips optional
NumPy/ONNX functionality. Source-integrity and DCC process/error-boundary tests
use self-contained fixtures and remain active in both profiles.

Machine-readable reports with every skip reason:
[core](validation/public-core.json), [tensor/ONNX](validation/public-onnx.json).
These are local clean-checkout observations recorded for the first public
release. The [GitHub CI workflow](.github/workflows/ci.yml) independently runs
Python 3.10 and 3.12, both dependency profiles, and a wheel installed away from
the checkout. Its live results appear in the repository's Actions tab; a local
report does not claim a remote CI job has passed.

```sh
python -m pip install '.[onnx]'
python scripts/ci_checks.py --profile onnx --report runs/public-tests.json
python examples/quickstart.py --out runs/quickstart
```

The example creates its own small tensors and an independent scalar reference,
executes an illustrative heterogeneous mapping and same-platform baseline,
verifies the sealed application, and replays it. Its device timing is an
uncalibrated analytical model, not a hardware measurement.

## Historical local research acceptance

Before this public packaging, a local workspace with separately built native
runtimes passed 518 tests without skips and verified 19 sealed application
bundles. Selected observations and original source-module hashes are preserved
in [local-v0.8.json](validation/local-v0.8.json). They include complete
DistilGPT-2 inference, a tiny full GPT-2/native HBM integration, and independent
quality and replay checks. The raw bundles are not redistributed because they
contain external assets and source snapshots. The JSON is an author-provided
summary, not independent attestation or a substitute for executing a study.

The public tests replace some old source-dependent setup with synthetic fixtures
and add explicit dependency gates. Consequently historical source hashes and
test environments must not be conflated with this public CI run.

DistilGPT-2 quality covers 128 next-token targets from a fixed WikiText test
prefix; tiny GPT-2 is a functional integration fixture. The HBM/GPU modeled
latency ratio was 1.01403 in the declared small fixture: it did not show a modeled
speedup. Neither result establishes physical performance or complete paper
reproduction. See [research evidence](docs/research-evidence.md) and
[limitations](docs/limitations.md).

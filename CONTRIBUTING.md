# Contributing

A useful change leaves a result that another researcher can inspect: the exact workload, actual outputs, cost assumptions, and checks that were run. Contributions may add applications, frontend operators, backends, evaluation protocols, tests, or documentation.

## Start with a reproducible case

For a bug, include the command or minimal Python example, package revision, Python/dependency versions, platform/backend configuration, expected behavior, and observed error. Keep model/input files small and shareable. Remove credentials, personal data, private paths, and licensed assets you cannot redistribute.

For a research result, state the model, input selection, precision, resource configuration, metric scope/units, aggregation, reference method, and repetition protocol. Distinguish observed output, model prediction, author-reported numbers, and independently measured hardware data. Include failures and excluded cases; a favorable ratio alone is insufficient.

An issue is helpful for a substantial new interface or backend. Small fixes can go directly to a pull request; an issue is not a prerequisite.

## Local development

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[onnx]'
python examples/quickstart.py --out runs/contributor-quickstart
python -m unittest discover -s tests -v
```

The core install has no required runtime dependencies; tensor and ONNX paths use optional extras. Tests needing absent optional dependencies or external runtimes report skips. Preserve the count and reason when reporting results. Current public-checkout evidence is in [VALIDATION.md](VALIDATION.md).

For a focused change, run the affected tests first, for example:

```sh
python -m unittest discover -s tests -p test_tensor_runtime.py -v
python -m unittest discover -s tests -p test_tensor_onnx.py -v
python -m unittest discover -s tests -p test_comparison.py -v
```

Run the broader suite when a change affects shared arithmetic, scheduling, schemas, adapters, or comparison behavior. Pure documentation changes need accurate examples and link checks rather than artificial tests. Optional native acceptance requires the external prerequisites described in the relevant backend documentation; do not report those checks as passed when they did not execute.

CI checks core and ONNX installation profiles and exercises an installed wheel through the offline quickstart. CI does not obtain every paper artifact, download pretrained models, or authenticate hardware performance.

## Implementation expectations

- Preserve explicit shape, dtype, rounding, overflow, state, and ownership contracts. Unsupported cases should fail before a plausible metric is reported.
- Validate numerical changes against an independent oracle. State tolerances and include meaningful failure cases, such as wrong reference outputs, cache contamination, incompatible shapes, or overflow.
- Keep simulator host runtime separate from modeled device latency. Include workspace, movement, and scope exclusions explicitly; unknown quantities stay unknown.
- Keep ordering and random seeds deterministic where promised. If nondeterminism is intentional, identify the sample unit and use an appropriate repetition protocol.
- Preserve content identities, asset coverage, and replay checks. A new schema or numerical contract needs a versioned compatibility decision.
- Keep changes focused. Document the concrete mechanism and its effect before introducing new terminology.

New backend contributions should include capability validation, actual numerical outputs or an explicit timing-only contract, runtime/source identity, accounting scope, and an independent correctness check. [Architecture](docs/architecture.md) identifies extension points; [backend-plugin.md](docs/backend-plugin.md) describes the existing matrix backend interface. Additional tensor operations may need explicit lowering and scheduling integration beyond backend registration.

A paper adapter should pin the source revision, preserve its execution semantics, check direct-versus-adapter output where possible, and record relevant upstream defects. A locally fixed variant must have a distinct identity and explain its effect. Reproducing one kernel or request window must not be described as reproducing the full paper.

## Submit a pull request

Use a descriptive branch and explain the concrete problem, resulting behavior, checks actually run, and any numerical/schema/evidence changes. Follow the repository's pull request template. Include source and license information for introduced material; avoid generated runs, external source checkouts, model weights, binaries, and datasets unless their addition has an explicit distribution rationale and compatible licensing.

First-party contributions are submitted under this project's [MIT license](LICENSE). Referenced third-party assets keep their own terms. Do not copy code or figures with uncertain redistribution rights into the repository.

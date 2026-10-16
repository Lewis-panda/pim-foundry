# Homepage figures and their evidence

These are original PIM Foundry figures. No paper figure, author logo, or third-party screenshot has been copied. The project [MIT license](../../LICENSE) applies to the artwork and plotting code; linked papers and external artifacts keep their own terms.

| Figure | Input and scope |
|---|---|
| [Overview SVG](overview.svg) | Editable architecture diagram, based on [the implemented execution interfaces](../architecture.md). It distinguishes shared tensor execution, native simulator integration, and analytical GPU/NPU models. It is a schematic, not a measured result. |
| [Validation PNG](validation-results.png) · [SVG](validation-results.svg) | [Public ONNX-profile test report](../../validation/public-onnx.json), [historical local v0.8 observations](../../validation/local-v0.8.json), and [the declared numerical policy](../transformer-application.md). The 493 passes and 25 skips are separate. DistilGPT-2 covers 128 next-token targets, eight generated tokens, and twelve K/V tensors. The plotted error bars are scaled to their labeled bounds; the complete logit check uses both absolute and relative tolerance. |
| [NeuPIMs PNG](neupims-comparison.png) · [SVG](neupims-comparison.svg) | The controlled RR/CLB observations in [the NeuPIMs study](../neupims-validation.md): 650,229 ns and 619,233 ns, converted to microseconds. Both use the same ordered 512 requests and platform. The axis starts at zero; the reduction is `100 × (1 − CLB / RR)`. The different-platform NPU-only result is excluded from this comparison. |

The historical observations are author-provided records, not independent attestation. Raw historical run bundles are not redistributed. The charts establish no measured hardware speedup or complete-paper reproduction.

## System and architecture gallery

- [System operation](system-operation.svg): application assets and scenario, actual tensor execution, separate resource prediction, independent checks and replay.
- [Supported combinations](supported-architectures.svg): active paths on the shared four-device platform; selected links are drawn, unused devices omitted.
- [Resource timeline PNG](architecture-timeline.png) / [SVG](architecture-timeline.svg): generated from the recorded events in [architecture-gallery.json](../../validation/architecture-gallery.json). Repeated transfer bands on endpoint DMA rows are the same reservation.
- [Interactive explorer](architecture-explorer.html): self-contained local HTML with embedded recorded data. It switches placements, filters requests, shows link resources and exposes each event's exact values. Download/open locally; GitHub displays HTML source.

Run [the executable example](../../examples/architecture_gallery.py) with an installed tensor extra, then use [render_architecture_gallery.py](../../scripts/render_architecture_gallery.py) with an already available Matplotlib installation. The [walkthrough](../architecture-gallery.md) gives both commands. No model assets or native runtimes are needed. These are synthetic-cost predictions and actual numerical checks, not physical performance data.

## Regenerate the result figures

The rendering dependency is separate from the framework's runtime requirements. From the repository root, use a dedicated documentation environment:

```sh
python3 -m venv .venv-figures
.venv-figures/bin/python -m pip install matplotlib==3.10.0
.venv-figures/bin/python scripts/render_readme_figures.py
```

The [generator](../../scripts/render_readme_figures.py) reads the checked-in evidence and extracts the policy and timing observations from their source documents. It fails if those fields no longer match the expected structure. Review the captions when updating an evidence source; a new count or value can also change its meaning. Fixed SVG IDs and metadata avoid timestamps. Exact image bytes can still depend on font and rendering-library versions.

Edit `overview.svg` directly for architecture changes. Keep its title, description, and execution-mode labels accurate. All diagrams are self-contained: no remote fonts, scripts, images, or active content.

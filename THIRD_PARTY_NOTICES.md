# Third-party sources and scope of the license

The root MIT license covers the original PIM Foundry Python package, tests,
examples, adapters, documentation and original portions of its patches. It does
not relicense models, datasets, papers, simulator binaries or author artifacts.

## Distributed patch context

`patches/dcc/0001-controller-lifetime-and-explicit-completion.patch` contains
changes and context from the `src/Ramulator2/HBMPIM` subtree of
[SPIN-Research-Group/DCC at 7ffb80e4f33e357177dc8ab148953c0484e1e322](https://github.com/SPIN-Research-Group/DCC/tree/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM).
That subtree carries the MIT license, copyright 2023 SAFARI Research Group at
ETH Zurich and Carnegie Mellon University. The original notice is reproduced
verbatim in [patches/dcc/UPSTREAM_LICENSE](patches/dcc/UPSTREAM_LICENSE).
The fixed source list and hashes are preserved in `patches/dcc/source-manifest.json`.
No full simulator source or binary is bundled.

## Sources intentionally not redistributed

The separate DCC configuration and trace-generator scripts were used in local
research, but no applicable license notice for those files was established.
Their bytes are therefore absent from this repository. `upstream/manifest.json`
retains their public source URLs, revision and hashes for provenance only.
Tests that execute these author scripts require a separately obtained checkout
and report explicit skips when it is unavailable. The MIT license of the
Ramulator subtree is not extended to those other files.

CIMFlow, NeuPIMs, STARC, PIMbench/PIMeval, PHSim, pretrained checkpoints and
benchmark datasets remain separate dependencies under their owners' terms.
Paper adapters call explicitly selected local artifacts. Model setup scripts
can obtain pinned public files only when invoked; the basic quickstart downloads
no model or author code. Obtain and review the relevant source's terms before
redistributing any resulting bundle, which can contain source snapshots,
weights, input data or absolute paths.

Python dependencies installed through package extras retain their own licenses.
Research methods and source references are documented in
[the evidence guide](docs/research-evidence.md). A reference is not a claim of
endorsement or of successful whole-paper reproduction.

# PIMbench: application validation and independently detected failures

The original histogram application counts every byte after the BMP pixel offset, including an embedded color profile. Its CPU reference repeats the same mistake and reports `Correct!`. Foundry now binds the input file, checks the image domain independently, and retains that run as `quality_failed` rather than admitting it to a performance comparison.

This adapter executes the **unmodified IISWC 2024 submission artifact**, release `v1.0.0`, commit `9792b8fc281737e1c4aa8c31b9e68b54225b21c2`, from [the authors' repository](https://github.com/UVA-LavaLab/PIMeval-PIMbench/releases/tag/v1.0.0). The paper is [Architectural Modeling and Benchmarking for Digital DRAM PIM](https://www.cs.virginia.edu/venkat/papers/iiswc2024.pdf). A newer TACO 2026 extension and current repository code exist; the findings below apply to this pinned 2024 release only.

## What is executed

Three intact C++ application implementations run through `pimbench-artifact/v1`, followed by independently constructed direct author commands and frozen-engine application replays. The PIMeval library executes functional PIM operations and supplies its original analytical runtime/energy counters. This is an original-artifact application integration; these applications have not been lowered into Foundry's matrix IR.

| Application | Fixed input | Reference evidence |
|---|---|---|
| Histogram | Deterministic 256 × 256, uncompressed 24-bit BMP; 65,536 pixels | Author checks all 3 × 256 bins against its CPU reference. Foundry validates pixel boundaries independently; full bin arrays are not exported. |
| Triangle count | 1,024 vertices; 3,711 edge rows; 128 complete eight-vertex cliques connected by bridges | PIM and author CPU counts are checked against an independent set-intersection oracle: 7,168 triangles. |
| KNN | 2,048 training points, 64 queries, two feature columns plus one label column, k = 3 | Author PIM versus CPU reference agreement. Task accuracy is not certified because the original algorithm includes the label in distance. |

All three applications run under bit-serial, Fulcrum, and bank-level PIM models. This yields nine integration cases and 27 native invocations including direct runs and replays. The unchanged original BMP counterexample also uses bounded geometry and adds two native invocations (application and replay); the KNN independent audit adds three harness invocations. These are complete executions of the selected application algorithms at disclosed validation scales, not reproduction of the publication's full-size speedup figures.

The input-generation formulas and exact file hashes are in the study's `specifications/input-provenance.json` and application manifests. Model/source definitions are semantic assets, so changing an algorithm revision changes task identity. Device architecture, geometry profile, runtime binding and mapping remain experiment settings.

## Measurement boundary

The `latency` and `energy` metrics are the sum of the original simulator's **PIM command totals plus modeled host/device copies**. Their units are `ms` and `mJ`. Native output prints six decimal places; the adapter preserves that precision and does not invent extra digits.

These metrics exclude host application work such as BMP parsing, adjacency construction, KNN voting, and verification. Native wall time and any author `Host elapsed time` are preserved in logs but excluded from modeled metrics and deterministic replay checks. Consequently, the study makes no end-to-end CPU/GPU-versus-PIM speedup claim.

The main nine cases use the explicit `bounded-functional` profile: one rank, four banks, four subarrays per bank, 1,024 rows, 8,192 columns, 16 MiB modeled capacity, one simulator worker. Architecture changes do not change these geometry values. This profile supports bounded functional validation; it is not the paper's performance setup.

The `paper-rank1` profile denotes the **unchanged configuration file distributed in the author release**. It does not certify that every parameter agrees with the prose of the publication. For example, these release configurations contain 8,192 rows per subarray while the paper's DRAM-background description uses 1,024. Source, full configuration bytes and hashes are retained, and profiles receive distinct platform identities. Cross-architecture comparisons are rejected by Foundry's same-platform comparison contract.

## Independent failure evidence

### BMP metadata accepted as pixels

The unchanged author `sample1.bmp` has a BITMAPV5 header, width 640, height 427, and 819,840 pixel-storage bytes. It also contains a 3,144-byte ICC profile after pixel storage. There are 273,280 image pixels. The original histogram reports 274,328 pixels: the 1,048 extra entries come from interpreting the ICC bytes as BGR triples. Its CPU verifier reports `Correct!` because it uses the same file-end boundary.

The application result records numerical-reference agreement as passed and task-domain correctness as failed. The common application layer records `quality_failed`, rejects even a self-comparison of this result, and reproduces the same failure on frozen replay. Row padding is also checked; it is not treated as image pixels.

### KNN target-label leakage

Both `runKNN` and its CPU verifier iterate over every input column, including the final label column. A separate audit executable includes the unchanged author source and calls the original functions. It supplies two training points:

- feature 0, label 0;
- feature 9, label 100.

With k = 1, query feature 1 has feature-only Manhattan distances 1 and 8. The correct feature-only prediction is label 0. Changing only the query's label field from 0 to 100 changes the original PIM prediction from 0 to 100. The author verifier accepts both. This counterexample is reproduced on all three architectures, with no voting or distance tie.

The ordinary KNN integration case therefore reports only agreement with the author reference, and leaves task quality unevaluated. It is ineligible for a quality-preserving comparison. The independent negative-control report marks the label-invariance test as failed. No original algorithm repair is mixed into reproduction evidence.

### Triangle-count host allocation independent of graph size

`run_adjlist` allocates two `unsigned int` vectors with `operandMaxNumberOfWords` elements, derived from the entire modeled device capacity. Under the unchanged release Rank1 configuration, each vector has 2,147,483,648 elements: **16 GiB combined**, even for the author's 20-vertex fixture. Initial native triangle-count attempts were terminated by signal 9; the logs do not independently establish the kernel's reason for that signal. Source inspection establishes the excessive allocation. A histogram rerun under the large release geometry was also terminated by signal 9 during concurrent validation, although an earlier identical-profile run completed. The final reproducible functional suite therefore uses bounded geometry for every native application, including the unchanged original BMP counterexample; it does not retry the resource-heavy configuration.

Preflight now rejects triangle count with `paper-rank1` before launching the process. The bounded profile keeps the original algorithm unchanged and checks the resulting count against an independent oracle. Sparse vertex IDs that exceed the source's bitmap representation, fewer than ten vertices that would trigger a zero progress interval, and self-loops are rejected explicitly.

## Reproduce

From the repository root, after cloning the pinned author release into `external/pimeval-pimbench`:

```sh
python3 scripts/evaluate_pimbench_application.py --out runs/pimeval-application-new
```

The script builds only the needed library and applications with at most four compiler jobs, captures build output, binds source/config/binary hashes, generates fixed inputs, runs the nine cases, compares independent direct observations, replays application bundles, executes the two negative controls, and verifies all 20 application bundles. The original author checkout remains unmodified.

To add a workload, use a semantic application manifest with `benchmark_definition` (`role=model`) and explicit input assets. The adapter accepts only the three audited applications and fixed native argument shapes; JSON cannot inject arbitrary commands. Extend the adapter with input validation, a scoped metric contract and actual reference evidence before admitting another application.

The standard-library regression suite checks the BMP false-positive case, KNN quality boundary, independent triangle counts, malformed inputs, missing native verification markers, source/config bindings and the preflight resource guard:

```sh
python3 -m unittest discover -s tests -p test_pimbench_adapter.py -v
```

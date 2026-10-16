> Paths under `runs/` or `external/` refer to locally generated or separately obtained artifacts, not bundled files. Start with [the public quickstart](getting-started.md); see [validation scope](../VALIDATION.md) for historical observations.

# IANUS artifact discovery

The original simulator, configurations and raw measurements are needed to re-run IANUS's reported numbers. This search located the paper and a later independent simulator exposing an IANUS allocation scheme, but **did not locate an authenticated public artifact from the IANUS authors**. No IANUS author artifact was executed. This is a bounded discovery result, not evidence that code was never released or that the reported results are false.

Checked 2026-10-15 UTC. Machine-readable request timestamps, response hashes, repository metadata, links and access errors are in discovery.json (local artifact: `../runs/ianus-validation/discovery.json`).

## Identity and coverage

Target: *IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System*, ASPLOS 2024, [DOI 10.1145/3620666.3651324](https://doi.org/10.1145/3620666.3651324), [author preprint v1](https://arxiv.org/html/2410.15008v1). The paper describes GDDR6 AiM; an HBM configuration would change the evaluated memory system.

| Source checked | Observation |
| --- | --- |
| arXiv abstract and v1 full text | No artifact, supplement, or code release identified. GitHub links are arXiv rendering/support infrastructure. The generic “Code, Data and Media” widgets do not establish code availability. |
| [ASPLOS official program](https://www.asplos-conference.org/asplos2024/main-program/index.html) | Paper entry located; no runnable artifact identified from that entry. |
| [KAIST publication record](https://pure.kaist.ac.kr/en/publications/ianus-integrated-accelerator-based-on-npu-pim-unified-memory-syst/) | Publication DOI and Scopus links, no code link identified. |
| [John Kim laboratory publications](https://icn.kaist.ac.kr/index.php/publications/) and [IMSI laboratory page](https://imsilab.github.io/imsi/) | Accessible; no IANUS release located. Absence on these particular pages is not an exhaustive author-site audit. |
| GitHub public repository API | `IANUS PIM`, `IANUS NPU`, `IANUS ASPLOS`: zero results each. `IANUS in:name`: 73 metadata records, no authenticated author artifact identified. `"IANUS" "PIM" in:readme`: 13 repositories, including paper lists and PHSim. DOI fragment search: zero results. This was repository/README search, not authenticated global code search. |
| Zenodo records API | `"IANUS" AND "PIM"`: zero records. |
| Crossref DOI metadata | Publication link; empty related-object mapping. This does not prove the absence of supplementary material. |
| ACM landing page and supplement endpoint | HTTP 403. Supplement availability remains unverified. |
| SNU CAPP lab site | Connection reset. Site contents remain unverified. |

Broader indexed searches covered the exact title, DOI, authors, “artifact”, “code”, “GitHub”, “Zenodo” and “Figshare”. Same-name archaeology, HTTP/Gemini, MRI and earlier FPGA spin-system projects are unrelated. Search results for another researcher named Minseok Seo are not author-identity evidence.

## Independent executable candidate

[PHSim](https://github.com/Yueq-Zhang/PHSim), revision [`18c424c80eadecc809bce3e3d9c1e90b54f4d2ae`](https://github.com/Yueq-Zhang/PHSim/commit/18c424c80eadecc809bce3e3d9c1e90b54f4d2ae), exposes an `IANUS` allocation option and a small Nano configuration. Its README describes a Linux C++17/CMake simulator with BLAS/Boost and a bundled NewtonSim backend. It identifies itself as an AICAS 2026 project; that publication claim was not independently established during discovery. Repository metadata reports creation on 2026-08-09.

No affiliation connecting this repository to the IANUS authors was established. Any run must be labeled **third-party implementation**, with mapping/scheduling correspondence audited against the paper. It cannot be counted as execution of the original IANUS artifact or reproduction of a paper figure merely because an option is named `IANUS`.

The README also states that its data container tracks data movement without numerically evaluating GEMM/GEMV. Its sampling and decode-pruning options require separate fidelity checks. The later bounded execution below establishes build and smoke-run feasibility; numerical correctness, paper-result fidelity and hardware accuracy remain unverified.

## Bounded third-party execution

The pinned PHSim source was cloned into `external/phsim` and remained unchanged (`git status --short` empty). Its MIT license and bundled dependency notices were inspected. The original build first failed because Boost headers were absent. The official Ubuntu `libboost1.83-dev` package was downloaded, checked against SHA256 `519ecf2c64308527e15b6582955681d192a832250baa4bc424967aaf7d02d68f`, and extracted inside the study directory. No system packages were installed. A global include-path flag was required because NewtonSim also includes Boost without linking the CMake Boost header target. Release compilation then completed successfully; this build uses the upstream Release assertion settings, without a sanitizer pass.

Results and commands are under phsim-thirdparty (local artifact: `../runs/ianus-validation/phsim-thirdparty/execution-status.json`):

| Run | Change and scope | Outcome |
| --- | --- | --- |
| Author smoke fixture | Original `tests/fixtures/smoke/simulation_cycle_accurate.json`, DASH layout, Add operation; unchanged | Exit 0 |
| IANUS-layout GEMV | Derived from that fixture; changed only `allocation_scheme` from `DASH` to `IANUS` and `test_single_op_name` from `add` to `pim_gemv`. Original asset paths were resolved explicitly. One 64×64 GEMV on the smoke LPDDR5 platform. | Exit 0 |
| Exact same IANUS-layout case again | Same configuration and executable | Exit 0; ten non-log output files byte-identical, timestamped logs retained separately |

The derived run reports 593 stage core cycles, request completion at core cycle 595, and 476 DRAM cycles at 800 MHz. These counters have different boundaries; they are not interchangeable. Each of two DRAM channels records 8 COMP, 4 GWRITE and 2 READRES commands. These are observed simulator counters, not independently established hardware latency or proof of a numerically correct GEMV.

Source inspection shows that `weight_2D_allocate_in_IANUS()` in `src/common_function.cpp:2030` places weight columns consecutively inside banks. `src/operations/PIMGEMV.cc:175` and `src/tensor/MyTensor.cpp` specialize access generation. The shared NewtonSim command queue (`ext/NewtonSim/src/newton_command_queue.cc:56`) waits for normal command queues to drain before issuing PIM commands and blocks new normal commands while PIM commands remain. The application scheduler chooses NPU versus PIM decode through the generic path in `src/Scheduler/MyScheduler.cpp:291`.

No IANUS-specific PAS implementation or its paper correspondence was established in this bounded inspection. No checked-in `configs/` or `tests/` JSON selected `IANUS`. The advertised Nano example selects DASH, LPDDR5 and a prefill-only workload; running it unchanged would not validate the original IANUS system. The successful test above is therefore **a third-party placement/command-generation smoke test**, with no original-paper reproduction claim.

## Required evidence before a reproduction claim

1. Establish artifact ownership, exact revision, license and configuration provenance.
2. Bind each measured or simulated number to model dimensions, precision, request lengths, batch size, operator coverage and accounting boundary.
3. Re-run original and integrated entry points with matching inputs; independently validate arithmetic and workload counts.
4. Audit baseline optimization, resource budgets, memory-access/PIM contention and excluded costs.
5. Keep prototype measurements, simulated predictions and any independent reconstruction separate. A missing artifact or an unexplained discrepancy is a reproducibility limitation, not a finding of fabrication.

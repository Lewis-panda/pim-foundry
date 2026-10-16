# Follow one application across four placements

The first projection produces 96 bytes of INT32 activations. In the mixed placement, SRAM sends those bytes to the GPU, which applies ReLU and explicitly casts them to 48 bytes of FP16. HBM consumes that array and returns 24 bytes of FP16 logits. The GPU converts them to FP32, computes softmax, and returns 48 output bytes to the host. Changing placement changes these transfers and the resources that can overlap.

## How the system executes

![Application assets feed actual tensor execution; a separate operation profile drives resource scheduling, capacity checking, and inspectable results.](assets/system-operation.svg)

An application fixes the graph, input/reference assets, arrivals, initial weight state, and quality policy. An experiment fixes devices, costs, capacity, links, and placement. Functional execution produces real arrays on the host. The scheduler predicts a separate compute/transfer timeline from the resulting operation profiles. A numeric pass does not calibrate those costs.

## Four supported combinations

![Four active placements on a shared declared platform: GPU, HBM plus GPU, HBM plus NPU, and SRAM plus HBM plus GPU.](assets/supported-architectures.svg)

These panels show selected directed links and active devices. Every gallery experiment declares the **same complete SRAM/HBM/GPU/NPU platform**; unused devices are omitted from the drawings. The comparison therefore changes placement only. The logical connectivity is not a die layout, a physical HBM interconnect specification, or evidence that any vendor device has this topology.

All four portable runs use `analytical` backends with identical illustrative coefficients. Compatible matrix operations can instead use the existing native SRAM or HBM adapters after their runtimes are separately built; this gallery does not execute those adapters. See [native setup](native-quickstart.md) and [the backend contract](tensor-platform.md).

## Actual graph and data movement

Each request contains a batch of two groups, each with three rows and four input features. There are two output scores per row. This small fixture makes individual transfers inspectable; it is not a large-model performance benchmark.

| Node | Inputs → output | Mixed placement | Output bytes per request |
|---|---|---|---:|
| Integer projection | INT8 `[2,3,4]` × INT8 `[4,4]` → INT32 `[2,3,4]` | SRAM | 96 |
| ReLU | INT32 `[2,3,4]` → INT32 `[2,3,4]` | GPU | 96 |
| Cast | INT32 `[2,3,4]` → FP16 `[2,3,4]` | GPU | 48 |
| FP16 projection | FP16 `[2,3,4]` × FP16 `[4,2]` → FP16 `[2,3,2]` | HBM | 24 |
| Cast | FP16 `[2,3,2]` → FP32 `[2,3,2]` | GPU | 48 |
| Softmax | FP32 `[2,3,2]` → FP32 `[2,3,2]` | GPU | 48 |

The independent reference uses Python scalar loops, explicit binary16/binary32 rounding, and `math.exp`. It shares no tensor operator implementation with the framework. All four mappings match this reference with maximum absolute output error **2.9802322387695312e-08**, under `atol=1e-7`, `rtol=1e-6`. This checks numerical behavior on generated inputs; there is no task dataset or task-quality claim.

## Read the recorded schedule

![Three-request schedule showing SRAM, HBM and GPU compute, endpoint DMA reservations, and the four placement results.](assets/architecture-timeline.png)

Download and open the [self-contained interactive explorer](assets/architecture-explorer.html) in a browser, or open the file directly from your checkout. GitHub displays HTML as source; the local viewer needs no server or network. Choose a placement, filter a request, show link resources, and select a bar to inspect its exact recorded event. [Recorded JSON](../validation/architecture-gallery.json) supplies all displayed intervals and metrics.

Three identical requests arrive at `t=0`, with `max_inflight=3`. Their initially empty weight cache is shared: the first request uploads each weight, later requests reuse the same bytes. One actual functional profile per mapping predicts this request stream; the three requests are not independent measured performance samples.

| Active placement | Stream makespan, inflight 3 | Serial-admission counterfactual, inflight 1 | Bytes across links | Peak live device bytes |
|---|---:|---:|---:|---|
| GPU | 254.5 ns | 283.5 ns | 248 B | GPU 272 |
| HBM + GPU | 266 ns | 327 ns | 464 B | HBM 88; GPU 280 |
| HBM + NPU | 266 ns | 327 ns | 464 B | HBM 88; NPU 280 |
| SRAM + HBM + GPU | 225 ns | 360 ns | 752 B | SRAM 232; HBM 112; GPU 288 |

In the mixed schedule, request 2 starts its SRAM projection at 28.5 ns while request 1 transfers its completed SRAM output to the GPU. These activities can overlap because they reserve `compute:sram` and `dma:sram` separately. Two transfers sharing the GPU DMA resource must wait for each other. The graph and event list make that distinction inspectable.

The mixed placement has more traffic than the all-GPU mapping, while its separate compute resources permit overlap under these coefficients. HBM+GPU and HBM+NPU give equal results because their rates, links, and placements are symmetric. These are consequences of the declared illustrative model; they establish no physical ranking between SRAM, HBM, GPU, and NPU.

Traffic has an independent byte-count check. Three input/output pairs cost `3 × (24 + 48) = 216 B`; the two warm weights add `16 + 16 = 32 B`. HBM split placements add `3 × (48 + 24) = 216 B`. The mixed placement also adds `3 × 96 = 288 B` for the SRAM output. Hence totals are 248, 464, and 752 B. The first weight uploads are included.

## Explicit assumptions and limits

All four devices use 4 KiB modeled capacity, 16 MAC/ns, 16 elements/ns, 64 local bytes/ns, and 10 ns per-operation launch cost. Each directed link has 16 bytes/ns bandwidth and 5 ns fixed latency. These are synthetic coefficients, chosen for an inspectable example. Energy/idle-power coefficients are zero assumptions and no energy result is claimed.

Each transfer reserves its directed link plus **both endpoint DMA resources**. Opposite directions share endpoint DMA and can serialize even though their link resources differ. Multi-hop routing uses deterministic shortest-hop paths with full-tensor store-and-forward; route choice is not bandwidth- or congestion-optimal. Compute uses a separate exclusive resource per device.

Memory includes live tensor copies, resident warm weights, outputs, and workspace. The schedule records peak bytes, not complete allocation histories. Capacity excess is rejected after schedule construction; there is no eviction or automatic memory-aware admission. The gallery deliberately checks that missing routes and a one-byte capacity fail.

All four sealed bundles verify and replay. Comparisons retain identical task/platform identities. Same-model comparison is eligible; task-quality-preserving and hardware-performance comparison remain ineligible for this generated-input example.

## Run and change the example

From a checkout with the tensor extra already installed:

```sh
python examples/architecture_gallery.py --out runs/architecture-gallery
```

Use a new output directory. The command generates `fixture/graph.json`, portable input/reference assets, `platform.json`, four placement experiments, sealed runs, replays, and `summary.json`. No upstream artifacts, native runtimes, downloads, or new dependencies beyond NumPy are needed.

Edit one generated `experiment.json` to change placement, link bandwidth, or device capacity, then execute it through the existing application interface:

```sh
python -m pim_foundry app-run \
  --application runs/architecture-gallery/fixture/application.json \
  --experiment runs/architecture-gallery/sram_hbm_gpu/experiment.json \
  --out runs/my-architecture-change
```

Changing devices or cost coefficients changes platform identity; treat it as a different modeled platform. Changing the graph, input, precision, or arrival policy changes the task. Use [the comparison protocol](comparison.md) to choose the appropriate comparison scope.

With Matplotlib already available, regenerate the static timeline and local explorer from any gallery summary:

```sh
python scripts/render_architecture_gallery.py \
  --summary runs/architecture-gallery/summary.json --out runs/architecture-gallery/figures
```

The checked-in [summary](../validation/architecture-gallery.json) records the example source SHA-256 and NumPy version. Its bundle paths are relative to a newly generated run directory; those bundles are not committed. Source: [example](../examples/architecture_gallery.py), [renderer](../scripts/render_architecture_gallery.py), [runtime and scheduling contract](tensor-platform.md).

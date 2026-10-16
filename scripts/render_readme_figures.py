#!/usr/bin/env python3
"""Render the README's measured validation summary from checked-in evidence.

Sources:
  validation/public-onnx.json: clean-checkout regression outcomes.
  validation/local-v0.8.json: historical local DistilGPT2 observations.
  docs/transformer-application.md: numerical policy and fixed-fixture scope.
  docs/neupims-validation.md: historical controlled channel-assignment ablation.

The latter observations are an author-provided historical summary, not an
independent attestation; raw historical bundles are not distributed. The plot
contains functional/numerical checks, not measured hardware performance.

Run from any directory with Python 3.10+ and Matplotlib (rendered with 3.10.0):
  python -m pip install matplotlib==3.10.0
  python scripts/render_readme_figures.py

PNG/SVG exports use fixed metadata and SVG IDs. Exact raster bytes can differ
with Matplotlib, FreeType, or font versions; the observations are never sampled.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pim-foundry-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
NAVY = "#152A43"
MUTED = "#52667B"
TEAL = "#087F8C"
AMBER = "#E7B75F"
LINE = "#DFE7EE"
TRACK = "#E9EFF4"


def load_evidence() -> dict:
    public = json.loads((ROOT / "validation/public-onnx.json").read_text())
    historical = json.loads((ROOT / "validation/local-v0.8.json").read_text())
    policy = (ROOT / "docs/transformer-application.md").read_text()
    match = re.search(
        r"absolute/relative logit tolerance (\d+\.\d+) and maximum NLL change (\d+\.\d+)",
        policy,
    )
    if match is None:
        raise ValueError("The documented numerical policy changed; review the chart policy.")
    model = historical["distilgpt2"]
    checks = model["quality"]["numerical"]["checks"]
    caches = [value for name, value in checks.items() if re.fullmatch(r"cache_\d+_[kv]", name)]
    if public["failures"] or public["errors"] or not public["passed"]:
        raise ValueError("The release validation figure requires a successful regression report.")
    if not checks["greedy_tokens"]["passed"] or not all(c["passed"] for c in caches):
        raise ValueError("Token/cache agreement changed; review the result presentation.")
    discovered = public["discovered"]
    if public["passed_tests"] + len(public["skipped"]) != discovered:
        raise ValueError("Regression totals do not reconcile.")
    return {
        "passed": public["passed_tests"],
        "skipped": len(public["skipped"]),
        "discovered": discovered,
        "python": ".".join(public["python"].split(".")[:2]),
        "generated": model["metrics"]["generated_tokens"]["value"],
        "targets": model["metrics"]["scored_tokens"]["value"],
        "caches": len(caches),
        "logit_error": checks["evaluation_logits"]["max_abs_error"],
        "nll_delta": model["quality"]["task"]["absolute_delta"],
        "logit_atol": float(match.group(1)),
        "nll_limit": float(match.group(2)),
    }


def main() -> None:
    evidence = load_evidence()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "text.color": NAVY,
        "svg.fonttype": "none",
        "svg.hashsalt": "pim-foundry-readme-v0.8.0",
    })
    fig = plt.figure(figsize=(16, 8.4), facecolor="#F4F7FB")

    def text(x, y, label, size=13, color=NAVY, weight="normal", **kwargs):
        return fig.text(x, y, label, fontsize=size, color=color, weight=weight, **kwargs)

    def rect(x, y, width, height, color, *, rounded=False, edgecolor="none", zorder=1):
        cls = FancyBboxPatch if rounded else Rectangle
        extra = {"boxstyle": "round,pad=0.0,rounding_size=0.016"} if rounded else {}
        patch = cls((x, y), width, height, transform=fig.transFigure,
                    facecolor=color, edgecolor=edgecolor, linewidth=0.8,
                    zorder=zorder, **extra)
        fig.add_artist(patch)

    text(0.035, 0.943, "PIM FOUNDRY  /  VALIDATION", 13, TEAL, "bold")
    text(0.035, 0.867, "Evidence you can inspect.", 32, weight="bold")
    text(0.035, 0.816, "Software regression and pretrained numerical agreement, with explicit evaluation scope.", 15, MUTED)

    rect(0.035, 0.185, 0.300, 0.59, "white", rounded=True, edgecolor=LINE)
    rect(0.355, 0.185, 0.610, 0.59, "white", rounded=True, edgecolor=LINE)

    text(0.057, 0.726, "Public regression suite", 18, weight="bold")
    text(0.057, 0.688, "Clean checkout · ONNX profile", 12, MUTED)
    text(0.057, 0.571, str(evidence["passed"]), 56, TEAL, "bold")
    text(0.180, 0.586, "passed", 20, TEAL)

    fraction = evidence["passed"] / evidence["discovered"]
    rect(0.057, 0.508, 0.255, 0.024, TRACK)
    rect(0.057, 0.508, 0.255 * fraction, 0.024, TEAL)
    rect(0.057 + 0.255 * fraction, 0.508, 0.255 * (1 - fraction), 0.024, AMBER)
    rect(0.057, 0.458, 0.009, 0.017, AMBER)
    text(0.074, 0.458, f'{evidence["skipped"]} optional tests skipped', 13, MUTED)
    text(0.057, 0.412, "0 failures · 0 errors", 16, weight="bold")
    rect(0.057, 0.369, 0.255, 0.0012, LINE)
    text(0.057, 0.325, f'{evidence["discovered"]} tests discovered · Python {evidence["python"]}', 12, MUTED)
    text(0.057, 0.272, "Skipped checks require separate", 12, MUTED)
    text(0.057, 0.239, "author sources or native runtimes.", 12, MUTED)

    text(0.378, 0.726, "DistilGPT2 reference agreement", 18, weight="bold")
    text(0.378, 0.688, "Historical local run · CPU float32 · fixed WikiText fixture", 12, MUTED)
    text(0.378, 0.619, f'{evidence["generated"]}/{evidence["generated"]}', 29, TEAL, "bold")
    text(0.440, 0.627, "greedy token IDs match", 13)
    text(0.663, 0.619, f'{evidence["caches"]}/{evidence["caches"]}', 29, TEAL, "bold")
    text(0.757, 0.627, "K/V tensors pass", 13)
    rect(0.378, 0.582, 0.564, 0.0012, LINE)

    def metric(y, label, observed, limit, observed_label, limit_label):
        text(0.378, y + 0.053, label, 13, weight="bold")
        text(0.599, y + 0.053, observed_label, 16, TEAL, "bold")
        text(0.942, y + 0.053, limit_label, 11, MUTED, ha="right")
        left, width = 0.378, 0.564
        rect(left, y + 0.009, width, 0.018, TRACK)
        rect(left, y + 0.009, width * observed / limit, 0.018, TEAL)
        rect(left + width - 0.001, y + 0.003, 0.0015, 0.031, NAVY)

    metric(0.467, "Max evaluation logit |Δ|", evidence["logit_error"], evidence["logit_atol"],
           f'{evidence["logit_error"]:.2e}', "absolute tolerance 3e−4")
    metric(0.357, "Mean next-token NLL |Δ|", evidence["nll_delta"], evidence["nll_limit"],
           f'{evidence["nll_delta"]:.2e}', "limit 1e−4 nats/token")
    text(0.378, 0.304, "Bars show observed error relative to the marked bound.", 11, MUTED)
    text(0.378, 0.266, f'{evidence["targets"]} scored targets · logits use atol = rtol = 3e−4', 12, MUTED)
    text(0.378, 0.229, "Quality preservation on this fixture; no general model-quality claim.", 11, MUTED)

    text(0.035, 0.128, "Scope: functional correctness and numerical agreement; no measured hardware speedup.", 12, weight="bold")
    text(0.035, 0.087, "Sources: validation/public-onnx.json · validation/local-v0.8.json", 11, MUTED)
    text(0.035, 0.052, "Policy: docs/transformer-application.md · Historical raw bundles are not distributed; no independent attestation.", 10.5, MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    description = (
        "PIM Foundry validation: sources validation/public-onnx.json and "
        "validation/local-v0.8.json; policy docs/transformer-application.md. "
        "493 passed, 25 optional skips, 518 discovered. Historical DistilGPT2 "
        "fixture: 8/8 greedy tokens, 12/12 K/V tensors, 128 scored targets. "
        "Functional and numerical evidence only, not measured hardware speedup. "
        "Historical raw bundles are not distributed; no independent attestation."
    )
    fig.savefig(OUT / "validation-results.png", dpi=160,
                metadata={"Software": "PIM Foundry", "Description": description})
    fig.savefig(OUT / "validation-results.svg",
                metadata={"Date": None, "Creator": "PIM Foundry", "Description": description})
    plt.close(fig)
    print(OUT / "validation-results.png")
    print(OUT / "validation-results.svg")
    render_neupims()


def render_neupims() -> None:
    """Plot the same-platform ablation; deliberately exclude NPU-only timing."""
    source = (ROOT / "docs/neupims-validation.md").read_text()

    def extract(variant: str) -> int:
        match = re.search(
            rf"\| NeuPIMs, {re.escape(variant)} assignment \| ([0-9,]+) \| ([0-9,]+) ns \|",
            source,
        )
        if match is None or match.group(1) != match.group(2):
            raise ValueError(f"Missing or inconsistent NeuPIMs observation: {variant}")
        return int(match.group(2).replace(",", ""))

    rr_ns = extract("regenerated RR")
    clb_ns = extract("author CLB")
    decrease = 100 * (1 - clb_ns / rr_ns)
    fig = plt.figure(figsize=(16, 6.2), facecolor="#F4F7FB")
    fig.text(0.035, 0.926, "NEUPIMS  /  ASPLOS 2024 ARTIFACT", fontsize=13, color=TEAL, weight="bold")
    fig.text(0.035, 0.823, "Change placement. Keep the workload fixed.", fontsize=29, color=NAVY, weight="bold")
    fig.text(0.035, 0.750, "Same 512 ordered requests and platform; only the channel assignment changes.",
             fontsize=15, color=MUTED)

    ax = fig.add_axes((0.205, 0.360, 0.540, 0.300), facecolor="#F4F7FB")
    values = [rr_ns / 1000, clb_ns / 1000]
    ax.barh([1, 0], values, height=0.46, color=["#95A8BA", TEAL], zorder=3)
    ax.set_yticks([1, 0], ["Regenerated RR", "Author CLB"], fontsize=14, color=NAVY)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(0, 750)
    ax.set_xticks([0, 200, 400, 600])
    ax.tick_params(axis="both", length=0, pad=10, labelcolor=MUTED, labelsize=13)
    ax.set_xlabel("Modeled batch latency (µs; lower is better)", fontsize=13, color=MUTED, labelpad=13)
    ax.grid(axis="x", color=LINE, linewidth=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    for y, value in zip([1, 0], values):
        ax.text(value + 10, y, f"{value:.3f}", va="center", fontsize=15, weight="bold", color=NAVY)

    fig.add_artist(FancyBboxPatch(
        (0.795, 0.342), 0.170, 0.303,
        boxstyle="round,pad=0,rounding_size=0.016", transform=fig.transFigure,
        facecolor="#E3F2F2", edgecolor="none", zorder=1,
    ))
    fig.text(0.880, 0.540, f"{decrease:.2f}%", ha="center", fontsize=33, color=TEAL, weight="bold")
    fig.text(0.880, 0.482, "lower modeled latency", ha="center", fontsize=12.5, color=NAVY)
    fig.text(0.880, 0.406, "CLB vs regenerated RR", ha="center", fontsize=11.5, color=MUTED)

    fig.text(0.035, 0.191, "Measurement boundary: one layer · one decode token · batch completion including simulator drain.",
             fontsize=12, color=NAVY, weight="bold")
    fig.text(0.035, 0.125, "Source: docs/neupims-validation.md · Historical local simulator study; raw run bundles are not distributed.",
             fontsize=11, color=MUTED)
    fig.text(0.035, 0.072, "Controlled ablation of the released artifact; task quality and physical hardware performance are unvalidated.",
             fontsize=11, color=MUTED)

    description = (
        "Source: docs/neupims-validation.md. Historical local controlled NeuPIMs "
        f"ablation: regenerated RR {rr_ns} ns, author CLB {clb_ns} ns, "
        f"{decrease:.4f}% lower modeled latency. Same ordered 512 requests and "
        "platform; one layer, one decode token, includes simulator drain. "
        "Zero-based latency axis. Raw historical run bundles are not distributed. "
        "Task quality and physical hardware performance are unvalidated."
    )
    fig.savefig(OUT / "neupims-comparison.png", dpi=160,
                metadata={"Software": "PIM Foundry", "Description": description})
    fig.savefig(OUT / "neupims-comparison.svg",
                metadata={"Date": None, "Creator": "PIM Foundry", "Description": description})
    plt.close(fig)
    print(OUT / "neupims-comparison.png")
    print(OUT / "neupims-comparison.svg")


if __name__ == "__main__":
    main()

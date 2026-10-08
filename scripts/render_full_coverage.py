#!/usr/bin/env python3
"""Render the full-coverage ContractNLI quality, time and cache accounting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, BLUE, TEAL, GRID = "#16243b", "#627086", "#7896c3", "#008f7a", "#edf1f6"


def load_summary(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("workload", {}).get("decisions") != 1088:
        raise ValueError("expected the schema-v1 full-coverage ContractNLI summary")
    return payload


def annotate_bars(axis, bars, values, *, suffix="", decimals=1):
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{value:.{decimals}f}{suffix}", ha="center", va="bottom",
                  color=INK, fontsize=10, weight="bold")


def render(summary, out):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#dce3ed", "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold",
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    quality = summary["quality"]
    performance = summary["fixed_work_performance"]
    cache = summary["cache_accounting"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.05))

    accuracy = [quality["original"]["accuracy"] * 100, quality["solo"]["accuracy"] * 100]
    bars = axes[0].bar(["Original", "SOLO"], accuracy, color=[BLUE, TEAL], width=.62)
    axes[0].set_ylim(0, 85)
    axes[0].set_ylabel("Accuracy against gold labels (%)")
    axes[0].set_title("Task quality", loc="left", pad=12)
    axes[0].yaxis.grid(True, color=GRID, zorder=0)
    annotate_bars(axes[0], bars, accuracy, suffix="%")
    axes[0].text(.5, -.18, "+7.72 percentage points\npaired exact p = 8.09e-8",
                 transform=axes[0].transAxes, ha="center", va="top", color=MUTED, fontsize=9)

    wall = [performance["original_wall_seconds_sum"], performance["solo_wall_seconds_sum"]]
    bars = axes[1].bar(["Original", "SOLO"], wall, color=[BLUE, TEAL], width=.62)
    axes[1].set_ylim(0, max(wall) * 1.18)
    axes[1].set_ylabel("Sum of complete-batch wall time (s)")
    axes[1].set_title("Fixed-work completion", loc="left", pad=12)
    axes[1].yaxis.grid(True, color=GRID, zorder=0)
    annotate_bars(axes[1], bars, wall, suffix=" s")
    axes[1].text(.5, -.18, f"{performance['original_over_solo_speedup']:.2f}× faster",
                 transform=axes[1].transAxes, ha="center", va="top", color=MUTED, fontsize=9)

    cache_values = [
        cache["exact_reusable_prefix_fraction"] * 100,
        cache["block_quantized_ceiling_fraction"] * 100,
        cache["measured_cached_fraction"] * 100,
    ]
    labels = ["Exact token\nprefix", "528-token\nblock ceiling", "vLLM\nmeasured"]
    bars = axes[2].bar(np.arange(3), cache_values,
                       color=["#9cb1d0", "#42aa98", TEAL], width=.66)
    axes[2].set_xticks(np.arange(3), labels)
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("Share of complete SOLO prompt tokens (%)")
    axes[2].set_title("Three cache denominators", loc="left", pad=12)
    axes[2].yaxis.grid(True, color=GRID, zorder=0)
    annotate_bars(axes[2], bars, cache_values, suffix="%")
    axes[2].text(.5, -.18, "Measured = 98.1% of block ceiling",
                 transform=axes[2].transAxes, ha="center", va="top", color=MUTED, fontsize=9)

    fig.suptitle("SOLO on all 1,088 ContractNLI decisions", x=.055, y=.98, ha="left",
                 fontsize=17, weight="bold", color=INK)
    fig.text(.055, .89,
             "17 already-ready batches of 64 · JEV-9B BF16 · RTX 4090 · concurrency 4 · APC enabled",
             ha="left", color=MUTED, fontsize=9.5)
    fig.subplots_adjust(left=.065, right=.985, bottom=.26, top=.70, wspace=.36)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png", "pdf"):
        fig.savefig(out.with_suffix(f".{extension}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="output basename without an extension")
    args = parser.parse_args()
    render(load_summary(args.summary), args.out)
    print(f"Wrote {args.out}.svg/.png/.pdf")


if __name__ == "__main__":
    main()

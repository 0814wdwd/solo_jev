#!/usr/bin/env python3
"""Render README figures directly from checked-in measurements.

Run from any directory. No model, tokenizer, GPU, or network is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, BLUE, TEAL = "#16243b", "#627086", "#99afd0", "#008f7a"
LABELS = {"original": "Original", "lexicographic": "Lexicographic", "random": "Random", "cardinality": "NDV", "solo": "SOLO"}


def trials_note(cases):
    counts = {s["repeats"] for c in cases for s in c["summary"]}
    return (f"{next(iter(counts))} cold-namespace trials per layout" if len(counts) == 1
            else "Trial counts: see source report")


def accuracy_note(cases):
    values = [s.get("accuracy") for c in cases for s in c["summary"]]
    if any(v is None for v in values) or not values:
        return "Accuracy: see source report."
    low, high = min(values), max(values)
    return (f"All measured layouts: {low:.0%} task accuracy." if low == high
            else f"Task accuracy: {low:.1%}–{high:.1%} across layouts.")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": "#dce3ed", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "svg.fonttype": "none", "savefig.facecolor": "white"})


def save(fig, out, name):
    for extension in ("svg", "png", "pdf"):
        fig.savefig(out / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def correlated(data, out):
    case = data["cases"][0]
    summary = case["summary"]
    workload = case["workload"]
    if workload.get("family") != "equal_ndv_correlated" or not workload.get("all_column_ndv_equal"):
        raise ValueError("The correlated figure requires an equal-NDV correlated workload report")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.35), gridspec_kw={"width_ratios": [1, 1.03]})
    fig.suptitle("Same NDV. Different structure. Faster decisions.", x=.075, y=1.02, ha="left", fontsize=18, weight="bold")
    y = np.arange(len(summary))
    values = [s["rows_per_second"] for s in summary]
    colors = [TEAL if s["method"] == "solo" else BLUE for s in summary]
    ax.barh(y, values, color=colors, height=.60, zorder=2)
    ax.set_yticks(y, [LABELS.get(s["method"], s["method"]) for s in summary])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.24)
    ax.set_xlabel("Rows / second  ·  higher is better")
    ax.set_title("Measured throughput", loc="left", pad=16)
    ax.xaxis.grid(True, color="#edf1f6", zorder=0)
    for i, s in enumerate(summary):
        ax.text(values[i] + .25, i, f"{values[i]:.2f}", va="center", weight="bold", color=colors[i] if s["method"] == "solo" else INK)
        trials = [r["rows_per_second"] for r in case["runs"] if r["method"] == s["method"]]
        ax.scatter(trials, [i] * len(trials), s=17, color=INK, zorder=3, linewidths=.6, edgecolors="white")
    all_counts = []
    for method, color, dash in (("cardinality", BLUE, "--"), ("solo", TEAL, "-")):
        counts = case["workload"]["prefix_distinct_counts"][method]
        all_counts.extend(counts)
        bx.plot(np.arange(1, len(counts) + 1), counts, marker="o", markersize=4, color=color, linestyle=dash, linewidth=2.6, label=LABELS[method])
    bx.set_title("Why the layout matters", loc="left", pad=16)
    bx.set_xlabel("Number of leading fields")
    bx.set_ylabel("Distinct prefix combinations  ·  fewer share more")
    if len(set(all_counts)) <= 6:
        bx.set_yticks(sorted(set(all_counts)))
    m = workload["columns"]
    bx.set_xticks(sorted({1, max(1, m // 4), max(1, m // 2), max(1, 3 * m // 4), m}))
    bx.set_ylim(0, max(all_counts) * 1.13)
    bx.yaxis.grid(True, color="#edf1f6")
    bx.legend(frameon=False, loc="center right")
    ndv = next(s for s in summary if s["method"] == "cardinality")
    bx.text(.035, .73, f"{ndv['solo_speedup_vs_this_method']:.2f}× faster than NDV", transform=bx.transAxes, fontsize=14, color=TEAL, weight="bold")
    ndv_count = next(iter(workload["column_ndv"].values()))
    description = (f"Synthetic · {workload['rows']:,} rows × {m} fields · every field: NDV {ndv_count}, "
                   f"{workload['tokens_per_field']} tokens · reference server: RTX 4090 / JEV-9B / vLLM 0.31.0")
    note = (f"{trials_note([case])}; bars = rows / median elapsed time, dots = trials. "
            f"Planning and cache fills included. {accuracy_note([case])}")
    fig.text(.075, -.045, description + "\n" + note, color=MUTED, fontsize=9, linespacing=1.5)
    fig.subplots_adjust(wspace=.48, bottom=.15, top=.85)
    save(fig, out, "correlated-benchmark")


def shared(data, out):
    cases = data["cases"]
    if any(c["workload"].get("family") != "shared_fields" for c in cases):
        raise ValueError("The shared figure requires a shared-fields workload report")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.25), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle("Long shared fields make input order matter.", x=.075, y=1.02, ha="left", fontsize=18, weight="bold")
    x = np.arange(len(cases))
    width = .31
    ratios, speeds = [], []
    all_values = []
    for case in cases:
        ratios.append(case["workload"]["reusable_value_token_fraction"] * 100)
        summary = {s["method"]: s for s in case["summary"]}
        speeds.append(summary["solo"]["speedup_vs_original"])
    for method, shift, color in (("original", -width / 2, BLUE), ("solo", width / 2, TEAL)):
        vals = [next(s for s in c["summary"] if s["method"] == method)["rows_per_second"] for c in cases]
        all_values.extend(vals)
        ax.bar(x + shift, vals, width, color=color, label=LABELS[method], zorder=2)
        for i, value in enumerate(vals):
            ax.text(i + shift, value + .16, f"{value:.2f}", ha="center", fontsize=10, weight="bold")
            trials = [r["rows_per_second"] for r in cases[i]["runs"] if r["method"] == method]
            ax.scatter([i + shift] * len(trials), trials, s=15, color=INK, edgecolor="white", linewidth=.5, zorder=3)
    ax.set_xticks(x, [f"{r:.1f}%".replace(".0%", "%") for r in ratios])
    ax.set_xlabel("Reusable field-value tokens in the generated input")
    ax.set_ylabel("Rows / second")
    ax.set_ylim(0, max(all_values) * 1.2)
    ax.yaxis.grid(True, color="#edf1f6", zorder=0)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_title("Same complete records, reordered", loc="left", pad=16)
    bx.plot(x, speeds, color=TEAL, linewidth=3, marker="o", markersize=9)
    bx.axhline(1, color=BLUE, linestyle="--", linewidth=1)
    for i, value in enumerate(speeds):
        bx.text(i, value + .32, f"{value:.2f}×", ha="center", color=TEAL, weight="bold", fontsize=16)
    bx.set_xticks(x, [f"{r:.1f}%".replace(".0%", "%") for r in ratios])
    bx.set_xlabel("Reusable field-value tokens")
    bx.set_ylabel("SOLO speedup over original order")
    bx.set_ylim(0, max(speeds) * 1.22)
    bx.set_xlim(-.3, len(cases) - .7)
    bx.yaxis.grid(True, color="#edf1f6")
    bx.set_title("End-to-end acceleration", loc="left", pad=16)
    row_counts = {c["workload"]["rows"] for c in cases}
    rows_note = (f"{next(iter(row_counts)):,} rows per case" if len(row_counts) == 1
                 else "row counts: see source report")
    description = (f"Synthetic · {rows_note} · ID-first baseline · reference server: RTX 4090 / JEV-9B / vLLM 0.31.0 "
                   f"· concurrency {data['configuration']['concurrency']}")
    note = (f"{trials_note(cases)}; planning and cache fills included. "
            "Reuse is a workload property, not the cache-hit rate. Padding is intentional.")
    fig.text(.075, -.045, description + "\n" + note, color=MUTED, fontsize=9, linespacing=1.5)
    fig.subplots_adjust(wspace=.37, bottom=.15, top=.85)
    save(fig, out, "shared-benchmark")


def overview(out):
    # An illustration of token-prefix identity, not a performance measurement.
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 380" role="img" aria-labelledby="title desc">
<title id="title">SOLO changes input layout to expose reusable prefixes</title>
<desc id="desc">Before, each row starts with its unique identifier. After, shared policy and account fields are first, followed by unique values. Every field is still sent, and output decisions are restored to the original row order.</desc>
<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="#71819a" stroke-width="1.5"/></marker></defs>
<rect width="1200" height="380" rx="22" fill="#f5f8fc"/>
<g font-family="DejaVu Sans,Arial,sans-serif" fill="#16243b">
<text x="38" y="47" font-size="23" font-weight="700">Same records. Better layout. More shared computation.</text>
<text x="38" y="79" font-size="14" fill="#627086">A layout optimizer between your data and a prefix-caching decision model</text>
<text x="38" y="121" font-size="16" font-weight="700">Original order</text>
<text x="579" y="121" font-size="16" font-weight="700">SOLO order</text>
<g font-size="14" text-anchor="middle">
<rect x="38" y="139" width="83" height="38" rx="6" fill="#f4d8c6"/><text x="79" y="163">id: 103</text><rect x="127" y="139" width="185" height="38" rx="6" fill="#d7e2f4"/><text x="219" y="163">policy: standard</text><rect x="318" y="139" width="144" height="38" rx="6" fill="#d7e2f4"/><text x="390" y="163">region: north</text>
<rect x="38" y="187" width="83" height="38" rx="6" fill="#f4d8c6"/><text x="79" y="211">id: 201</text><rect x="127" y="187" width="185" height="38" rx="6" fill="#cfece2"/><text x="219" y="211">policy: premium</text><rect x="318" y="187" width="144" height="38" rx="6" fill="#cfece2"/><text x="390" y="211">region: south</text>
<rect x="38" y="235" width="83" height="38" rx="6" fill="#f4d8c6"/><text x="79" y="259">id: 104</text><rect x="127" y="235" width="185" height="38" rx="6" fill="#d7e2f4"/><text x="219" y="259">policy: standard</text><rect x="318" y="235" width="144" height="38" rx="6" fill="#d7e2f4"/><text x="390" y="259">region: north</text>
<rect x="579" y="139" width="185" height="38" rx="6" fill="#d7e2f4"/><text x="671" y="163">policy: standard</text><rect x="770" y="139" width="144" height="38" rx="6" fill="#d7e2f4"/><text x="842" y="163">region: north</text><rect x="920" y="139" width="83" height="38" rx="6" fill="#f4d8c6"/><text x="961" y="163">id: 103</text>
<rect x="579" y="187" width="185" height="38" rx="6" fill="#d7e2f4"/><text x="671" y="211">policy: standard</text><rect x="770" y="187" width="144" height="38" rx="6" fill="#d7e2f4"/><text x="842" y="211">region: north</text><rect x="920" y="187" width="83" height="38" rx="6" fill="#f4d8c6"/><text x="961" y="211">id: 104</text>
<rect x="579" y="235" width="185" height="38" rx="6" fill="#cfece2"/><text x="671" y="259">policy: premium</text><rect x="770" y="235" width="144" height="38" rx="6" fill="#cfece2"/><text x="842" y="259">region: south</text><rect x="920" y="235" width="83" height="38" rx="6" fill="#f4d8c6"/><text x="961" y="259">id: 201</text>
<rect x="1052" y="148" width="110" height="116" rx="12" fill="#fff" stroke="#cad5e3"/><text x="1107" y="187" font-size="15" font-weight="700">Decision</text><text x="1107" y="209" font-size="15" font-weight="700">model</text><text x="1107" y="239" font-size="12" fill="#627086">prefix cache</text>
</g>
<path d="M478,206 L552,206" stroke="#71819a" stroke-width="2" marker-end="url(#arrow)"/><path d="M1016,206 L1041,206" stroke="#71819a" stroke-width="2" marker-end="url(#arrow)"/>
<rect x="574" y="133" width="345" height="98" rx="10" fill="none" stroke="#008f7a" stroke-width="2" stroke-dasharray="5 4"/>
<text x="38" y="305" font-size="13" fill="#627086">Unique fields break shared prefixes early.</text><text x="579" y="305" font-size="13" fill="#008f7a">Repeated prefixes become reusable by the server.</text>
<text x="38" y="350" font-size="14" font-weight="700">Pandas · NumPy · JSON / JSONL</text><text x="430" y="350" font-size="14">Full records in every request</text><text x="805" y="350" font-size="14">Decisions in original row order</text>
</g></svg>'''
    (out / "layout-overview.svg").write_text(svg, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--correlated", type=Path, default=ROOT / "validation/correlated-live.json")
    p.add_argument("--shared", type=Path, default=ROOT / "validation/shared-live.json")
    p.add_argument("--out", type=Path, default=ROOT / "assets")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    overview(args.out)
    correlated(json.loads(args.correlated.read_text()), args.out)
    if args.shared.exists():
        shared(json.loads(args.shared.read_text()), args.out)
    else:
        print(f"Shared benchmark not yet present: {args.shared}")
    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()

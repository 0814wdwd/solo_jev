#!/usr/bin/env python3
"""Render profiling figures from paired cache-enabled/disabled raw reports."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, BLUE, TEAL, GRID = "#16243b", "#627086", "#7896c3", "#008f7a", "#edf1f6"
LABELS = {"original": "Original", "solo": "SOLO"}


def load_report(path, cache_mode):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "complete":
        raise ValueError(f"{path} is not a complete schema-v1 profiling report")
    actual = report.get("configuration", {}).get("cache_mode")
    if actual != cache_mode:
        raise ValueError(f"{path} declares cache_mode={actual!r}, expected {cache_mode!r}")
    return report


def validate_pair(enabled, disabled):
    if enabled["workload"]["name"] != disabled["workload"]["name"]:
        raise ValueError("enabled and disabled reports use different workloads")
    left = enabled["provenance"]
    right = disabled["provenance"]
    for key in ("tokenizer_sha256", "source_sha256"):
        if left.get(key) != right.get(key):
            raise ValueError(f"enabled and disabled reports differ in {key}")
    enabled_batches = {(run["batch_id"], run["batch_size"], run["batch_index"])
                       for run in enabled["batch_runs"]}
    disabled_batches = {(run["batch_id"], run["batch_size"], run["batch_index"])
                        for run in disabled["batch_runs"]}
    if enabled_batches != disabled_batches:
        raise ValueError("enabled and disabled reports do not contain the same input batches")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#dce3ed", "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold",
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def save(fig, out, name):
    for extension in ("svg", "png", "pdf"):
        fig.savefig(out / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def latency_bins(report):
    ranges = ((0, 512, "≤512"), (512, 2048, "513–2K"), (2048, 4096, "2–4K"),
              (4096, 8192, "4–8K"), (8192, float("inf"), ">8K"))
    points = []
    for item in report["single_requests"]:
        request = item["request"]
        points.append({
            "tokens": item["local_prompt_tokens"],
            "client": request["client_request_seconds"],
            "prefill": (request["engine_prefill_interval_ms"] / 1000
                        if request.get("engine_prefill_interval_ms") is not None else None),
            "queue": (request["queue_time_ms"] / 1000
                      if request.get("queue_time_ms") is not None else None),
            "completion_tokens": request.get("completion_tokens"),
        })
    if not points:
        raise ValueError("enabled report has no single-request observations")
    bins = []
    for low, high, label in ranges:
        rows = [point for point in points if low < point["tokens"] <= high]
        if not rows:
            continue
        bins.append({
            "label": label,
            "tokens": float(np.median([row["tokens"] for row in rows])),
            "client_seconds": float(np.median([row["client"] for row in rows])),
            "prefill_seconds": (float(np.median([row["prefill"] for row in rows
                                                  if row["prefill"] is not None]))
                                if any(row["prefill"] is not None for row in rows) else None),
            "queue_seconds": (float(np.median([row["queue"] for row in rows
                                                if row["queue"] is not None]))
                              if any(row["queue"] is not None for row in rows) else None),
            "samples": len(rows),
        })
    return points, bins


def render_latency(report, out, prefix):
    points, bins = latency_bins(report)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.scatter([p["tokens"] for p in points], [p["client"] for p in points],
               s=25, color=BLUE, alpha=.32, edgecolors="none", label="Individual requests")
    x = [row["tokens"] for row in bins]
    client = [row["client_seconds"] for row in bins]
    ax.plot(x, client, marker="o", linewidth=2.6, color=BLUE, label="Median client latency")
    if all(row["prefill_seconds"] is not None for row in bins):
        ax.plot(x, [row["prefill_seconds"] for row in bins], marker="o", linewidth=2.6,
                color=TEAL, label="Median engine first-token interval")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Prompt tokens · natural public-workload requests")
    ax.set_ylabel("Seconds · lower is better")
    ax.set_title("A short decision still has to process its complete input", loc="left", pad=14)
    ax.grid(True, color=GRID, zorder=0)
    ax.legend(frameon=False)
    completion = [p["completion_tokens"] for p in points if p["completion_tokens"] is not None]
    output_note = (f"Measured output: median {np.median(completion):g} token"
                   if completion else "Output-token usage unavailable")
    ax.text(.99, .03, output_note + "\nEngine interval is not pure GPU kernel time.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=MUTED)
    fig.tight_layout()
    save(fig, out, f"{prefix}-decision-profile")
    return bins


def grouped_runs(report):
    grouped = defaultdict(list)
    for run in report["batch_runs"]:
        grouped[(run["batch_size"], run["method"])].append(run)
    return grouped


def median_metric(runs, key):
    values = [run["metrics"].get(key) for run in runs if run["metrics"].get(key) is not None]
    return float(np.median(values)) if values else None


def aggregate_batches(report):
    """Return compact, publication-facing aggregates while raw runs stay archived."""
    rows = []
    for (batch_size, method), runs in sorted(grouped_runs(report).items()):
        agreements = [run.get("agreement_with_original") for run in runs
                      if run.get("agreement_with_original") is not None]
        rows.append({
            "batch_size": batch_size,
            "method": method,
            "runs": len(runs),
            "wall_seconds_median": median_metric(runs, "wall_seconds"),
            "rows_per_second_median": median_metric(runs, "rows_per_second"),
            "cached_fraction_median": median_metric(runs, "cached_fraction"),
            "prompt_tokens_median": median_metric(runs, "prompt_tokens"),
            "accuracy_median": median_metric(runs, "accuracy"),
            "agreement_with_original_median": (
                float(np.median(agreements)) if agreements else None
            ),
        })
    return rows


def paired_speedups(report, baseline="original", candidate="solo"):
    """Summarize within-batch wall-time ratios instead of ratios of medians."""
    indexed = {
        (run["batch_id"], run["batch_size"], run["batch_index"], run["repeat"], run["method"]): run
        for run in report["batch_runs"]
    }
    by_size = defaultdict(list)
    for key, run in indexed.items():
        if key[-1] != baseline:
            continue
        peer = indexed.get((*key[:-1], candidate))
        if peer is None:
            continue
        by_size[key[1]].append(
            run["metrics"]["wall_seconds"] / peer["metrics"]["wall_seconds"]
        )
    return [{
        "batch_size": size,
        "baseline": baseline,
        "candidate": candidate,
        "pairs": len(values),
        "speedup_median": float(np.median(values)),
        "speedup_min": float(np.min(values)),
        "speedup_max": float(np.max(values)),
    } for size, values in sorted(by_size.items())]


def render_causality(enabled, disabled, out, prefix):
    on, off = grouped_runs(enabled), grouped_runs(disabled)
    common_sizes = sorted({size for size, _ in on} & {size for size, _ in off})
    if not common_sizes:
        raise ValueError("paired reports have no common batch size")
    batch_size = common_sizes[-1]
    methods = [method for method in ("original", "solo")
               if (batch_size, method) in on and (batch_size, method) in off]
    if not methods:
        raise ValueError("paired reports need original or SOLO runs")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.8, 4.45))
    x = np.arange(2)
    width = .34
    summary = []
    for index, method in enumerate(methods):
        wall = [median_metric(off[batch_size, method], "wall_seconds"),
                median_metric(on[batch_size, method], "wall_seconds")]
        cached = [median_metric(off[batch_size, method], "cached_fraction"),
                  median_metric(on[batch_size, method], "cached_fraction")]
        color = TEAL if method == "solo" else BLUE
        shift = (index - (len(methods) - 1) / 2) * width
        ax.bar(x + shift, wall, width, color=color, label=LABELS.get(method, method))
        bx.bar(x + shift, [value * 100 if value is not None else 0 for value in cached],
               width, color=color, label=LABELS.get(method, method))
        summary.append({"method": method, "batch_size": batch_size,
                        "cache_off_wall_seconds": wall[0], "cache_on_wall_seconds": wall[1],
                        "cache_off_cached_fraction": cached[0], "cache_on_cached_fraction": cached[1]})
    for axis in (ax, bx):
        axis.set_xticks(x, ["APC off", "APC on"])
        axis.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_ylabel("Median complete-batch time (s)")
    ax.set_title(f"Same {batch_size}-record batches", loc="left", pad=14)
    bx.set_ylabel("Measured cached prompt tokens (%)")
    bx.set_title("Actual input reuse", loc="left", pad=14)
    bx.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.suptitle("Layout × prefix cache: the causal comparison", x=.065, ha="left",
                 fontsize=17, weight="bold")
    fig.subplots_adjust(top=.78, wspace=.34)
    save(fig, out, f"{prefix}-cache-causality")
    return summary


def render_microbatch(enabled, out, prefix):
    grouped = grouped_runs(enabled)
    methods = [method for method in ("original", "solo") if any(key[1] == method for key in grouped)]
    sizes = sorted({key[0] for key in grouped})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.8, 4.4))
    summary = []
    for method in methods:
        color = TEAL if method == "solo" else BLUE
        wall, p95 = [], []
        for size in sizes:
            runs = grouped.get((size, method), [])
            wall.append(median_metric(runs, "wall_seconds"))
            p95.append(median_metric(runs, "batch_sojourn_p95_seconds"))
            summary.append({"method": method, "batch_size": size,
                            "wall_seconds": wall[-1], "batch_sojourn_p95_seconds": p95[-1],
                            "accuracy": median_metric(runs, "accuracy")})
        ax.plot(sizes, wall, marker="o", linewidth=2.6, color=color, label=LABELS.get(method, method))
        bx.plot(sizes, p95, marker="o", linewidth=2.6, color=color, label=LABELS.get(method, method))
    for axis in (ax, bx):
        axis.set_xlabel("Already-ready batch size")
        axis.set_xticks(sizes)
        axis.yaxis.grid(True, color=GRID)
    ax.set_ylabel("Median batch completion time (s)")
    ax.set_title("All decisions restored", loc="left", pad=14)
    bx.set_ylabel("Median trial P95 from batch readiness (s)")
    bx.set_title("Per-record completion", loc="left", pad=14)
    ax.legend(frameon=False)
    fig.suptitle("Microbatch latency with APC enabled", x=.065, ha="left",
                 fontsize=17, weight="bold")
    fig.subplots_adjust(top=.78, wspace=.34)
    save(fig, out, f"{prefix}-microbatch")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled", type=Path, required=True)
    parser.add_argument("--disabled", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("assets/profiling"))
    args = parser.parse_args()
    enabled = load_report(args.enabled, "enabled")
    disabled = load_report(args.disabled, "disabled")
    validate_pair(enabled, disabled)
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    prefix = enabled["workload"]["name"].replace("_", "-")
    summary = {
        "workload": enabled["workload"]["name"],
        "source_reports": {"enabled": args.enabled.name, "disabled": args.disabled.name},
        "latency_bins": render_latency(enabled, args.out, prefix),
        "cache_causality": render_causality(enabled, disabled, args.out, prefix),
        "microbatch": render_microbatch(enabled, args.out, prefix),
        "batch_aggregates": {
            "enabled": aggregate_batches(enabled),
            "disabled": aggregate_batches(disabled),
        },
        "paired_speedups": {
            "enabled": paired_speedups(enabled),
            "disabled": paired_speedups(disabled),
        },
    }
    (args.out / f"{prefix}-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote profiling figures and summary to {args.out}")


if __name__ == "__main__":
    main()

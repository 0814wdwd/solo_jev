#!/usr/bin/env python3
"""Render prefill/first-token and decode timing from a full quality report.

The JEV path emits exactly one decision token. vLLM's per-request metrics call
the scheduled-to-first-output interval ``time_to_first_token_ms`` and the
subsequent inter-token interval ``generation_time_ms``. The latter is therefore
zero for this workload. This renderer keeps those two measurements separate
and derives an additive mean client-time breakdown from the same requests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#16243b"
MUTED = "#627086"
BLUE = "#7896c3"
TEAL = "#008f7a"
GRID = "#edf1f6"
COMPONENT_COLORS = ("#c8d4e5", "#7896c3", "#f0a44b", "#dfe5ed")


TEXT = {
    "en": {
        "title": "SOLO cuts the time before a one-token decision",
        "subtitle": (
            "1,088 ContractNLI decisions · JEV-9B BF16 · RTX 4090 · "
            "concurrency 4 · prefix cache on"
        ),
        "ylabel": "Mean client request time (ms)",
        "original": "Original",
        "solo": "SOLO",
        "components": (
            "Engine queue",
            "Prefill + first decision token",
            "Post-first-token decode",
            "Client / other",
        ),
        "median": "Median prefill / first-token interval",
        "decode": "Post-first-token decode",
        "decode_note": "0.0 ms in all {requests:,} measurements · 1 output token per request",
        "total": "{value:.1f} ms total",
    },
    "zh": {
        "title": "SOLO 减少单 token 决策前的等待",
        "subtitle": (
            "ContractNLI 全部 1,088 条决策 · JEV-9B BF16 · RTX 4090 · "
            "并发 4 · 开启前缀缓存"
        ),
        "ylabel": "单请求平均耗时（毫秒）",
        "original": "原布局",
        "solo": "SOLO",
        "components": (
            "引擎排队",
            "Prefill ＋ 首个决策 token",
            "首 token 后 Decode",
            "客户端及其他",
        ),
        "median": "Prefill／首 token 区间中位数",
        "decode": "首 token 后 Decode",
        "decode_note": "全部 {requests:,} 次测量均为 0.0 ms · 每次请求只输出 1 token",
        "total": "合计 {value:.1f} ms",
    },
}


def load_report(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "complete":
        raise ValueError("expected a complete schema-v1 profiling report")
    runs = report.get("quality_runs") or []
    if {run.get("method") for run in runs} != {"original", "solo"}:
        raise ValueError("quality report must contain original and solo runs")
    return report


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def summarize_requests(requests):
    required = (
        "client_request_seconds",
        "queue_time_ms",
        "engine_prefill_interval_ms",
        "generation_time_ms",
        "completion_tokens",
    )
    missing = [
        (index, key)
        for index, request in enumerate(requests)
        for key in required
        if request.get(key) is None
    ]
    if missing:
        index, key = missing[0]
        raise ValueError(f"request {index} is missing {key}")

    client = np.asarray(
        [request["client_request_seconds"] * 1000 for request in requests], dtype=float
    )
    queue = np.asarray([request["queue_time_ms"] for request in requests], dtype=float)
    first = np.asarray(
        [request["engine_prefill_interval_ms"] for request in requests], dtype=float
    )
    decode = np.asarray([request["generation_time_ms"] for request in requests], dtype=float)
    other = client - queue - first - decode
    if np.min(other) < -1e-6:
        raise ValueError("client time is smaller than its measured engine components")
    other = np.maximum(other, 0)
    completion = np.asarray([request["completion_tokens"] for request in requests], dtype=int)

    def distribution(values):
        return {
            "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)),
            "p95_ms": _percentile(values, 95),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
        }

    return {
        "requests": len(requests),
        "completion_tokens": {
            "sum": int(np.sum(completion)),
            "min_per_request": int(np.min(completion)),
            "max_per_request": int(np.max(completion)),
        },
        "client": distribution(client),
        "queue": distribution(queue),
        "prefill_and_first_token": distribution(first),
        "decode_after_first_token": distribution(decode),
        "client_and_other": distribution(other),
    }


def summarize_report(report):
    runs = {run["method"]: run for run in report["quality_runs"]}
    methods = {
        method: summarize_requests(runs[method]["requests"])
        for method in ("original", "solo")
    }
    original = methods["original"]
    solo = methods["solo"]
    if original["requests"] != solo["requests"]:
        raise ValueError("original and solo contain different request counts")
    if original["completion_tokens"] != solo["completion_tokens"]:
        raise ValueError("original and solo contain different output-token counts")

    first_original = original["prefill_and_first_token"]
    first_solo = solo["prefill_and_first_token"]
    client_original = original["client"]
    client_solo = solo["client"]
    return {
        "schema_version": 1,
        "workload": {
            "name": report["workload"]["source"]["dataset"],
            "decisions_per_layout": original["requests"],
            "layouts": 2,
            "total_request_measurements": original["requests"] + solo["requests"],
        },
        "runtime": {
            "model": "JEV-9B BF16",
            "gpu": report.get("provenance", {}).get("gpu"),
            "concurrency": report["configuration"]["concurrency"],
            "prefix_cache": report["configuration"]["cache_mode"],
            "quality_batch_size": report["configuration"]["quality_batch_size"],
        },
        "definitions": {
            "prefill_and_first_token": (
                "vLLM time_to_first_token_ms: scheduled to first output token; "
                "includes prefill and production of the first decision token"
            ),
            "decode_after_first_token": (
                "vLLM generation_time_ms: first output token to last output token"
            ),
            "aggregation": (
                "component means use the same requests and are additive; percentiles "
                "are reported separately and are not stacked"
            ),
        },
        "methods": methods,
        "comparison": {
            "prefill_first_token_median_speedup": (
                first_original["median_ms"] / first_solo["median_ms"]
            ),
            "prefill_first_token_mean_speedup": (
                first_original["mean_ms"] / first_solo["mean_ms"]
            ),
            "client_mean_speedup": client_original["mean_ms"] / client_solo["mean_ms"],
            "all_decode_measurements_zero": all(
                methods[method]["decode_after_first_token"]["max_ms"] == 0
                for method in methods
            ),
        },
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def style(locale):
    # Debian's Noto CJK TTC files are registered with Matplotlib under their
    # first face name (JP); that face still contains the Simplified-Chinese
    # glyphs used here.
    family = "Noto Sans CJK JP" if locale == "zh" else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family,
        "font.size": 11,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#dce3ed",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def render(summary, out, locale):
    labels = TEXT[locale]
    style(locale)
    methods = ("original", "solo")
    names = [labels[method] for method in methods]
    keys = ("queue", "prefill_and_first_token", "decode_after_first_token", "client_and_other")
    components = np.asarray([
        [summary["methods"][method][key]["mean_ms"] for key in keys]
        for method in methods
    ])
    totals = components.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    x = np.arange(len(methods))
    bottoms = np.zeros(len(methods))
    for index, (key, label, color) in enumerate(zip(keys, labels["components"], COMPONENT_COLORS)):
        values = components[:, index]
        ax.bar(x, values, bottom=bottoms, width=.55, color=color, label=label, zorder=3)
        if key == "prefill_and_first_token":
            for position, (bottom, value) in enumerate(zip(bottoms, values)):
                ax.text(position, bottom + value / 2, f"{value:.1f} ms",
                        ha="center", va="center", color="white", fontsize=11.5, weight="bold")
        bottoms += values

    for position, value in enumerate(totals):
        ax.text(position, value + max(totals) * .025, labels["total"].format(value=value),
                ha="center", va="bottom", color=INK, fontsize=11, weight="bold")

    original = summary["methods"]["original"]["prefill_and_first_token"]["median_ms"]
    solo = summary["methods"]["solo"]["prefill_and_first_token"]["median_ms"]
    speedup = summary["comparison"]["prefill_first_token_median_speedup"]
    requests = summary["workload"]["total_request_measurements"]
    note = (
        f"{labels['median']}:  {original:.1f} → {solo:.1f} ms  ({speedup:.2f}×)\n"
        f"{labels['decode']}:  {labels['decode_note'].format(requests=requests)}"
    )
    ax.text(.5, -.22, note, transform=ax.transAxes, ha="center", va="top",
            color=INK, fontsize=10.5, linespacing=1.6,
            bbox={"boxstyle": "round,pad=.65", "facecolor": "#f7f9fc", "edgecolor": "#e2e8f0"})

    ax.set_xticks(x, names)
    ax.set_ylabel(labels["ylabel"])
    ax.set_ylim(0, max(totals) * 1.18)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9.5)
    fig.suptitle(labels["title"], x=.08, y=.98, ha="left", fontsize=18,
                 weight="bold", color=INK)
    fig.text(.08, .91, labels["subtitle"], ha="left", color=MUTED, fontsize=10)
    fig.subplots_adjust(left=.10, right=.97, bottom=.30, top=.79)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png", "pdf"):
        fig.savefig(out.with_suffix(f".{extension}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="output figure basename without an extension")
    parser.add_argument("--summary", type=Path, required=True,
                        help="machine-readable timing summary")
    parser.add_argument("--locale", choices=("en", "zh"), default="en")
    args = parser.parse_args()
    summary = summarize_report(load_report(args.report))
    write_json(args.summary, summary)
    render(summary, args.out, args.locale)
    print(f"Wrote {args.out}.svg/.png/.pdf and {args.summary}")


if __name__ == "__main__":
    main()

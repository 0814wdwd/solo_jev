#!/usr/bin/env python3
"""Profile JEV-9B on public structured decision workloads.

The script has two phases:

* ``single`` selects requests across the natural prompt-length distribution and
  measures the one-token decision path without cross-request cache reuse.
* ``batch`` compares complete original and SOLO layouts inside already-ready
  batches. It does not wait for future requests or remove any input field.

Run the cache-enabled and cache-disabled conditions against separately started
vLLM servers. ``--cache-mode`` records the operator-controlled server condition;
it does not pretend that a new cache salt is equivalent to disabling APC.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import solo_layout
from solo_layout import (DecisionEngine, LayoutOptimizer, interleaved_batches,
                           load_contract_nli, load_mind, read_json)
from solo_layout._table import Serializer, as_table


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_jsonl_path(path):
    path = Path(path)
    return path.with_name(path.stem + "-requests.jsonl")


def write_request_jsonl(path, report):
    """Write one provenance-linked observation per request, without source text."""
    path = request_jsonl_path(path)
    temporary = path.with_name(path.name + ".tmp")
    lines = []
    for item in report["single_requests"]:
        lines.append(json.dumps({
            "phase": "single",
            "workload": report["workload"]["name"],
            "cache_mode": report["configuration"]["cache_mode"],
            "cache_salt": item["cache_salt"],
            "local_prompt_tokens": item["local_prompt_tokens"],
            **item["request"],
        }, ensure_ascii=False))
    for run in report["batch_runs"]:
        shared = {
            "phase": "batch",
            "workload": report["workload"]["name"],
            "cache_mode": report["configuration"]["cache_mode"],
            "batch_id": run["batch_id"],
            "batch_size": run["batch_size"],
            "batch_index": run["batch_index"],
            "repeat": run["repeat"],
            "method": run["method"],
            "cache_salt": run["cache_salt"],
        }
        lines.extend(json.dumps({**shared, **request}, ensure_ascii=False)
                     for request in run["requests"])
    for run in report.get("quality_runs", []):
        shared = {
            "phase": "quality",
            "workload": report["workload"]["name"],
            "cache_mode": report["configuration"]["cache_mode"],
            "method": run["method"],
        }
        if run.get("cache_salt") is not None:
            shared["cache_salt"] = run["cache_salt"]
        lines.extend(json.dumps({**shared, **request}, ensure_ascii=False)
                     for request in run["requests"])
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(path)


def save_report(path, report):
    atomic_json(path, report)
    write_request_jsonl(path, report)


def package_versions():
    versions = {}
    for name in ("numpy", "numba", "requests", "tokenizers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_information():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return None


def prompt_lengths(rows, method, tokenizer, spec, seed):
    data = read_json([row.state for row in rows])
    table = as_table(data)
    plan = LayoutOptimizer(method, seed=seed).plan(data)
    serialize = Serializer(table, plan.column_order)
    lengths = []
    for i in range(len(rows)):
        prompt = spec.prompt(serialize(i))
        lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False).ids))
    return lengths, list(plan.ordered_columns)


def select_quantiles(items, lengths, count):
    ordered = sorted(zip(items, lengths), key=lambda item: (item[1], item[0].row_id))
    if len(ordered) <= count:
        return tuple(ordered)
    if count == 1:
        return (ordered[len(ordered) // 2],)
    indices = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return tuple(ordered[i] for i in indices)


def coverage_batches(groups, batch_size, *, seed=0):
    """Cover every row once while keeping reuse groups local to ready batches."""
    import random

    groups = list(groups)
    rng = random.Random(seed)
    rng.shuffle(groups)
    stream = []
    cursor = 0
    while cursor < len(groups):
        selected, available = [], 0
        while cursor < len(groups) and available < batch_size:
            selected.append(groups[cursor])
            available += len(groups[cursor].rows)
            cursor += 1
        offset = 0
        while True:
            progressed = False
            for group in selected:
                if offset < len(group.rows):
                    stream.append(group.rows[offset])
                    progressed = True
            if not progressed:
                break
            offset += 1
    return tuple(tuple(stream[start:start + batch_size])
                 for start in range(0, len(stream), batch_size))


def json_value(value):
    return value.item() if isinstance(value, np.generic) else value


def measured_result(result, rows, truth):
    decisions = [json_value(value) for value in result.decisions.tolist()]
    traces = []
    for position, trace in enumerate(result.request_traces):
        item = trace.to_dict()
        item.update({
            "row_id": rows[position].row_id,
            "group_id": rows[position].group_id,
            "gold": json_value(truth[position]),
            "decision": decisions[position],
            "correct": bool(decisions[position] == truth[position]),
            "probabilities": result.probabilities[position].tolist(),
        })
        traces.append(item)
    metrics = result.metrics()
    metrics["accuracy"] = float(np.mean(result.decisions == np.asarray(truth))) if truth else None
    return metrics, traces, decisions


def require_metrics(result, cache_mode, allow_missing):
    if cache_mode == "disabled" and result.cached_tokens not in (None, 0):
        raise RuntimeError(
            "the server was labelled cache-disabled but returned cached prompt tokens; "
            "restart it with JEV_PREFIX_CACHING=0"
        )
    missing = [trace.row_position for trace in result.request_traces
               if trace.queue_time_ms is None or trace.engine_prefill_interval_ms is None]
    if missing and not allow_missing:
        raise RuntimeError(
            "vLLM per-request metrics are missing; restart with "
            "JEV_PER_REQUEST_METRICS=1 or pass --allow-missing-server-metrics"
        )


def usable_groups(workload, tokenizer, methods, max_prompt_tokens, seed):
    spec = workload.decision_spec()
    accepted, rejected = [], []
    lengths_by_row = {}
    orders = {}
    measured_methods = tuple(dict.fromkeys(("original", *methods)))
    for group in workload.groups:
        group_max = 0
        for method in measured_methods:
            lengths, order = prompt_lengths(group.rows, method, tokenizer, spec, seed)
            orders.setdefault(method, order)
            group_max = max(group_max, max(lengths))
            if method == "original":
                lengths_by_row.update(zip((row.row_id for row in group.rows), lengths))
        if group_max <= max_prompt_tokens:
            accepted.append(group)
        else:
            rejected.append({"group_id": group.group_id, "max_prompt_tokens": group_max})
    if not accepted:
        raise ValueError("every selected workload group exceeds --max-prompt-tokens")
    return tuple(accepted), rejected, lengths_by_row, orders


def run_single(args, report, workload, groups, lengths_by_row):
    candidates = tuple(row for group in groups for row in group.rows)
    selected = select_quantiles(candidates, [lengths_by_row[row.row_id] for row in candidates],
                                args.single_samples)
    shortest = min(selected, key=lambda item: item[1])[0]
    with DecisionEngine(args.url, model_dir=args.model_dir, concurrency=1,
                        api_key=args.api_key, timeout=args.timeout) as engine:
        warm = None
        for _ in range(args.warmups):
            warm = engine.scan(read_json([shortest.state]), workload.question, method="original",
                               kind=workload.kind, options=workload.options,
                               cache_salt=uuid.uuid4().hex)
        if warm is not None:
            require_metrics(warm, args.cache_mode, args.allow_missing_server_metrics)
        for row, local_prompt_tokens in selected:
            salt = uuid.uuid4().hex
            result = engine.scan(read_json([row.state]), workload.question, method="original",
                                 kind=workload.kind, options=workload.options,
                                 cache_salt=salt)
            require_metrics(result, args.cache_mode, args.allow_missing_server_metrics)
            metrics, traces, decisions = measured_result(result, (row,), (row.gold,))
            report["single_requests"].append({
                "row_id": row.row_id,
                "group_id": row.group_id,
                "local_prompt_tokens": local_prompt_tokens,
                "cache_salt": salt,
                "metrics": metrics,
                "request": traces[0],
                "decision": decisions[0],
            })
            save_report(args.out, report)


def run_batches(args, report, workload, groups, tokenizer):
    spec = workload.decision_spec()
    with DecisionEngine(args.url, model_dir=args.model_dir, concurrency=args.concurrency,
                        api_key=args.api_key, timeout=args.timeout) as engine:
        warm_row = groups[0].rows[0]
        for _ in range(args.warmups):
            warm = engine.scan(
                read_json([warm_row.state]), workload.question, method="original",
                kind=workload.kind, options=workload.options, cache_salt=uuid.uuid4().hex,
            )
            require_metrics(warm, args.cache_mode, args.allow_missing_server_metrics)
        for batch_size in args.batch_sizes:
            batches = interleaved_batches(
                groups, batch_size, args.batches_per_size,
                seed=args.seed + batch_size * 1009,
            )
            for batch_index, rows in enumerate(batches):
                truth = tuple(row.gold for row in rows)
                local_lengths = {}
                local_orders = {}
                for method in args.methods:
                    lengths, order = prompt_lengths(rows, method, tokenizer, spec, args.seed)
                    if max(lengths) > args.max_prompt_tokens:
                        raise ValueError(
                            f"batch {batch_size}/{batch_index} exceeds the prompt limit under {method}"
                        )
                    local_lengths[method] = lengths
                    local_orders[method] = order
                batch_id = hashlib.sha256("\n".join(row.row_id for row in rows).encode()).hexdigest()[:16]
                for repeat in range(args.repeats):
                    order = args.methods if repeat % 2 == 0 else tuple(reversed(args.methods))
                    predictions = {}
                    repeat_runs = []
                    for method in order:
                        salt = uuid.uuid4().hex
                        result = engine.scan(
                            read_json([row.state for row in rows]), workload.question,
                            method=method, kind=workload.kind, options=workload.options,
                            seed=args.seed, cache_salt=salt,
                        )
                        require_metrics(result, args.cache_mode, args.allow_missing_server_metrics)
                        metrics, traces, decisions = measured_result(result, rows, truth)
                        predictions[method] = decisions
                        for position, trace in enumerate(traces):
                            trace["local_prompt_tokens"] = local_lengths[method][position]
                        repeat_runs.append({
                            "phase": "batch",
                            "batch_id": batch_id,
                            "batch_size": batch_size,
                            "batch_index": batch_index,
                            "group_count": len({row.group_id for row in rows}),
                            "repeat": repeat,
                            "method": method,
                            "cache_salt": salt,
                            "metrics": metrics,
                            "column_order": local_orders[method],
                            "row_ids": [row.row_id for row in rows],
                            "requests": traces,
                        })
                    original = predictions.get("original")
                    for run in repeat_runs:
                        if original is not None:
                            run["agreement_with_original"] = float(np.mean(
                                np.asarray(predictions[run["method"]], dtype=object)
                                == np.asarray(original, dtype=object)
                            ))
                        report["batch_runs"].append(run)
                    save_report(args.out, report)


def summarize_quality(runs):
    """Summarize one full-coverage run per layout without hiding paired outcomes."""
    indexed = {run["method"]: run for run in runs}
    summary = {"methods": {}}
    for method, run in indexed.items():
        requests = run["requests"]
        summary["methods"][method] = {
            "rows": len(requests),
            "correct": sum(bool(item["correct"]) for item in requests),
            "accuracy": run["metrics"]["accuracy"],
        }
    original, solo = indexed.get("original"), indexed.get("solo")
    if original is None or solo is None:
        return summary
    left, right = original["requests"], solo["requests"]
    if [item["row_id"] for item in left] != [item["row_id"] for item in right]:
        raise RuntimeError("full-quality layouts returned different row identities")
    paired = {
        "agreement": 0,
        "same_correct": 0,
        "same_wrong": 0,
        "solo_fixes": 0,
        "solo_breaks": 0,
        "both_wrong_different": 0,
    }
    for baseline, candidate in zip(left, right):
        if baseline["decision"] == candidate["decision"]:
            paired["agreement"] += 1
            paired["same_correct" if baseline["correct"] else "same_wrong"] += 1
        elif candidate["correct"] and not baseline["correct"]:
            paired["solo_fixes"] += 1
        elif baseline["correct"] and not candidate["correct"]:
            paired["solo_breaks"] += 1
        else:
            paired["both_wrong_different"] += 1
    paired["rows"] = len(left)
    paired["agreement"] /= len(left) if left else 1
    paired["accuracy_delta"] = (
        summary["methods"]["solo"]["accuracy"]
        - summary["methods"]["original"]["accuracy"]
    )
    summary["paired"] = paired
    return summary


def run_quality(args, report, workload, groups, tokenizer):
    """Evaluate every usable row once in bounded, already-ready batches."""
    batches = coverage_batches(groups, args.quality_batch_size, seed=args.seed)
    spec = workload.decision_spec()
    accumulators = {method: {"requests": [], "batches": []} for method in args.methods}
    with DecisionEngine(args.url, model_dir=args.model_dir, concurrency=args.concurrency,
                        api_key=args.api_key, timeout=args.timeout) as engine:
        warm_row = batches[0][0]
        for _ in range(args.warmups):
            warm = engine.scan(
                read_json([warm_row.state]), workload.question, method="original",
                kind=workload.kind, options=workload.options, cache_salt=uuid.uuid4().hex,
            )
            require_metrics(warm, args.cache_mode, args.allow_missing_server_metrics)
        for batch_index, rows in enumerate(batches):
            truth = tuple(row.gold for row in rows)
            method_order = args.methods if batch_index % 2 == 0 else tuple(reversed(args.methods))
            for method in method_order:
                lengths, column_order = prompt_lengths(rows, method, tokenizer, spec, args.seed)
                if max(lengths) > args.max_prompt_tokens:
                    raise ValueError(
                        f"quality batch {batch_index} exceeds the prompt limit under {method}"
                    )
                salt = uuid.uuid4().hex
                result = engine.scan(
                    read_json([row.state for row in rows]), workload.question,
                    method=method, kind=workload.kind, options=workload.options,
                    seed=args.seed, cache_salt=salt,
                )
                require_metrics(result, args.cache_mode, args.allow_missing_server_metrics)
                metrics, traces, _ = measured_result(result, rows, truth)
                for position, trace in enumerate(traces):
                    trace.update({
                        "local_prompt_tokens": lengths[position],
                        "quality_batch_index": batch_index,
                        "cache_salt": salt,
                    })
                accumulators[method]["requests"].extend(traces)
                accumulators[method]["batches"].append({
                    "batch_index": batch_index,
                    "rows": len(rows),
                    "group_count": len({row.group_id for row in rows}),
                    "cache_salt": salt,
                    "column_order": column_order,
                    "metrics": metrics,
                })
            report["quality_progress"] = {
                "completed_batches": batch_index + 1,
                "total_batches": len(batches),
                "rows_completed_per_method": sum(len(batch) for batch in batches[:batch_index + 1]),
            }
            save_report(args.out, report)
    for method in args.methods:
        accumulator = accumulators[method]
        requests = accumulator["requests"]
        batch_metrics = [item["metrics"] for item in accumulator["batches"]]
        prompt = sum(item["prompt_tokens"] for item in batch_metrics)
        cached = sum(item["cached_tokens"] for item in batch_metrics)
        wall = sum(item["wall_seconds"] for item in batch_metrics)
        orders = [item["column_order"] for item in accumulator["batches"]]
        report["quality_runs"].append({
                "phase": "quality",
                "rows": len(requests),
                "group_count": len(groups),
                "method": method,
                "quality_batch_size": args.quality_batch_size,
                "batches": accumulator["batches"],
                "column_order": orders[0] if all(order == orders[0] for order in orders) else None,
                "column_orders": orders,
                "metrics": {
                    "rows": len(requests),
                    "wall_seconds": wall,
                    "rows_per_second": len(requests) / wall,
                    "prompt_tokens": prompt,
                    "cached_tokens": cached,
                    "cached_fraction": cached / prompt if prompt else None,
                    "accuracy": sum(bool(item["correct"]) for item in requests) / len(requests),
                },
                "requests": requests,
            })
    report["quality_summary"] = summarize_quality(report["quality_runs"])
    save_report(args.out, report)


def workload_from_args(args):
    if args.workload == "contract-nli":
        return load_contract_nli(args.data, max_documents=args.max_documents)
    return load_mind(
        args.news, args.behaviors,
        history_items=args.history_items,
        max_impressions=args.max_impressions,
    )


def source_hashes(args):
    if args.workload == "contract-nli":
        return {"contract_nli_json": file_sha256(args.data)}
    return {"mind_news_tsv": file_sha256(args.news),
            "mind_behaviors_tsv": file_sha256(args.behaviors)}


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default="http://127.0.0.1:8000")
    common.add_argument("--model-dir", type=Path, required=True)
    common.add_argument("--out", type=Path, required=True)
    common.add_argument("--phase", choices=("single", "batch", "quality", "all"), default="all",
                        help="'all' retains the historical single+batch run; quality is explicit")
    common.add_argument("--cache-mode", choices=("enabled", "disabled"), required=True)
    common.add_argument("--methods", choices=("original", "solo"), nargs="+",
                        default=("original", "solo"))
    common.add_argument("--repeats", type=int, default=5)
    common.add_argument("--batch-sizes", type=int, nargs="+", default=(8, 16, 32, 64))
    common.add_argument("--batches-per-size", type=int, default=3)
    common.add_argument("--single-samples", type=int, default=32)
    common.add_argument("--quality-batch-size", type=int, default=64,
                        help="already-ready batch size for full-coverage quality evaluation")
    common.add_argument("--warmups", type=int, default=2)
    common.add_argument("--concurrency", type=int, default=4)
    common.add_argument("--max-prompt-tokens", type=int, default=16000)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--timeout", type=float, default=300)
    common.add_argument("--api-key")
    common.add_argument("--allow-missing-server-metrics", action="store_true")
    common.add_argument("--dry-run", action="store_true")

    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="workload", required=True)
    contract = sub.add_parser("contract-nli", parents=[common])
    contract.add_argument("--data", type=Path, required=True,
                          help="Official train/dev/test JSON obtained under the dataset terms")
    contract.add_argument("--max-documents", type=int, default=64)
    mind = sub.add_parser("mind", parents=[common])
    mind.add_argument("--news", type=Path, required=True)
    mind.add_argument("--behaviors", type=Path, required=True)
    mind.add_argument("--history-items", type=int, default=20)
    mind.add_argument("--max-impressions", type=int, default=200)
    return root


def validate_args(args, root):
    for name in ("repeats", "batches_per_size", "single_samples", "quality_batch_size",
                 "concurrency", "max_prompt_tokens"):
        if getattr(args, name) < 1:
            root.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        root.error("--warmups cannot be negative")
    if any(value < 1 for value in args.batch_sizes):
        root.error("--batch-sizes values must be positive")
    if len(set(args.methods)) != len(args.methods):
        root.error("--methods must be unique")


def main():
    root = parser()
    args = root.parse_args()
    validate_args(args, root)
    workload = workload_from_args(args)
    tokenizer_path = args.model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        root.error(f"tokenizer not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    groups, rejected, lengths, orders = usable_groups(
        workload, tokenizer, args.methods, args.max_prompt_tokens, args.seed
    )
    private = {"api_key", "url", "out"}
    configuration = {}
    for key, value in vars(args).items():
        if key in private:
            continue
        configuration[key] = value.name if isinstance(value, Path) else value
    report = {
        "schema_version": 1,
        "status": "dry-run" if args.dry_run else "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "provenance": {
            "solo_layout_version": solo_layout.__version__,
            "python_version": platform.python_version(),
            "dependencies": package_versions(),
            "gpu": gpu_information(),
            "tokenizer_sha256": file_sha256(tokenizer_path),
            "source_sha256": source_hashes(args),
            "server_claim": {
                "prefix_cache": args.cache_mode,
                "per_request_metrics_expected": not args.allow_missing_server_metrics,
                "note": "Server flags are operator-supplied; archive the deployment configuration with the result.",
            },
        },
        "workload": {
            "name": workload.name,
            "source": workload.source,
            "question": workload.question,
            "kind": workload.kind,
            "options": workload.options,
            "loaded_groups": len(workload.groups),
            "usable_groups": len(groups),
            "usable_rows": sum(len(group.rows) for group in groups),
            "rejected_oversize_groups": rejected,
            "representative_column_orders": orders,
            "selection_note": "Contract text is not truncated. MIND history length is the explicit history_items setting. No prompt is silently clipped; groups above the configured prompt limit are excluded and listed.",
        },
        "measurement": {
            "single": "Concurrency 1; fresh cache namespace per request; selected across the natural prompt-length distribution.",
            "batch": "All rows are ready before planning. wall_seconds includes planning, serialization, bounded submission, inference and output restoration.",
            "quality": "Every usable workload row is evaluated once per requested layout against its gold label, partitioned into bounded already-ready batches.",
            "engine_intervals": "Server-reported wall-clock intervals, not summed GPU kernel time.",
            "cache": "A fresh namespace prevents cross-trial hits but still includes the first cache fill. APC enabled/disabled requires separate server starts.",
        },
        "single_requests": [],
        "batch_runs": [],
        "quality_runs": [],
        "request_jsonl": request_jsonl_path(args.out).name,
    }
    save_report(args.out, report)
    if args.dry_run:
        print(json.dumps(report["workload"], ensure_ascii=False, indent=2))
        print(f"Dry-run report: {args.out}")
        return
    try:
        if args.phase in ("single", "all"):
            run_single(args, report, workload, groups, lengths)
        if args.phase in ("batch", "all"):
            run_batches(args, report, workload, groups, tokenizer)
        if args.phase == "quality":
            run_quality(args, report, workload, groups, tokenizer)
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        save_report(args.out, report)
        raise
    report["status"] = "complete"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save_report(args.out, report)
    print(f"Complete profiling report: {args.out}")


if __name__ == "__main__":
    main()

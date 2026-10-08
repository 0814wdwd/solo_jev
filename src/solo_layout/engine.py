"""One table scan, with bounded request concurrency and positional outputs."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
import statistics
import threading
import time
import uuid

import numpy as np

from ._table import JSONInput, Serializer, as_table, read_json
from .backend import DecisionSpec, JevBackend
from .layout import LayoutOptimizer, LayoutPlan


@dataclass
class RequestTrace:
    """One request timeline, ordered by its original input row position.

    Offsets use the start of :meth:`DecisionEngine.scan` as the batch-ready
    origin. Engine intervals are optional server-reported wall-clock metrics;
    they are not CUDA-kernel timings.
    """
    row_position: int
    execution_position: int
    submit_offset_seconds: float
    request_start_offset_seconds: float
    complete_offset_seconds: float
    client_request_seconds: float
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    created_cache_tokens: int | None = None
    queue_time_ms: float | None = None
    engine_prefill_interval_ms: float | None = None
    generation_time_ms: float | None = None
    request_id: str | None = None

    @property
    def batch_sojourn_seconds(self):
        return self.complete_offset_seconds

    @property
    def executor_wait_seconds(self):
        return self.request_start_offset_seconds - self.submit_offset_seconds

    def to_dict(self):
        return {
            "row_position": self.row_position,
            "execution_position": self.execution_position,
            "submit_offset_seconds": self.submit_offset_seconds,
            "request_start_offset_seconds": self.request_start_offset_seconds,
            "complete_offset_seconds": self.complete_offset_seconds,
            "batch_sojourn_seconds": self.batch_sojourn_seconds,
            "executor_wait_seconds": self.executor_wait_seconds,
            "client_request_seconds": self.client_request_seconds,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "created_cache_tokens": self.created_cache_tokens,
            "queue_time_ms": self.queue_time_ms,
            "engine_prefill_interval_ms": self.engine_prefill_interval_ms,
            "generation_time_ms": self.generation_time_ms,
            "request_id": self.request_id,
        }


@dataclass
class ScanResult:
    decisions: np.ndarray
    probabilities: np.ndarray
    options: tuple
    layout: LayoutPlan
    wall_seconds: float
    inference_seconds: float
    preparation_seconds: float
    prompt_tokens: int | None
    cached_tokens: int | None
    latencies_seconds: np.ndarray
    index: object = None
    completion_tokens: int | None = None
    created_cache_tokens: int | None = None
    request_traces: tuple[RequestTrace, ...] = ()

    @property
    def rows_per_second(self):
        return len(self.decisions) / self.wall_seconds if self.wall_seconds else 0.0

    def metrics(self):
        def average(name, scale=1.0):
            values = [getattr(trace, name) for trace in self.request_traces
                      if getattr(trace, name) is not None]
            return float(np.mean(values)) * scale if values else None

        sojourn = [trace.batch_sojourn_seconds for trace in self.request_traces]
        return {
            "method": self.layout.method, "rows": len(self.decisions),
            "wall_seconds": self.wall_seconds, "rows_per_second": self.rows_per_second,
            "preparation_seconds": self.preparation_seconds,
            "planning_seconds": self.layout.planning_seconds,
            "inference_seconds": self.inference_seconds,
            "prompt_tokens": self.prompt_tokens, "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "created_cache_tokens": self.created_cache_tokens,
            "cached_fraction": (self.cached_tokens / self.prompt_tokens
                                if self.prompt_tokens and self.cached_tokens is not None else None),
            "latency_p50_seconds": float(np.median(self.latencies_seconds)) if len(self.decisions) else 0.0,
            "latency_p95_seconds": float(np.percentile(self.latencies_seconds, 95)) if len(self.decisions) else 0.0,
            "batch_sojourn_p50_seconds": float(np.median(sojourn)) if sojourn else 0.0,
            "batch_sojourn_p95_seconds": float(np.percentile(sojourn, 95)) if sojourn else 0.0,
            "engine_queue_mean_seconds": average("queue_time_ms", .001),
            "engine_prefill_interval_mean_seconds": average("engine_prefill_interval_ms", .001),
            "engine_generation_mean_seconds": average("generation_time_ms", .001),
            "column_order": list(self.layout.ordered_columns),
        }

    def to_pandas(self):
        import pandas as pd
        columns = {"decision": self.decisions}
        columns.update({f"p_{str(option).lower()}": self.probabilities[:, i]
                        for i, option in enumerate(self.options)})
        return pd.DataFrame(columns, index=self.index)


@dataclass
class ComparisonResult:
    runs: list[dict]
    summary: list[dict]

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self.summary).set_index("method")


class DecisionEngine:
    def __init__(self, base_url="http://127.0.0.1:8000", *, model="jev-decision",
                 model_dir=None, concurrency=4, api_key=None, timeout=180, backend=None):
        if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        self.backend = backend if backend is not None else JevBackend(
            base_url, model=model, model_dir=model_dir, api_key=api_key, timeout=timeout)
        self.concurrency = concurrency
        self._pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="solo")
        self._lock = threading.RLock()
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self._lock:
            if not self._closed:
                self._pool.shutdown(wait=True, cancel_futures=True)
                if hasattr(self.backend, "close"):
                    self.backend.close()
                self._closed = True

    def scan(self, data, question, *, columns=None, method="solo", kind="noul",
             options=None, seed=0, sample_size=None, cache_salt=None):
        """Decide once per complete row, returning results in input order.

        Accepts DataFrames, arrays, records, JSON text and JSON/JSONL Paths.
        Use read_json(parsed_records) to preserve JSON scalar types instead
        of the historical str(cell) convention for ordinary tabular input.
        """
        with self._lock:
            return self._scan(data, question, columns=columns, method=method, kind=kind,
                              options=options, seed=seed, sample_size=sample_size, cache_salt=cache_salt)

    def explain(self, data, *, columns=None,
                methods=("original", "lexicographic", "random", "solo"), seed=0, sample_size=None):
        """Inspect layout structure offline; this never contacts the backend."""
        return LayoutOptimizer(seed=seed, sample_size=sample_size).explain(
            data, columns=columns, methods=methods)

    def _scan(self, data, question, *, columns, method, kind, options, seed, sample_size, cache_salt):
        if self._closed:
            raise RuntimeError("DecisionEngine is closed")
        started = time.perf_counter()
        spec = DecisionSpec.create(question, kind, options)
        optimizer = LayoutOptimizer(method, seed=seed, sample_size=sample_size)
        table = as_table(data, columns)
        n = table.shape[0]
        # Metadata loading occurs before requests, never once per row.
        if n and hasattr(self.backend, "prepare"):
            self.backend.prepare(spec)
        plan = optimizer._plan(table)
        serialize = Serializer(table, plan.column_order)
        probabilities = np.empty((n, len(spec.options)), dtype=float)
        latencies = np.empty(n, dtype=float)
        traces = [None] * n
        preparation = time.perf_counter() - started - plan.planning_seconds
        prompt_tokens, cached_tokens, completion_tokens, created_cache_tokens = 0, 0, 0, 0
        has_prompt, has_cache, has_completion, has_created = True, True, True, True

        def ask(i, execution_position, submitted):
            request_started = time.perf_counter()
            response = self.backend.decide(serialize(i), spec, cache_salt=cache_salt)
            completed = time.perf_counter()
            p = np.asarray(response.probabilities, dtype=float)
            if (p.shape != (len(spec.options),) or not np.isfinite(p).all()
                    or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0)):
                raise ValueError("backend returned an invalid probability distribution")
            trace = RequestTrace(
                row_position=i,
                execution_position=execution_position,
                submit_offset_seconds=submitted - started,
                request_start_offset_seconds=request_started - started,
                complete_offset_seconds=completed - started,
                client_request_seconds=completed - request_started,
                prompt_tokens=response.prompt_tokens,
                cached_tokens=response.cached_tokens,
                completion_tokens=response.completion_tokens,
                created_cache_tokens=response.created_cache_tokens,
                queue_time_ms=response.queue_time_ms,
                engine_prefill_interval_ms=response.time_to_first_token_ms,
                generation_time_ms=response.generation_time_ms,
                request_id=response.request_id,
            )
            return i, response, trace

        inference_started = time.perf_counter()
        jobs = iter(enumerate(int(i) for i in plan.row_order))
        pending = {}

        def submit_next():
            item = next(jobs, None)
            if item is None:
                return False
            execution_position, i = item
            submitted = time.perf_counter()
            pending[self._pool.submit(ask, i, execution_position, submitted)] = i
            return True

        try:
            for _ in range(min(self.concurrency, n)):
                submit_next()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Sorting simultaneous completions makes refill order reproducible.
                for future in sorted(done, key=lambda f: pending[f]):
                    i = pending.pop(future)
                    try:
                        _, response, trace = future.result()
                    except Exception as exc:
                        raise RuntimeError(f"decision failed at input row position {i}: {exc}") from exc
                    probabilities[i] = response.probabilities
                    latencies[i] = trace.client_request_seconds
                    traces[i] = trace
                    has_prompt &= response.prompt_tokens is not None
                    has_cache &= response.cached_tokens is not None
                    has_completion &= response.completion_tokens is not None
                    has_created &= response.created_cache_tokens is not None
                    prompt_tokens += response.prompt_tokens or 0
                    cached_tokens += response.cached_tokens or 0
                    completion_tokens += response.completion_tokens or 0
                    created_cache_tokens += response.created_cache_tokens or 0
                    submit_next()
        finally:
            for future in pending:
                future.cancel()
            if pending:
                wait(pending)
        inference_seconds = time.perf_counter() - inference_started
        decisions = np.asarray(spec.options)[np.argmax(probabilities, axis=1)]
        return ScanResult(
            decisions=decisions,
            probabilities=probabilities,
            options=spec.options,
            layout=plan,
            wall_seconds=time.perf_counter() - started,
            inference_seconds=inference_seconds,
            preparation_seconds=preparation,
            prompt_tokens=prompt_tokens if has_prompt else None,
            cached_tokens=cached_tokens if has_cache else None,
            latencies_seconds=latencies,
            index=table.index,
            completion_tokens=completion_tokens if has_completion else None,
            created_cache_tokens=created_cache_tokens if has_created else None,
            request_traces=tuple(traces),
        )

    def compare(self, data, question, *, methods=("original", "lexicographic", "random", "solo"),
                repeats=2, truth=None, columns=None, kind="noul", options=None, seed=0, sample_size=None):
        """AB/BA trials, fresh APC namespace per trial; includes first cache fills.

        Requires a backend that implements vLLM-style cache_salt. This does not
        flush other users' caches. It isolates matches, not GPU contention.
        JSON text/files are loaded once before trials; per-trial timings still
        include table normalization, layout planning and row serialization.
        """
        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        methods = tuple(methods)
        if not methods or len(set(methods)) != len(methods):
            raise ValueError("methods must be nonempty and unique")
        for method in methods:
            LayoutOptimizer(method, seed=seed, sample_size=sample_size)
        if not getattr(self.backend, "supports_cache_salt", False):
            raise ValueError("compare requires a backend with cache_salt support")
        # Reusable input; a generator must not be consumed afresh by every trial.
        if isinstance(data, (JSONInput, str, PathLike, Mapping)):
            data = read_json(data)
        elif not hasattr(data, "shape"):
            data = list(data)
        if truth is not None:
            truth = np.asarray(truth)
            if truth.ndim != 1 or len(truth) != len(data):
                raise ValueError("truth must have one label per input row")
        runs, predictions = [], {}
        with self._lock:
            for repeat in range(repeats):
                order = methods if repeat % 2 == 0 else methods[::-1]
                for method in order:
                    salt = uuid.uuid4().hex
                    result = self.scan(data, question, columns=columns, method=method, kind=kind,
                                       options=options, seed=seed, sample_size=sample_size, cache_salt=salt)
                    metrics = result.metrics()
                    metrics.update(repeat=repeat, cache_salt=salt)
                    if truth is not None:
                        metrics["accuracy"] = float(np.mean(result.decisions == truth)) if len(truth) else None
                    runs.append(metrics)
                    predictions[method, repeat] = result.decisions
        summaries = []
        for method in methods:
            trials = [r for r in runs if r["method"] == method]
            summary = {"method": method, "rows": len(data), "repeats": repeats}
            for key in ("wall_seconds", "planning_seconds", "prompt_tokens", "cached_tokens",
                        "cached_fraction", "accuracy"):
                values = [r[key] for r in trials if r.get(key) is not None]
                summary[key] = statistics.median(values) if values else None
            summary["rows_per_second"] = len(data) / summary["wall_seconds"] if summary["wall_seconds"] else 0.0
            if "original" in methods and len(data):
                summary["agreement_with_original"] = statistics.mean(
                    float(np.mean(predictions[method, r] == predictions["original", r])) for r in range(repeats))
            summaries.append(summary)
        times = {r["method"]: r["wall_seconds"] for r in summaries}
        for summary in summaries:
            if "original" in times:
                summary["speedup_vs_original"] = times["original"] / summary["wall_seconds"] if summary["wall_seconds"] else None
            if "solo" in times:
                summary["solo_speedup_vs_this_method"] = summary["wall_seconds"] / times["solo"] if times["solo"] else None
        return ComparisonResult(runs, summaries)

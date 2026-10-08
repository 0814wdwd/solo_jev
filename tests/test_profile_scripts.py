import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_request_jsonl_has_one_provenance_linked_row_per_observation(tmp_path):
    pytest.importorskip("tokenizers")
    module = load_script("profile_public")
    report = {
        "configuration": {"cache_mode": "enabled"},
        "workload": {"name": "fixture"},
        "single_requests": [{
            "cache_salt": "single-salt", "local_prompt_tokens": 10,
            "request": {"row_id": "one", "client_request_seconds": .1},
        }],
        "batch_runs": [{
            "batch_id": "batch", "batch_size": 2, "batch_index": 0,
            "repeat": 1, "method": "solo", "cache_salt": "batch-salt",
            "requests": [{"row_id": "two"}, {"row_id": "three"}],
        }],
        "quality_runs": [{
            "method": "original", "cache_salt": "quality-salt",
            "requests": [{"row_id": "four"}],
        }],
    }
    target = tmp_path / "report.json"
    module.write_request_jsonl(target, report)
    rows = [json.loads(line) for line in (tmp_path / "report-requests.jsonl").read_text().splitlines()]
    assert [row["row_id"] for row in rows] == ["one", "two", "three", "four"]
    assert rows[0]["phase"] == "single" and rows[1]["phase"] == "batch"
    assert rows[2]["method"] == "solo" and rows[2]["cache_mode"] == "enabled"
    assert rows[3]["phase"] == "quality" and rows[3]["method"] == "original"


def test_full_quality_summary_keeps_paired_outcome_counts():
    pytest.importorskip("tokenizers")
    module = load_script("profile_public")
    def request(row_id, decision, gold):
        return {"row_id": row_id, "decision": decision, "gold": gold,
                "correct": decision == gold}
    runs = [{
        "method": "original", "metrics": {"accuracy": .5},
        "requests": [request("a", "yes", "yes"), request("b", "yes", "no")],
    }, {
        "method": "solo", "metrics": {"accuracy": 1.0},
        "requests": [request("a", "yes", "yes"), request("b", "no", "no")],
    }]
    assert module.summarize_quality(runs) == {
        "methods": {
            "original": {"rows": 2, "correct": 1, "accuracy": .5},
            "solo": {"rows": 2, "correct": 2, "accuracy": 1.0},
        },
        "paired": {
            "agreement": .5, "same_correct": 1, "same_wrong": 0,
            "solo_fixes": 1, "solo_breaks": 0, "both_wrong_different": 0,
            "rows": 2, "accuracy_delta": .5,
        },
    }


def test_coverage_batches_use_every_row_once_and_keep_groups_local():
    pytest.importorskip("tokenizers")
    module = load_script("profile_public")
    class Group:
        def __init__(self, name):
            self.rows = tuple(f"{name}-{i}" for i in range(3))
    groups = [Group(str(i)) for i in range(4)]
    batches = module.coverage_batches(groups, 5, seed=3)
    flattened = [row for batch in batches for row in batch]
    assert len(batches) == 3
    assert sorted(flattened) == sorted(row for group in groups for row in group.rows)
    assert len(flattened) == len(set(flattened)) == 12


def test_renderer_rejects_different_batches_and_bins_single_requests():
    pytest.importorskip("matplotlib")
    module = load_script("render_profiling")

    def report(batch_id):
        return {
            "workload": {"name": "fixture"},
            "provenance": {"tokenizer_sha256": "t", "source_sha256": {"data": "d"}},
            "single_requests": [{
                "local_prompt_tokens": 500,
                "request": {"client_request_seconds": .2, "engine_prefill_interval_ms": 150,
                            "queue_time_ms": 2, "completion_tokens": 1},
            }],
            "batch_runs": [{"batch_id": batch_id, "batch_size": 8, "batch_index": 0}],
        }

    left, right = report("same"), report("same")
    module.validate_pair(left, right)
    points, bins = module.latency_bins(left)
    assert points[0]["completion_tokens"] == 1
    assert bins == [{"label": "≤512", "tokens": 500.0, "client_seconds": .2,
                     "prefill_seconds": .15, "queue_seconds": .002, "samples": 1}]
    right["batch_runs"][0]["batch_id"] = "different"
    with pytest.raises(ValueError, match="same input batches"):
        module.validate_pair(left, right)


def test_renderer_batch_aggregates_include_quality_and_reuse():
    pytest.importorskip("matplotlib")
    module = load_script("render_profiling")
    report = {"batch_runs": [
        {"batch_size": 8, "method": "solo", "agreement_with_original": .75,
         "metrics": {"wall_seconds": 2, "rows_per_second": 4,
                     "cached_fraction": .5, "prompt_tokens": 100, "accuracy": .5}},
        {"batch_size": 8, "method": "solo", "agreement_with_original": 1,
         "metrics": {"wall_seconds": 4, "rows_per_second": 2,
                     "cached_fraction": .75, "prompt_tokens": 100, "accuracy": 1}},
    ]}
    assert module.aggregate_batches(report) == [{
        "batch_size": 8, "method": "solo", "runs": 2,
        "wall_seconds_median": 3.0, "rows_per_second_median": 3.0,
        "cached_fraction_median": .625, "prompt_tokens_median": 100.0,
        "accuracy_median": .75, "agreement_with_original_median": .875,
    }]


def test_renderer_speedup_is_paired_by_batch_and_repeat():
    pytest.importorskip("matplotlib")
    module = load_script("render_profiling")
    runs = []
    for batch_id, repeat, original, solo in (("a", 0, 10, 2), ("b", 0, 2, 1)):
        common = {"batch_id": batch_id, "batch_size": 8, "batch_index": 0,
                  "repeat": repeat}
        runs.extend((
            {**common, "method": "original", "metrics": {"wall_seconds": original}},
            {**common, "method": "solo", "metrics": {"wall_seconds": solo}},
        ))
    assert module.paired_speedups({"batch_runs": runs}) == [{
        "batch_size": 8, "baseline": "original", "candidate": "solo", "pairs": 2,
        "speedup_median": 3.5, "speedup_min": 2.0, "speedup_max": 5.0,
    }]


def test_phase_profile_uses_additive_means_and_keeps_decode_separate():
    pytest.importorskip("matplotlib")
    module = load_script("render_phase_profile")

    def request(client, queue, first, decode):
        return {
            "client_request_seconds": client / 1000,
            "queue_time_ms": queue,
            "engine_prefill_interval_ms": first,
            "generation_time_ms": decode,
            "completion_tokens": 1,
        }

    report = {
        "workload": {"source": {"dataset": "fixture"}},
        "configuration": {
            "concurrency": 4, "cache_mode": "enabled", "quality_batch_size": 8,
        },
        "provenance": {"gpu": ["fixture GPU"]},
        "quality_runs": [
            {"method": "original", "requests": [
                request(120, 10, 80, 0), request(180, 20, 100, 0),
            ]},
            {"method": "solo", "requests": [
                request(80, 5, 40, 0), request(100, 5, 50, 0),
            ]},
        ],
    }
    summary = module.summarize_report(report)
    assert summary["methods"]["original"]["client_and_other"]["mean_ms"] == 45
    assert summary["methods"]["solo"]["prefill_and_first_token"]["median_ms"] == 45
    assert summary["comparison"]["prefill_first_token_median_speedup"] == 2
    assert summary["comparison"]["all_decode_measurements_zero"] is True


def test_phase_profile_rejects_missing_server_timing():
    pytest.importorskip("matplotlib")
    module = load_script("render_phase_profile")
    with pytest.raises(ValueError, match="generation_time_ms"):
        module.summarize_requests([{
            "client_request_seconds": .1,
            "queue_time_ms": 1,
            "engine_prefill_interval_ms": 80,
            "completion_tokens": 1,
        }])

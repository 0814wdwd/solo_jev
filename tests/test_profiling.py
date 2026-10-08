import json

import pytest

from solo_layout import interleaved_batches, load_contract_nli, load_mind


def test_contract_nli_loader_preserves_full_text_and_natural_groups(tmp_path):
    payload = {
        "labels": {
            "nda-1": {"hypothesis": "The agreement survives termination."},
            "nda-2": {"hypothesis": "Information must be marked confidential."},
        },
        "documents": [
            {
                "id": 7,
                "text": "FULL CONTRACT A\nNo truncation.",
                "document_type": "sec-text",
                "annotation_sets": [{"annotations": {
                    "nda-1": {"choice": "Entailment", "spans": [0]},
                    "nda-2": {"choice": "NotMentioned", "spans": []},
                }}],
            },
            {
                "id": 8,
                "text": "FULL CONTRACT B",
                "document_type": "search-pdf",
                "annotation_sets": [{"annotations": {
                    "nda-1": {"choice": "Contradiction", "spans": [1]},
                    "nda-2": {"choice": "Entailment", "spans": [2]},
                }}],
            },
        ],
    }
    source = tmp_path / "test.json"
    source.write_text(json.dumps(payload))
    workload = load_contract_nli(source)
    assert workload.kind == "choice"
    assert workload.options == ("Entailment", "Contradiction", "NotMentioned")
    assert len(workload.groups) == 2 and all(len(group.rows) == 2 for group in workload.groups)
    first = workload.groups[0].rows[0]
    assert list(first.state)[0] == "record_id"
    assert first.state["contract_text"] == payload["documents"][0]["text"]
    assert first.gold == "Entailment"
    assert workload.decision_spec().options == workload.options


def test_mind_loader_expands_history_and_keeps_candidate_labels(tmp_path):
    news = tmp_path / "news.tsv"
    news.write_text(
        "N1\tnews\tlocal\tFirst title\tFirst abstract\thttps://1\t[]\t[]\n"
        "N2\tsports\tball\tSecond title\tSecond abstract\thttps://2\t[]\t[]\n"
        "N3\ttech\tai\tThird title\tThird abstract\thttps://3\t[]\t[]\n"
    )
    behaviors = tmp_path / "behaviors.tsv"
    behaviors.write_text("12\tU1\t11/13/2019\tN1\tN2-1 N3-0\n")
    workload = load_mind(news, behaviors, history_items=1)
    rows = workload.groups[0].rows
    assert [row.gold for row in rows] == [True, False]
    assert rows[0].state["history"] == [{
        "news_id": "N1", "category": "news", "subcategory": "local",
        "title": "First title", "abstract": "First abstract",
    }]
    assert rows[0].state["history"] is rows[1].state["history"]
    assert rows[0].state["candidate"]["news_id"] == "N2"


def test_missing_or_duplicate_source_ids_get_deterministic_first_column(tmp_path):
    payload = {
        "labels": {"nda-1": {"hypothesis": "H"}},
        "documents": [{
            "text": "first", "annotation_sets": [{"annotations": {
                "nda-1": {"choice": "Entailment"},
            }}],
        }, {
            "text": "second", "annotation_sets": [{"annotations": {
                "nda-1": {"choice": "NotMentioned"},
            }}],
        }],
    }
    source = tmp_path / "missing-ids.json"
    source.write_text(json.dumps(payload))
    rows = load_contract_nli(source).rows
    assert [row.row_id for row in rows] == ["row-000000001:nda-1", "row-000000002:nda-1"]
    assert len({row.row_id for row in rows}) == len(rows)
    assert all(list(row.state)[0] == "record_id" for row in rows)


def test_interleaved_batches_are_group_mixed_and_do_not_reuse_groups(tmp_path):
    payload = {
        "labels": {"nda-1": {"hypothesis": "H1"}, "nda-2": {"hypothesis": "H2"}},
        "documents": [{
            "id": i, "text": f"contract {i}", "annotation_sets": [{"annotations": {
                "nda-1": {"choice": "Entailment"}, "nda-2": {"choice": "NotMentioned"},
            }}]
        } for i in range(6)],
    }
    source = tmp_path / "data.json"
    source.write_text(json.dumps(payload))
    groups = load_contract_nli(source).groups
    batches = interleaved_batches(groups, 4, 2, seed=3)
    assert len(batches) == 2 and all(len(batch) == 4 for batch in batches)
    assert all(batch[0].group_id != batch[1].group_id for batch in batches)
    assert set(row.group_id for row in batches[0]).isdisjoint(
        row.group_id for row in batches[1]
    )


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_public_workload_limits_are_rejected(tmp_path, limit):
    with pytest.raises(ValueError):
        load_contract_nli(tmp_path / "missing.json", max_documents=limit)

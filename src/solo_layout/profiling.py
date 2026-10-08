"""Public structured-decision workloads used by the profiling harness.

The loaders do not download or modify third-party datasets. Callers must obtain
the source files under their respective licenses and pass explicit local paths.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

from .backend import DecisionSpec


@dataclass(frozen=True)
class WorkloadRow:
    row_id: str
    group_id: str
    state: dict
    gold: object


@dataclass(frozen=True)
class WorkloadGroup:
    group_id: str
    rows: tuple[WorkloadRow, ...]


@dataclass(frozen=True)
class PublicWorkload:
    name: str
    question: str
    kind: str
    options: tuple | None
    groups: tuple[WorkloadGroup, ...]
    source: dict

    @property
    def rows(self):
        return tuple(row for group in self.groups for row in group.rows)

    def decision_spec(self):
        return DecisionSpec.create(self.question, self.kind, self.options)


def _positive_limit(value, name):
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
        raise ValueError(f"{name} must be a positive integer or None")


def _unique_identifier(candidate, seen, position):
    """Prefer a source identifier; otherwise use a deterministic 1-based row ID."""
    identifier = str(candidate).strip() if candidate is not None else ""
    if not identifier or identifier in seen:
        identifier = f"row-{position:09d}"
        while identifier in seen:
            position += 1
            identifier = f"row-{position:09d}"
    seen.add(identifier)
    return identifier


def load_contract_nli(path, *, max_documents=None) -> PublicWorkload:
    """Load an official ContractNLI split without truncating contract text.

    Each contract is one natural reuse group and each annotated hypothesis is a
    one-token three-way decision. The original field order deliberately retains
    the source record ID first; layout methods remain responsible for changing
    order, and every method still receives every field.
    """
    _positive_limit(max_documents, "max_documents")
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read ContractNLI JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
        raise ValueError("ContractNLI JSON must contain a documents array")
    labels = payload.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("ContractNLI JSON must contain nonempty labels")
    hypotheses = {}
    for key, label in labels.items():
        if not isinstance(key, str) or not isinstance(label, dict):
            raise ValueError("invalid ContractNLI label entry")
        hypothesis = label.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError(f"ContractNLI label {key!r} has no hypothesis")
        hypotheses[key] = hypothesis

    valid_choices = ("Entailment", "Contradiction", "NotMentioned")
    groups = []
    seen_groups, seen_rows = set(), set()
    row_position = 0
    for document_position, document in enumerate(payload["documents"][:max_documents], 1):
        if not isinstance(document, dict):
            raise ValueError("invalid ContractNLI document entry")
        source_document_id = document.get("id")
        text = document.get("text")
        sets = document.get("annotation_sets")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("ContractNLI document requires nonempty text")
        document_id = _unique_identifier(
            f"contract-{source_document_id}" if source_document_id is not None else None,
            seen_groups, document_position,
        )
        if not isinstance(sets, list) or len(sets) != 1 or not isinstance(sets[0], dict):
            raise ValueError(f"ContractNLI document {document_id!r} must have one annotation set")
        annotations = sets[0].get("annotations")
        if not isinstance(annotations, dict) or not annotations:
            raise ValueError(f"ContractNLI document {document_id!r} has no annotations")
        group_id = document_id
        rows = []
        for hypothesis_id, hypothesis in hypotheses.items():
            annotation = annotations.get(hypothesis_id)
            if annotation is None:
                continue
            if not isinstance(annotation, dict) or annotation.get("choice") not in valid_choices:
                raise ValueError(f"invalid annotation {hypothesis_id!r} in document {document_id!r}")
            row_position += 1
            row_id = _unique_identifier(f"{group_id}:{hypothesis_id}", seen_rows, row_position)
            state = {
                "record_id": row_id,
                "hypothesis_id": hypothesis_id,
                "hypothesis": hypothesis,
                "document_id": source_document_id,
                "document_type": document.get("document_type"),
                "contract_text": text,
            }
            rows.append(WorkloadRow(row_id, group_id, state, annotation["choice"]))
        if rows:
            groups.append(WorkloadGroup(group_id, tuple(rows)))
    if not groups:
        raise ValueError("ContractNLI selection contains no annotated documents")
    return PublicWorkload(
        name="contract_nli",
        question="What is the relationship between the contract and the stated hypothesis?",
        kind="choice",
        options=valid_choices,
        groups=tuple(groups),
        source={
            "dataset": "ContractNLI",
            "homepage": "https://stanfordnlp.github.io/contract-nli/",
            "license": "CC BY 4.0",
            "input_file": path.name,
            "transformation": "Full contract and all available annotated hypotheses; evidence-span prediction is out of scope.",
        },
    )


def _read_tsv(path, columns):
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, values in enumerate(csv.reader(handle, delimiter="\t"), 1):
                if len(values) != len(columns):
                    raise ValueError(
                        f"{path} line {line_number} has {len(values)} columns; expected {len(columns)}"
                    )
                yield dict(zip(columns, values))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def load_mind(news_path, behaviors_path, *, history_items=20, max_impressions=None) -> PublicWorkload:
    """Load MIND impressions as user-history by candidate decision groups.

    The user history is expanded to public title/abstract metadata so repeated
    context is represented as model input rather than only a list of IDs.
    """
    _positive_limit(history_items, "history_items")
    _positive_limit(max_impressions, "max_impressions")
    news_columns = ("news_id", "category", "subcategory", "title", "abstract", "url",
                    "title_entities", "abstract_entities")
    news = {}
    for row in _read_tsv(news_path, news_columns):
        news[row["news_id"]] = {
            "news_id": row["news_id"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "title": row["title"],
            "abstract": row["abstract"],
        }
    if not news:
        raise ValueError("MIND news file is empty")

    behavior_columns = ("impression_id", "user_id", "time", "history", "impressions")
    groups = []
    seen_groups, seen_rows = set(), set()
    row_position = 0
    for impression_position, behavior in enumerate(_read_tsv(behaviors_path, behavior_columns), 1):
        history_ids = behavior["history"].split()
        if history_items is not None:
            history_ids = history_ids[-history_items:]
        history = [news[item] for item in history_ids if item in news]
        rows = []
        source_impression_id = behavior["impression_id"].strip()
        group_id = _unique_identifier(
            f"impression-{source_impression_id}" if source_impression_id else None,
            seen_groups, impression_position,
        )
        for labelled in behavior["impressions"].split():
            try:
                candidate_id, label = labelled.rsplit("-", 1)
            except ValueError as exc:
                raise ValueError(f"invalid MIND impression label {labelled!r}") from exc
            if label not in ("0", "1"):
                raise ValueError(f"invalid MIND click label {label!r}")
            candidate = news.get(candidate_id)
            if candidate is None:
                continue
            row_position += 1
            row_id = _unique_identifier(f"{group_id}:{candidate_id}", seen_rows, row_position)
            state = {
                "record_id": row_id,
                "candidate_id": candidate_id,
                "candidate": candidate,
                "impression_id": behavior["impression_id"],
                "user_id": behavior["user_id"],
                "history": history,
            }
            rows.append(WorkloadRow(row_id, group_id, state, label == "1"))
        if rows:
            groups.append(WorkloadGroup(group_id, tuple(rows)))
            if max_impressions is not None and len(groups) >= max_impressions:
                break
    if not groups:
        raise ValueError("MIND selection contains no usable impressions")
    return PublicWorkload(
        name="mind",
        question="Based on the user's reading history, is the candidate news likely to be clicked?",
        kind="noul",
        options=None,
        groups=tuple(groups),
        source={
            "dataset": "MIND",
            "homepage": "https://learn.microsoft.com/azure/open-datasets/dataset-microsoft-news",
            "license": "Microsoft Research License Terms",
            "news_file": Path(news_path).name,
            "behaviors_file": Path(behaviors_path).name,
            "history_items": history_items,
            "transformation": "Each labelled candidate becomes one decision; history IDs are expanded to supplied news metadata.",
        },
    )


def interleaved_batches(groups, batch_size, count, *, seed=0):
    """Create deterministic, nonempty batches with reuse groups interleaved."""
    import random

    _positive_limit(batch_size, "batch_size")
    _positive_limit(count, "count")
    groups = list(groups)
    if not groups:
        raise ValueError("at least one workload group is required")
    rng = random.Random(seed)
    rng.shuffle(groups)
    cursor, batches = 0, []
    for _ in range(count):
        selected, available = [], 0
        while available < batch_size:
            if cursor >= len(groups):
                raise ValueError("not enough distinct workload groups for the requested batches")
            selected.append(groups[cursor])
            available += len(groups[cursor].rows)
            cursor += 1
        batch = []
        offset = 0
        while len(batch) < batch_size:
            progressed = False
            for group in selected:
                if offset < len(group.rows) and len(batch) < batch_size:
                    batch.append(group.rows[offset])
                    progressed = True
            if not progressed:
                break
            offset += 1
        if len(batch) != batch_size:
            raise AssertionError("batch construction lost workload rows")
        batches.append(tuple(batch))
    return tuple(batches)

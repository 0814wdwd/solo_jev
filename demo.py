#!/usr/bin/env python3
"""Reproducible full-record demos: shared fields or equal-NDV correlated columns.

python demo.py --model-dir ../jev9b-deploy/models/JEV-9B --dry-run
python demo.py --model-dir /path/to/JEV-9B --workload correlated --field-tokens 512
python demo.py --model-dir /path/to/JEV-9B --workload shared --repeat-ratios .80 .97 .983
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import itertools
import json
from pathlib import Path
import platform
import random
import uuid

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from solo_layout import DecisionEngine, DecisionSpec, LAYOUTS, LayoutOptimizer
from solo_layout._table import Serializer, as_table
from solo_layout._integer import PrefixGroups
from solo_layout.layout import encode_columns
import solo_layout

QUESTION = "Does the customer_request ask for a monetary refund?"
POLICY = ("Staff review the complete case history and document the next step. "
          "The customer record includes store information and service details. ")
REQUESTS = (
    "The item arrived broken. Please return the money I paid to my card.",
    "The item arrived broken. Please send a replacement of the same model.",
)


def provenance(model_dir):
    """Record the actual client and tokenizer used, without host credentials."""
    versions = {}
    for name in ("numpy", "numba", "pandas", "tokenizers", "requests"):
        versions[name] = metadata.version(name)
    package_root = Path(solo_layout.__file__).parent
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "client_version": solo_layout.__version__,
        "python_version": platform.python_version(),
        "client_dependencies": versions,
        "client_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(package_root.glob("*.py"))},
        "tokenizer_sha256": hashlib.sha256((model_dir / "tokenizer.json").read_bytes()).hexdigest(),
        "server_configuration": "Record the serving process configuration separately; the client cannot infer it from a URL.",
    }


def sized(text, budget, count):
    """Deterministic dummy padding; preserve the meaningful text at the front."""
    if count(text) > budget:
        raise ValueError("field token budget is shorter than its meaningful text")
    block = " " + POLICY
    text += block * max(0, (budget - count(text)) // count(block))
    while count(text + " background") <= budget:
        text += " background"
    if count(text) != budget:
        raise ValueError("tokenizer cannot produce the requested exact field length")
    return text


def correlated(rows, cardinality, related, tokens, seed, count):
    """Equal marginal NDV and token length; distinct balanced latent triples.

    A-family columns are bijections of A. Request and B are two other latent
    variables. Cyclic orbits give exactly uniform marginals without duplicate
    full rows. Input order interleaves the three families before the other A's.
    """
    if cardinality < 2 or cardinality % 2 or rows % cardinality or not cardinality <= rows <= cardinality**3:
        raise ValueError("use an even cardinality K, rows divisible by K, and K <= rows <= K^3")
    if related < 2:
        raise ValueError("correlated-columns must be at least 2")
    rng = random.Random(seed)
    orbits = list(itertools.product(range(cardinality), repeat=2))
    rng.shuffle(orbits)
    triples = [((t % cardinality), (b+t) % cardinality, (c+t) % cardinality)
               for b, c in orbits[:rows // cardinality] for t in range(cardinality)]
    rng.shuffle(triples)
    names = ["a_00", "customer_request", "b_00"] + [f"a_{i:02d}" for i in range(1, related)]
    templates = {}
    for name in names:
        templates[name] = [sized(
            (REQUESTS[0 if v < cardinality // 2 else 1] if name == "customer_request" else POLICY)
            + f" Profile {v:03d}.", tokens, count) for v in range(cardinality)]
    records, truth = [], []
    for a, b, c in triples:
        record = {}
        for name in names:
            v = b if name == "customer_request" else c if name == "b_00" else (a + int(name[2:])) % cardinality
            record[name] = templates[name][v]
        records.append(record)
        truth.append(b < cardinality // 2)
    frame = pd.DataFrame(records)
    assert len(frame.drop_duplicates()) == rows
    assert (frame.nunique() == cardinality).all()
    return frame, np.array(truth), {
        "family": "equal_ndv_correlated", "cardinality": cardinality,
        "correlated_columns": related, "tokens_per_field": tokens,
        "construction": "A columns are bijective copies of one latent variable; request and B use two other balanced variables. All columns have identical marginal NDV and token length. No identifier column.",
    }


def shared(rows, ratio, seed, count):
    """The earlier ID-first workload, with two enlarged constant fields."""
    if not .8 <= ratio < .995:
        raise ValueError("repeat ratios must be in [0.8, 0.995)")
    rng = random.Random(seed)
    unique_budget = 264
    long_budget = round((unique_budget * ratio / (1-ratio) - 6*132) / 2)
    names = ["company_policy", "service_context"] + [f"profile_{c}" for c in range(6)]
    cards = [1, 1, 4, 8, 16, 32, 64, 64]
    templates = {name: [sized(f"Profile {v:03d}. " + POLICY,
                             long_budget if c < 2 else 132, count) for v in range(cards[c])]
                 for c, name in enumerate(names)}
    labels = [i % 2 == 0 for i in range(rows)]
    rng.shuffle(labels)
    records = []
    for i in range(rows):
        identifier = "case-" + uuid.UUID(int=rng.getrandbits(128)).hex
        message = f"Customer Alex Chen, contact reference {i:06d}. " + REQUESTS[0 if labels[i] else 1]
        message = sized(message, unique_budget - count(identifier), count)
        records.append({"record_id": identifier, "customer_request": message,
                        **{name: values[i % len(values)] for name, values in templates.items()}})
    permutation = list(range(rows))
    rng.shuffle(permutation)
    return pd.DataFrame([records[i] for i in permutation]), np.asarray(labels)[permutation], {
        "family": "shared_fields", "target_reusable_value_token_fraction": ratio,
        "constant_field_token_budget": long_budget,
        "construction": "Unique ID and customer message first; eight reusable template fields, including two long global constants. Synthetic dummy padding; complete rows sent in every method.",
    }


def describe(frame, count, spec):
    cards = frame.nunique().to_dict()
    costs = {c: {value: count(value) for value in frame[c].unique()} for c in frame}
    total = sum(sum(costs[c][v] for v in frame[c]) for c in frame)
    reusable = sum(sum(costs[c][v] for v in frame[c]) for c in frame if cards[c] < len(frame))
    blob = frame.to_json(orient="records", force_ascii=False)
    return {**spec, "rows": len(frame), "columns": len(frame.columns), "synthetic": True,
            "dataset_sha256": hashlib.sha256(blob.encode()).hexdigest(),
            "column_ndv": cards, "all_column_ndv_equal": len(set(cards.values())) == 1,
            "field_token_ranges": {c: [min(v.values()), max(v.values())] for c, v in costs.items()},
            "reusable_value_token_fraction": reusable / total,
            "reuse_definition": "Fraction of cell-value tokens in columns whose whole-field NDV is less than row count; this is not the prefix-cache hit rate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--workload", choices=["correlated", "shared"], default="correlated")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--cardinality", type=int, default=8)
    parser.add_argument("--correlated-columns", type=int, default=12)
    parser.add_argument("--field-tokens", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--repeat-ratios", type=float, nargs="+", default=[.80, .97, .983])
    parser.add_argument("--methods", choices=LAYOUTS, nargs="+")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("demo-results.json"))
    args = parser.parse_args()
    if args.rows < 2:
        parser.error("rows must be at least 2")
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    methods = args.methods or (["original", "lexicographic", "random", "cardinality", "solo"]
                               if args.workload == "correlated" else ["original", "lexicographic", "random", "solo"])
    settings = args.field_tokens if args.workload == "correlated" else args.repeat_ratios
    report = {"configuration": {**vars(args), "model_dir": str(args.model_dir), "out": str(args.out), "methods": methods},
              "provenance": provenance(args.model_dir),
              "measurement": "Fresh cache_salt per trial, identical backend and concurrency; alternating method order. Timed scans include normalization, planning, serialization, network and initial cache fills. Kernel/planner warmup is separate.",
              "cases": []}
    with DecisionEngine(args.url, model_dir=args.model_dir, concurrency=args.concurrency) as engine:
        for setting in settings:
            if args.workload == "correlated":
                frame, truth, spec = correlated(args.rows, args.cardinality, args.correlated_columns, setting, args.seed, count)
            else:
                frame, truth, spec = shared(args.rows, setting, args.seed, count)
            info = describe(frame, count, spec)
            codes = encode_columns(frame.to_numpy(dtype=object))
            table = as_table(frame)
            decision_spec = DecisionSpec.create(QUESTION)
            largest_prompt = 0
            info["prefix_distinct_counts"] = {}
            # Warm the actual full-table planner signatures before timings.
            for method in methods:
                plan = LayoutOptimizer(method, seed=args.seed).plan(frame)
                state = PrefixGroups(codes)
                info["prefix_distinct_counts"][method] = [state.refine(int(c)) for c in plan.column_order]
                serialize = Serializer(table, plan.column_order)
                largest_prompt = max(largest_prompt, max(count(decision_spec.prompt(serialize(i))) for i in range(len(frame))))
            info["max_prompt_tokens_across_methods"] = largest_prompt
            if largest_prompt + 1 > args.max_context_tokens:
                parser.error(f"workload needs {largest_prompt + 1} tokens; increase server context and --max-context-tokens, or reduce field lengths")
            print(f"{info['family']}: {len(frame)} rows, {len(frame.columns)} columns, "
                  f"reusable field tokens {info['reusable_value_token_fraction']:.1%}, "
                  f"max prompt {largest_prompt} tokens, equal NDV={info['all_column_ndv_equal']}", flush=True)
            if args.workload == "correlated":
                for method in ("cardinality", "solo"):
                    if method in info["prefix_distinct_counts"]:
                        print(f"  {method} prefix counts: {info['prefix_distinct_counts'][method]}", flush=True)
            case = {"workload": info}
            if not args.dry_run:
                engine.scan(frame.iloc[:4], QUESTION, method="original", cache_salt=uuid.uuid4().hex)
                result = engine.compare(frame, QUESTION, methods=methods, repeats=args.repeats,
                                        truth=truth, seed=args.seed)
                case.update(summary=result.summary, runs=result.runs)
                print(result.to_pandas()[["rows_per_second", "cached_fraction", "accuracy",
                                         "solo_speedup_vs_this_method"]].to_string(), flush=True)
            report["cases"].append(case)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.out.with_name(args.out.name + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.out)


if __name__ == "__main__":
    main()

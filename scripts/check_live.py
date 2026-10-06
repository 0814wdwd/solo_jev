#!/usr/bin/env python3
"""Check installed-client inputs and all three heads against a real JEV API.

This is a small integration check, not a throughput or model-quality benchmark.
Requires the pandas extra. No model weights are loaded by this process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import uuid

import numpy as np
import pandas as pd

import solo_decision
from solo_decision import DecisionEngine, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("live-check.json"))
    args = parser.parse_args()
    flat = [
        {"id": "T-002", "request": "Please send a replacement item. I do not want a refund."},
        {"id": "T-001", "request": "Please refund the money I paid back to my card."},
    ]
    nested = [{**row, "customer": {"tier": "gold", "active": True},
               "events": ["ordered", "delivered"], "amount": 24.5, "review": None}
              for row in flat]
    question = "Does the customer's request ask for a monetary refund?"
    expected = [False, True]
    checks = []
    with tempfile.TemporaryDirectory() as directory:
        jsonl = Path(directory) / "tickets.jsonl"
        jsonl.write_text("\n".join(json.dumps(row) for row in nested) + "\n", encoding="utf-8")
        frame = pd.DataFrame(flat, index=pd.Index(["same", "same"], name="ticket"))
        inputs = [
            ("pandas_duplicate_index", frame, {}),
            ("numpy", frame.to_numpy(), {"columns": list(frame.columns)}),
            ("record_list", flat, {}),
            ("nested_json_text", json.dumps(nested), {}),
            ("jsonl_path", jsonl, {}),
            ("parsed_json", read_json(nested), {}),
        ]
        with DecisionEngine(args.url, model_dir=args.model_dir) as engine:
            for name, data, kwargs in inputs:
                result = engine.scan(data, question, cache_salt=uuid.uuid4().hex, **kwargs)
                assert result.decisions.tolist() == expected, (name, result.decisions.tolist())
                if name == "pandas_duplicate_index":
                    pd.testing.assert_index_equal(result.to_pandas().index, frame.index)
                checks.append({"input": name, "decisions": result.decisions.tolist(),
                               "probabilities": result.probabilities.tolist(), "passed": True})
            choice = engine.scan(read_json(nested), "Which outcome does the customer request?",
                                 kind="choice", options=["replacement", "monetary refund"],
                                 cache_salt=uuid.uuid4().hex)
            assert choice.decisions.tolist() == ["replacement", "monetary refund"]
            score = engine.scan(read_json(nested),
                                "Rate how explicitly the request asks for a monetary refund, from 0 to 5.",
                                kind="score", cache_salt=uuid.uuid4().hex)
            assert score.probabilities.shape == (2, 6)
            assert np.all((score.decisions >= 0) & (score.decisions <= 5))
    package = Path(solo_decision.__file__).parent
    report = {
        "client_version": solo_decision.__version__, "python_version": platform.python_version(),
        "purpose": "Small live integration check; not a throughput or quality benchmark",
        "inputs": checks, "choice": choice.decisions.tolist(), "score": score.decisions.tolist(),
        "all_three_heads_valid": True, "all_input_checks_passed": True,
        "client_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(package.glob("*.py"))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {len(checks)} input checks and all three decision heads. Report: {args.out}")


if __name__ == "__main__":
    main()

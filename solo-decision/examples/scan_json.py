"""Optional live inference: python examples/scan_json.py --model-dir /path/to/JEV-9B."""
import argparse
import json
from pathlib import Path

from solo_decision import DecisionEngine, read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--input", type=Path, help="Your JSON or JSONL records; otherwise use the example tickets")
    args = p.parse_args()
    records = args.input if args.input else read_json([
        {"id": 104, "policy": {"returns": "30 days", "region": "north"}, "tags": ["billing"], "request": "Please refund my order."},
        {"id": 105, "policy": {"returns": "30 days", "region": "north"}, "tags": ["exchange"], "request": "Can I exchange the size?"},
    ])
    with DecisionEngine(base_url=args.url, model_dir=args.model_dir) as engine:
        result = engine.scan(records, "Does the request ask for a monetary refund?")
    print(json.dumps({"decisions": result.decisions.tolist(), "probabilities": result.probabilities.tolist(), "metrics": result.metrics()}, indent=2))


if __name__ == "__main__":
    main()

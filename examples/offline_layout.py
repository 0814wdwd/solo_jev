"""Inspect and apply a layout without a GPU, model, tokenizer, or network."""
from pprint import pprint

from solo_layout import LayoutOptimizer, read_json

records = read_json([
    {"id": 104, "policy": "Refunds within 30 days", "account": {"tier": "gold"}, "request": "Please refund my order."},
    {"id": 201, "policy": "Exchanges within 60 days", "account": {"tier": "silver"}, "request": "Please exchange the size."},
    {"id": 105, "policy": "Refunds within 30 days", "account": {"tier": "gold"}, "request": "I would like my money back."},
    {"id": 202, "policy": "Exchanges within 60 days", "account": {"tier": "silver"}, "request": "Can I replace this item?"},
])

optimizer = LayoutOptimizer("solo")
plan = optimizer.plan(records)
print("Planned field order:", plan.ordered_columns)
print("Planned row positions:", plan.row_order.tolist())
pprint(list(plan.apply(records)), sort_dicts=False)
print("\nLayout diagnostics (bytes describe structure, not predicted speedup):")
pprint(optimizer.explain(records).to_dict(), sort_dicts=False)

# If a backend returns one prediction per planned row, restore by position:
planned_ids = [104, 201, 105, 202]
restored_ids = plan.restore([planned_ids[i] for i in plan.row_order])
assert restored_ids.tolist() == planned_ids
print("\nOutputs restored to input order:", restored_ids.tolist())

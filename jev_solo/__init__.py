"""SOLO for Jev: prefix-cache-aware reordering, retargeted at a token budget.

Jev bills input tokens only and documents no cross-request cache, so SOLO's
mechanism (cross-request prefix KV reuse) does not transfer. What transfers is
the combinatorial core: G^(k), weighted by per-column token cost.
"""
from .tokens import get_counter, column_value_weights  # noqa: F401
from .encodings import encode, ENCODINGS  # noqa: F401
from .plan import plan, plan_columns, PLANNERS, apply_order, lex_sort_rows  # noqa: F401
from .objective import analytic_factored_cost, measured_cost, validate, encode_columns  # noqa: F401
from .api import Scan, ScanResult  # noqa: F401
from .recalibrate import Calibrator, fit as fit_calibrator  # noqa: F401
from . import datasets  # noqa: F401

__all__ = ["Scan", "ScanResult", "Calibrator", "fit_calibrator", "datasets",
           "encode", "ENCODINGS", "plan", "PLANNERS", "get_counter"]

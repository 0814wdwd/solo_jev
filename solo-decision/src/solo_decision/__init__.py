"""Structured Input Layout Optimization for Decision Models."""
from .layout import LAYOUTS, LayoutExplanation, LayoutOptimizer, LayoutPlan
from ._table import JSONInput, read_json

__version__ = "0.2.0"
__all__ = ["LayoutOptimizer", "LayoutPlan", "LAYOUTS", "JSONInput", "read_json",
           "LayoutExplanation"]

"""Structured Input Layout Optimization for Decision Models."""
from .backend import DecisionResponse, DecisionSpec, JevBackend
from .engine import ComparisonResult, DecisionEngine, ScanResult
from .layout import LAYOUTS, LayoutExplanation, LayoutOptimizer, LayoutPlan
from ._table import JSONInput, read_json

__version__ = "0.2.0"
__all__ = ["DecisionEngine", "ScanResult", "ComparisonResult", "JevBackend",
           "DecisionSpec", "DecisionResponse", "LayoutOptimizer", "LayoutPlan", "LAYOUTS",
           "JSONInput", "read_json", "LayoutExplanation"]

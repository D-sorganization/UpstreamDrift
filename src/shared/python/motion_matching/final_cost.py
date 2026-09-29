"""Backward-compat alias: final_cost -> cost."""

from src.shared.python.motion_matching.cost import (
    CostBreakdown,
    CostOptions,
    SimOutput,
    compute_cost,
    compute_total_work,
)

__all__ = [
    "CostBreakdown",
    "CostOptions",
    "SimOutput",
    "compute_cost",
    "compute_total_work",
]

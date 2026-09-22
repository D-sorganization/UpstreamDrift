"""Hybrid native refinement boundary for masked proposals (NM-06)."""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.motion_matching.inverse.proposal_shared import (
    parse_native_polish_outcome,
)

__all__ = [
    "ProposalPolishResult",
    "RefinePolishFn",
    "refine_proposal_hybrid",
]

RefinePolishFn = Callable[[object, np.ndarray], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ProposalPolishResult:
    controls: np.ndarray
    projection_cost: float
    independent_replay: bool
    polish_phase: dict[str, Any]
    duration_s: float


def refine_proposal_hybrid(
    *,
    target: object,
    proposal_controls: np.ndarray,
    polish_fn: RefinePolishFn,
    require_independent_replay: bool = True,
    expected_control_dim: int | None = None,
) -> ProposalPolishResult:
    """Fail-closed polish with dependency-injected native solver mapping."""
    warm = np.asarray(proposal_controls, dtype=np.float64).reshape(-1)
    if warm.size < 1 or not bool(np.all(np.isfinite(warm))):
        raise ValueError("proposal_controls must be a non-empty finite vector")
    if expected_control_dim is not None and warm.size != int(expected_control_dim):
        raise ValueError(
            f"proposal_controls length must be {expected_control_dim}, got {warm.size}"
        )

    t0 = _time.perf_counter()
    polish_out = polish_fn(target, warm)
    polished, cost, independent, polish_map = parse_native_polish_outcome(
        polish_out,
        warm=warm,
        require_independent_replay=require_independent_replay,
    )

    return ProposalPolishResult(
        controls=polished,
        projection_cost=cost,
        independent_replay=independent,
        polish_phase=polish_map,
        duration_s=_time.perf_counter() - t0,
    )

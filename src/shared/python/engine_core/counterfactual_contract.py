"""Typed counterfactual results for ZTCF/ZVCF (SPEC §4, issue #14943)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = [
    "COUNTERFACTUAL_DEFINITION_VERSION",
    "CounterfactualKind",
    "CounterfactualResult",
    "ensure_comparable_counterfactuals",
    "wrap_counterfactual",
]

# Frozen label stamped into exports; bump when SPEC §4 semantics change.
COUNTERFACTUAL_DEFINITION_VERSION = "spec-section-4-fixed-state-v1"


class CounterfactualKind(str, Enum):
    """Physical quantity encoded in a counterfactual acceleration sample."""

    DRIFT = "drift"  # ZTCF: passive plant at fixed state (τ = 0)
    CONTROL = "control"  # ZVCF: M(q)⁻¹ τ with velocity/gravity terms zeroed
    GRAVITY_ONLY = "gravity_only"  # legacy diagnostic; not SPEC ZVCF


@dataclass(frozen=True)
class CounterfactualResult:
    """Acceleration sample with explicit counterfactual kind and definition version."""

    kind: CounterfactualKind
    definition_version: str
    qdd: np.ndarray
    frame: str = "model_root"
    units: str = "rad/s^2"

    def __post_init__(self) -> None:
        qdd = np.asarray(self.qdd, dtype=float)
        if not np.all(np.isfinite(qdd)):
            raise ValueError("CounterfactualResult.qdd must be finite")
        object.__setattr__(self, "qdd", qdd)


def wrap_counterfactual(
    kind: CounterfactualKind,
    qdd: np.ndarray,
    *,
    frame: str = "model_root",
    units: str = "rad/s^2",
    definition_version: str = COUNTERFACTUAL_DEFINITION_VERSION,
) -> CounterfactualResult:
    """Build a stamped result using the current definition version."""
    return CounterfactualResult(
        kind=kind,
        definition_version=definition_version,
        qdd=qdd,
        frame=frame,
        units=units,
    )


def ensure_comparable_counterfactuals(
    left: CounterfactualResult, right: CounterfactualResult
) -> None:
    """Refuse comparisons that mix counterfactual kinds or definition versions."""
    if left.kind != right.kind:
        raise ValueError(
            f"Cannot compare counterfactuals: {left.kind.value} vs {right.kind.value}"
        )
    if left.definition_version != right.definition_version:
        raise ValueError(
            "Cannot compare counterfactuals with mixed definition_version "
            f"({left.definition_version!r} vs {right.definition_version!r})"
        )

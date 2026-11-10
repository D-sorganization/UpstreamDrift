"""Typed counterfactual results for ZTCF/ZVCF (SPEC §4, issue #14943)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

COUNTERFACTUAL_DEFINITION_VERSION = "spec-section-4-fixed-state-v1"


class CounterfactualKind(str, Enum):
    DRIFT = "drift"
    CONTROL = "control"
    GRAVITY_ONLY = "gravity_only"


@dataclass(frozen=True)
class CounterfactualResult:
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
    return CounterfactualResult(kind, definition_version, qdd, frame, units)


def ensure_comparable_counterfactuals(
    left: CounterfactualResult, right: CounterfactualResult
) -> None:
    if left.kind != right.kind:
        raise ValueError(
            f"Cannot compare counterfactuals: {left.kind.value} vs {right.kind.value}"
        )
    if left.definition_version != right.definition_version:
        raise ValueError(
            f"Cannot compare counterfactuals with mixed definition_version ({left.definition_version!r} vs {right.definition_version!r})"
        )

"""Typed contracts for NM-05 dynamics baselines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "BASELINE_SCHEMA",
    "DynamicsTaskKind",
    "InverseLabelConditioning",
]

BASELINE_SCHEMA = "neural-dynamics-baselines/1.0.0"


class DynamicsTaskKind(str, Enum):
    """Supervised dynamics tasks trained separately (issue #10620)."""

    FORWARD_ACCELERATION = "forward_acceleration"
    FORWARD_NEXT_STATE = "forward_next_state"
    INVERSE_CONTROL = "inverse_control"


@dataclass(frozen=True, slots=True)
class InverseLabelConditioning:
    """Contact / actuation / allocation assumptions for inverse labels.

    Design by Contract:
    - All fields are non-empty strings.
    - A ``(q,v,a)->tau`` map without reaction assumptions remains ambiguous;
      callers must declare an allocation policy explicitly.
    """

    contact_mode: str
    actuation_mode: str
    allocation_policy: str

    def __post_init__(self) -> None:
        for name, value in (
            ("contact_mode", self.contact_mode),
            ("actuation_mode", self.actuation_mode),
            ("allocation_policy", self.allocation_policy),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

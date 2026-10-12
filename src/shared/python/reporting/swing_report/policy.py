"""Quantity output policy for the swing report contract (#12324, #10382).

Fail-closed: only the three advertised quantities may carry values, and only
together with provisional cross-engine-spread uncertainty. Everything else,
including any id not listed here, is withheld as "not validated". Promoting a
quantity out of ``WITHHELD`` needs an owner ruling and a MAJOR schema bump.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final


class QuantityClass(str, Enum):
    """Output class of a report quantity."""

    ADVERTISED = "advertised_with_uncertainty"
    WITHHELD = "withheld"


STATUS_ADVERTISED: Final[str] = QuantityClass.ADVERTISED.value
STATUS_WITHHELD: Final[str] = QuantityClass.WITHHELD.value
STATUS_VALIDATED: Final[str] = "validated"  # reserved for DV-10382; rejected in v1

NOT_VALIDATED_LABEL: Final[str] = "not validated"
UNCERTAINTY_KIND: Final[str] = "engine_spread"
UNCERTAINTY_BUDGET_SOURCE: Final[str] = "provisional_cross_engine_spread"

_ADVERTISED_IDS: Final[tuple[str, ...]] = (
    "clubhead_position",
    "clubhead_speed",
    "kinematic_sequence",
)
_WITHHELD_IDS: Final[tuple[str, ...]] = (
    "face_angle",
    "joint_load",
    "joint_torque",
    "ground_reaction_force",
    "impact_state",
    "ball_flight",
    "joint_angle",
    "segment_angular_velocity",
)

POLICY: Final = MappingProxyType(
    {
        **dict.fromkeys(_ADVERTISED_IDS, QuantityClass.ADVERTISED),
        **dict.fromkeys(_WITHHELD_IDS, QuantityClass.WITHHELD),
    }
)
ADVERTISED: Final[frozenset[str]] = frozenset(_ADVERTISED_IDS)
WITHHELD: Final[frozenset[str]] = frozenset(_WITHHELD_IDS)


def classify(quantity_id: str) -> QuantityClass:
    """Return the output class of ``quantity_id``; unknown ids are withheld."""
    return POLICY.get(quantity_id, QuantityClass.WITHHELD)

"""Wrist and forearm entries for the golf-term registry (spine/golf_terms.py).

This module only describes the terms and builds their evaluators; the single
registry (``golf_terms._TERMS``) stays the only place terms are registered.
Inputs are relative rotations in the Wu / ISB-elbow frames.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np

from src.shared.python.anatomical_angles.wrist import OBSERVABILITY, decompose_relative
from src.shared.python.core.handedness import lead_side, trail_side

WRIST_INPUT = "hand_in_radius_rotation"
FOREARM_INPUT = "forearm_in_humerus_rotation"
_WU = "Wu et al. 2005, J Biomech 38:981 (ISB part II) 4.3-4.4"


class WristTermSpec(NamedTuple):
    term_id: str
    role: str  # "lead" or "trail"
    profile: str
    component: int
    description: str
    frame: str
    sign_convention: str
    required_input: str


_WRIST_COMPONENTS = (
    ("wrist_flexion_extension", 0, "flexion positive, extension negative"),
    ("wrist_ulnar_deviation", 1, "ulnar positive, radial negative"),
    ("wrist_axial_rotation", 2, "pronation sense positive (third wrist angle)"),
)


def wrist_term_specs() -> tuple[WristTermSpec, ...]:
    """Specs of the lead/trail wrist and forearm terms, in registry order."""
    specs: list[WristTermSpec] = []
    for role in ("lead", "trail"):
        for name, idx, sign in _WRIST_COMPONENTS:
            specs.append(
                WristTermSpec(
                    f"{role}_{name}",
                    role,
                    "isb_wrist",
                    idx,
                    f"{role.capitalize()}-hand {name.replace('_', ' ')}; "
                    f"observability = {OBSERVABILITY} (model-estimated, not "
                    "measured hand anatomy)",
                    "isb_wrist ZXY on R_radius->hand; X volar, Y proximal, Z=XxY",
                    sign + "; same sense both hands, handedness picks the side",
                    WRIST_INPUT,
                )
            )
        specs.append(
            WristTermSpec(
                f"{role}_forearm_pronation",
                role,
                "isb_elbow",
                2,
                f"{role.capitalize()}-hand forearm pronation, separate from the "
                f"wrist third angle; observability = {OBSERVABILITY}",
                "isb_elbow ZXY on R_humerus->forearm",
                "pronation positive, supination negative",
                FOREARM_INPUT,
            )
        )
    return tuple(specs)


def wrist_term_reference() -> str:
    return _WU


def evaluator(
    spec: WristTermSpec,
) -> Callable[[Mapping[str, Any], str], np.ndarray]:
    """``(inputs, handedness) -> degrees`` for ``spec``; NaN where singular."""

    def fn(inputs: Mapping[str, Any], handedness: str) -> np.ndarray:
        side = lead_side(handedness) if spec.role == "lead" else trail_side(handedness)
        angles, _ = decompose_relative(inputs[spec.required_input], spec.profile, side)
        return np.rad2deg(angles[:, spec.component])

    return fn

"""Arms that reach a lift bundle's bar for the OpenSim lift IK (OSL-12, #13048).

The OSL-2 squat hold (:data:`.msk_lift_model.HOLD_POSES`) is a back rack: the
hands sit behind the shoulders, putting the bar 0.194 m behind and 0.073 m
above the shoulder-centre midpoint.  The LIFT-5 back-squat bundle's hands hold
its bar 0.157 m in front of and 0.159 m above that midpoint (the torso shelf
that ``bundles`` fits is not an anatomical site).  The bar is 0.25 m from each
shoulder, against a Rajagopal reach of 0.636 m.  The OpenSim IK starts from
the hold, and its weak arm tasks hold it there, so the hands never leave the
back rack and the bar marker stays 0.18-0.28 m off.

:func:`reaching_lift_build` therefore runs two passes:

1. **Seed pass.**  The hold-pose model's IK with the bar unweighted, so the
   bar cannot pull the trunk, gives the body pose at the first state.
2. **Reach solve.**  :func:`reaching_arm_coordinates` holds that body and
   solves each arm's shoulder and elbow coordinates
   (:data:`ARM_REACH_COORDINATES`), bounded by the model ranges, so the hand's
   centre of mass (the weld's grip point) lies on the bundle's grip point for
   that hand.  The forearm and wrist stay at the hold.

The lift model is then rebuilt with those arms as its hold
(``build_lift_model(arm_coordinates=...)``), so the bar weld is placed at the
reaching pose, and the IK tracks the bar at full weight.

OpenSim is imported lazily; callers check availability first.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
from src.engines.physics_engines.opensim.python.msk_lift_model import (
    ARM_COORDINATE_PREFIXES,
    LiftModelBuild,
    build_lift_model,
)
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.lifting.bundle_targets import LiftTargets

Array = NDArray[np.float64]

__all__ = [
    "ARM_REACH_COORDINATES",
    "REACH_PRIOR_WEIGHT",
    "ArmReach",
    "reaching_arm_coordinates",
    "reaching_lift_build",
]

#: Arm coordinates the reach solve moves, per side (``_r`` / ``_l`` suffix).
ARM_REACH_COORDINATES: tuple[str, ...] = (
    "arm_flex",
    "arm_add",
    "arm_rot",
    "elbow_flex",
)
#: Weight (m per rad) pulling the solve towards the hold pose.  Three
#: position equations leave one arm freedom open; this small prior picks the
#: solution nearest the hold, and costs well under a millimetre of reach.
REACH_PRIOR_WEIGHT = 1e-3
#: The bundle's grip of each Rajagopal hand (OpenSim ``_l`` / ``_r``).
_GRIP_OF = {"l": "L", "r": "R"}
_REACH_XTOL = 1e-12


def _vec3(v: Any) -> Array:
    return np.array([float(v.get(i)) for i in range(3)], dtype=float)


@dataclass(frozen=True)
class ArmReach:
    """Arm coordinates (radians) putting the hands on the bundle's grips.

    ``reach_error_m`` maps ``L``/``R`` to the distance, at state ``frame``,
    from the hand's centre of mass to its bundle grip.
    """

    coordinates: dict[str, float]
    reach_error_m: dict[str, float]
    frame: int

    def record(self) -> dict[str, Any]:
        """The IK receipt's ``arm_reach`` entry."""
        return {
            "method": "seed pass: OpenSim IK of the hold-pose model with the "
            "bar unweighted; then, with that body pose held, bounded least "
            "squares (scipy TRF, model ranges) on "
            + "/".join(ARM_REACH_COORDINATES)
            + " per side so each hand's centre of mass lies on the bundle's "
            f"grip point (prior weight {REACH_PRIOR_WEIGHT} m/rad to the hold)",
            "frame": self.frame,
            "coordinates_rad": dict(self.coordinates),
            "reach_error_m": dict(self.reach_error_m),
        }


def _hand_centre(model: Any, state: Any, side: str) -> Array:
    hand = model.getBodySet().get(f"hand_{side}")
    return _vec3(hand.findStationLocationInGround(state, hand.getMassCenter()))


def _solve_side(
    model: Any, state: Any, side: str, target: Array
) -> tuple[dict[str, float], float]:
    """Bounded reach of one hand; the rest of ``state`` is held."""
    coords = model.getCoordinateSet()
    names = [f"{name}_{side}" for name in ARM_REACH_COORDINATES]
    low = np.array([coords.get(n).getRangeMin() for n in names])
    high = np.array([coords.get(n).getRangeMax() for n in names])
    prior = np.array([coords.get(n).getDefaultValue() for n in names])
    start = np.clip(prior, low, high)

    def residual(x: Array) -> Array:
        for name, value in zip(names, x, strict=True):
            coords.get(name).setValue(state, float(value), False)
        model.realizePosition(state)
        miss = _hand_centre(model, state, side) - target
        return np.concatenate([miss, REACH_PRIOR_WEIGHT * (x - start)])

    fit = least_squares(
        residual, start, bounds=(low, high), xtol=_REACH_XTOL, ftol=_REACH_XTOL
    )
    error = float(np.linalg.norm(residual(fit.x)[:3]))
    return dict(zip(names, map(float, fit.x), strict=True)), error


def reaching_arm_coordinates(
    build: LiftModelBuild, targets: LiftTargets, seed: ikid.IKResult
) -> ArmReach:
    """Arm coordinates that put each hand centre on its bundle grip at state 0.

    The body is held at the seed IK's first state; only
    :data:`ARM_REACH_COORDINATES` move, inside the model ranges.  Constraints
    (the bar weld) are not enforced while solving.

    Postconditions: every returned coordinate lies inside its model range;
    ``reach_error_m`` holds both hands.

    Raises:
        ValueError: when ``seed`` does not name the model's coordinates.
    """
    model = build.model
    state = model.initSystem()
    coords = model.getCoordinateSet()
    if set(seed.coordinates) != {c.getName() for c in coords}:
        raise ValueError("seed IK coordinates do not match the model")
    for j, name in enumerate(seed.coordinates):
        coordinate = coords.get(name)
        # The arm chain restarts from the hold, as the rebuilt model will.
        held = name.startswith(ARM_COORDINATE_PREFIXES)
        value = coordinate.getDefaultValue() if held else float(seed.q[0, j])
        coordinate.setValue(state, value, False)
    solved: dict[str, float] = {}
    errors: dict[str, float] = {}
    for side, grip in _GRIP_OF.items():
        values, errors[grip] = _solve_side(model, state, side, targets.grip_m[grip][0])
        solved.update(values)
    return ArmReach(coordinates=solved, reach_error_m=errors, frame=0)


def reaching_lift_build(
    lift: str,
    targets: LiftTargets,
    base_model_path: str | Path,
    segment_scales: Mapping[str, Any],
) -> tuple[LiftModelBuild, ArmReach]:
    """The lift model with its hold arms seeded on the bundle's grips.

    Runs the seed pass (hold-pose model, bar unweighted), solves the reach
    (:func:`reaching_arm_coordinates`) and rebuilds the model with those arms.

    Postcondition: the returned model's arm defaults equal the reach solve's.
    """
    common: dict[str, Any] = {
        "body_mass_kg": targets.body_mass_kg,
        "height_m": targets.height_m,
        "bar_mass_kg": targets.bar_mass_kg,
        "base_model_path": base_model_path,
        "segment_scales": segment_scales,
        "tibial_torsion": targets.tibial_torsion,
    }
    seed_build = build_lift_model(lift, **common)
    pelvis = ikid.pelvis_landmark_targets(seed_build, targets, base_model_path)
    seed = ikid.run_inverse_kinematics(seed_build, targets, pelvis, bar_weight=0.0)
    reach = reaching_arm_coordinates(seed_build, targets, seed)
    if not all(math.isfinite(v) for v in reach.coordinates.values()):
        raise ValueError(f"non-finite reach solution {reach.coordinates}")
    build = build_lift_model(lift, **common, arm_coordinates=reach.coordinates)
    return build, reach

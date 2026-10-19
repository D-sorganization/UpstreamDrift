"""The OpenSim lift foot fitted to a same-input bundle's foot (OSL-18, #13157).

Two properties of the bundle foot that the generic Rajagopal foot lacks:

* **Size.**  :func:`bundle_foot_scale` -- the bundle's heel and forefoot
  landmarks are its own calcaneus stations, so their distance over the
  stations' generic separation sizes the foot (talus, calcaneus, toes).  The
  squat bundle's foot is the generic size (0.150 m, factor 1.000) while its
  knee-to-ankle ratio, which scaled the foot before, is 1.10.
* **Torsion.**  :func:`apply_tibial_torsion`, below.

A same-input lift bundle grafts the Rajagopal lower limbs and then turns each
foot out by a static tibial-torsion offset
(:mod:`src.shared.python.motion_matching.tibial_torsion`, #12814; default
10 deg, Seber et al. 2000, doi:10.1007/s004020050459, CT tibiofibular torsion
of 50 adult men, 16-50 deg).  The spec rotates the ankle joint's parent frame
on the tibia about the shank long axis (knee centre to ankle centre) through
the ankle centre, so the talus, calcaneus, toes, their stations and the ankle
axis all turn with it.

The OpenSim lift model is the same generic Rajagopal leg, which carries only
about 6 deg of torsion between its knee and ankle axes
(``tibial_torsion.GENERIC_LEG_TORSION_DEG``).  :func:`apply_tibial_torsion`
applies the bundle's recorded offset to the model the same way: it rotates
each ``ankle_<side>`` parent offset frame (``tibia_<side>_offset``) about the
model's own shank axis through the ankle centre.  The joint's child frame and
coordinates are untouched, so the coordinate set is unchanged and the foot at
zero ankle and subtalar angle sits in the tibia as the bundle's foot does.

OpenSim conventions used (RajagopalLaiUhlrich2023): every body frame has
``x`` anterior, ``y`` superior and ``z`` to the subject's right, on both legs;
the ``walker_knee_<side>`` child frame is located on the tibia at the knee
centre; the ``ankle_<side>`` parent frame is located on the tibia at the ankle
centre.  A torsion offset is external: the toes turn laterally (``+z`` on the
right leg, ``-z`` on the left).

OpenSim is imported lazily; callers check availability first.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.lifting.bundle_targets import LiftTargets
from src.shared.python.motion_matching.tibial_torsion import (
    SIDES,
    TIBIAL_TORSION_KEY,
    validate_tibial_torsion,
)

__all__ = ["apply_tibial_torsion", "bundle_foot_scale", "shank_axis_in_tibia"]

#: OpenSim body-fixed Euler sequence of a ``PhysicalOffsetFrame`` orientation.
_EULER: Final[Literal["XYZ"]] = "XYZ"
#: Lateral direction of each leg in its tibia frame (``z`` is the right).
_LATERAL = {"r": np.array([0.0, 0.0, 1.0]), "l": np.array([0.0, 0.0, -1.0])}


def bundle_foot_scale(targets: LiftTargets, side: str) -> float:
    """Bundle foot length over the generic calcaneus station separation.

    The heel and forefoot landmarks are the spec's own calcaneus stations
    (``targets.foot_stations_m``), so their bundle distance over the stations'
    separation is the size of the bundle's calcaneus relative to the generic
    one.  It also sizes the talus and toes, which carry no landmark of their
    own (one rigid foot scale, as the shank factor scaled them before).

    Postcondition: the factor is positive and finite.

    Raises:
        ValueError: when the two stations coincide or the factor is degenerate.
    """
    heel = np.asarray(targets.foot_stations_m[f"heel_{side}"][1], dtype=float)
    forefoot = np.asarray(targets.foot_stations_m[f"forefoot_{side}"][1], dtype=float)
    separation = float(np.linalg.norm(heel - forefoot))
    points = targets.landmarks(0)
    length = float(np.linalg.norm(points[f"heel_{side}"] - points[f"forefoot_{side}"]))
    factor = length / separation if separation > 0.0 else math.nan
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"degenerate {side} foot scale {length} / {separation}")
    return factor


def _vec(v: Any) -> np.ndarray:
    return np.array([v.get(0), v.get(1), v.get(2)], dtype=float)


def _tibia_frame(model: Any, joint_name: str, side: str) -> Any:
    """The joint's offset frame that lives on ``tibia_<side>``."""
    joint = model.updJointSet().get(joint_name)
    name = f"tibia_{side}_offset"
    for i in range(joint.getPropertyByName("frames").size()):
        frame = joint.upd_frames(i)
        if frame.getName() == name:
            return frame
    raise ValueError(f"joint {joint_name!r} has no frame {name!r} on the tibia")


def shank_axis_in_tibia(model: Any, side: str) -> np.ndarray:
    """Unit shank long axis (ankle centre to knee centre) in the tibia frame."""
    knee = _vec(_tibia_frame(model, f"walker_knee_{side}", side).get_translation())
    ankle = _vec(_tibia_frame(model, f"ankle_{side}", side).get_translation())
    axis = knee - ankle
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"walker_knee_{side} and ankle_{side} centres coincide")
    return axis / norm


def _turn(model: Any, side: str, offset_deg: float) -> None:
    """Rotate ``ankle_<side>``'s tibia frame by an external torsion offset."""
    osim = import_opensim()
    frame = _tibia_frame(model, f"ankle_{side}", side)
    base = Rotation.from_euler(_EULER, _vec(frame.get_orientation()))
    up = shank_axis_in_tibia(model, side)
    toes = base.as_matrix()[:, 0]
    sign = 1.0 if float(np.cross(up, toes) @ _LATERAL[side]) > 0.0 else -1.0
    turn = Rotation.from_rotvec(sign * math.radians(offset_deg) * up)
    frame.set_orientation(osim.Vec3(*(turn * base).as_euler(_EULER)))


def apply_tibial_torsion(
    model: Any, record: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Turn each foot of ``model`` out by a bundle's tibial-torsion record.

    ``record`` is a spec's ``tibial_torsion_deg`` entry (``{"r", "l",
    "source"}``) or ``None`` / empty for the generic leg (model untouched).
    Mutates ``model`` and calls ``finalizeFromProperties`` so the edited
    frames survive ``initSystem``.

    Postcondition: returns a copy of the applied record, or ``None``.

    Raises:
        ValueError: when an offset is outside the cited adult range or the
            record has no source (``tibial_torsion.validate_tibial_torsion``),
            or the model lacks the Rajagopal knee and ankle tibia frames.
    """
    if not record:
        return None
    validate_tibial_torsion({TIBIAL_TORSION_KEY: record})
    for side in SIDES:
        _turn(model, side, float(record[side]))
    model.finalizeFromProperties()
    return {**{side: float(record[side]) for side in SIDES}, "source": record["source"]}

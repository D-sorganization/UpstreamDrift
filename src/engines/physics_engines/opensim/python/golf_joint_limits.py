"""Joint-range overrides of the golf humanoid models (single source, #12315).

The golf models start from the Rajagopal2015 OpenSense skeleton and widen a few
coordinate ranges for the swing (wrist deviation, #10003; shoulder flexion;
lumbar rotation) and, since #12315, the forearm. These ranges live here, once:
``scripts/build_humanoid_osim.py`` writes them into the ``.osim`` files and the
address calibration (``msk_club_calibration``) derives its static address
limits from them. No other physics engine models forearm pronation: the
full-body spec used by the MuJoCo, Drake and Pinocchio generators has no
radioulnar joint.

Forearm convention (Rajagopal ``pro_sup_r`` / ``pro_sup_l``): the coordinate is
the radioulnar angle, positive toward pronation. Zero is the anatomical
position (forearm fully supinated, palm forward with the arm hanging), so
:data:`FOREARM_NEUTRAL_RAD` (pi/2) is the neutral thumb-forward forearm. The
Rajagopal range ``[0, pi/2]`` therefore allowed supination and no pronation at
all, which pinned the lead forearm of every golf grip at its neutral limit.

Source of the new range: American Academy of Orthopaedic Surgeons normative
active range of motion (Greene and Heckman, *The Clinical Measurement of Joint
Motion*, AAOS, 1994): forearm pronation 80 deg and supination 80 deg from
neutral. Soucie et al., *Haemophilia* 17(3), 2011 (reference values for range
of motion in healthy people) report the same order. The literature tops out at
about 90 deg either side of neutral, which is the :data:`ANATOMICAL_BOUNDS`
ceiling that :func:`validate_coordinate_ranges` enforces.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

Range = tuple[float, float]

#: Radioulnar angle of the neutral (thumb-forward) forearm, in radians.
FOREARM_NEUTRAL_RAD = math.pi / 2.0
#: Forearm pronation from neutral, in degrees (AAOS normative, see module doc).
FOREARM_PRONATION_MAX_DEG = 80.0
#: Literature ceiling of forearm rotation from neutral on either side, degrees.
FOREARM_ROM_CEILING_DEG = 90.0

#: Physiological bounds a golf range may not leave, by coordinate. Only the
#: coordinates with a published normative range are declared; ranges widened
#: for the swing beyond physiology (wrist deviation) are not listed.
ANATOMICAL_BOUNDS: dict[str, Range] = {
    side: (
        0.0,
        FOREARM_NEUTRAL_RAD + math.radians(FOREARM_ROM_CEILING_DEG),
    )
    for side in ("pro_sup_r", "pro_sup_l")
}

_FOREARM_RANGE: Range = (
    0.0,
    FOREARM_NEUTRAL_RAD + math.radians(FOREARM_PRONATION_MAX_DEG),
)

#: Coordinate ranges (rad) the golf models override on the Rajagopal base.
GOLF_COORDINATE_RANGES: dict[str, Range] = {
    "arm_flex_r": (math.radians(-120.0), math.radians(180.0)),
    "arm_flex_l": (math.radians(-120.0), math.radians(180.0)),
    "lumbar_rotation": (math.radians(-120.0), math.radians(120.0)),
    "wrist_dev_r": (math.radians(-45.0), math.radians(45.0)),
    "wrist_dev_l": (math.radians(-45.0), math.radians(45.0)),
    "pro_sup_r": _FOREARM_RANGE,
    "pro_sup_l": _FOREARM_RANGE,
}


def validate_coordinate_ranges(
    ranges: Mapping[str, Range],
    bounds: Mapping[str, Range] = ANATOMICAL_BOUNDS,
) -> None:
    """Check joint ranges; raise ``ValueError`` on the first violation.

    Preconditions: every range is a finite ``(lower, upper)`` pair with
    ``lower < upper``, and where ``bounds`` declares an anatomical range for
    the coordinate the range lies inside it. Postcondition: ``ranges`` is
    unchanged.
    """
    for name, pair in ranges.items():
        if len(pair) != 2 or not all(math.isfinite(v) for v in pair):
            raise ValueError(f"{name}: range must be a finite (lower, upper), {pair!r}")
        lower, upper = pair
        if not lower < upper:
            raise ValueError(f"{name}: lower {lower} must be below upper {upper}")
        if name in bounds:
            low, high = bounds[name]
            if lower < low or upper > high:
                raise ValueError(
                    f"{name}: range ({lower}, {upper}) leaves the anatomical "
                    f"bounds ({low}, {high})"
                )


validate_coordinate_ranges(GOLF_COORDINATE_RANGES)

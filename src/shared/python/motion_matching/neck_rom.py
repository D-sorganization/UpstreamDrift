"""Cited neck range of motion for the three ``NeckInput`` coordinates (#14553).

One source of truth: :mod:`range_of_motion` re-exports these bounds, the MJCF
export turns them into joint limits, and a test pins the spec JSON to them.

The neck is modelled as a universal joint (``Rx`` lateral bend, ``Ry``
flexion/extension) followed by an axial rotation (``Rz``), the order of the
spec's primitive chain.  The bounds are symmetric, conservative *envelopes* for
a healthy adult: they sit inside the spread of the published active-ROM
surveys below (about 45 deg lateral, 50-70 deg flexion/extension and 70-90
deg axial rotation each way).  Published means differ between flexion and
extension; a symmetric envelope trades that asymmetry for a limit no healthy
golfer should reach, which is what a fit constraint needs.  The numbers are an
assumption to be refined per subject, not measured data, and none is
``verified_from_primary`` yet (the citations were not checked against the
source documents).

Matched tour driver swing (``ground_support/anthro_driver_drake``): the fitted
``NeckInputZ`` reaches -80.1 deg, i.e. the axial limit binds at the top of the
backswing; ``NeckInputX`` stays within -14..34 deg and ``NeckInputY`` within
-26..0 deg (``evidence/neck_rom/driver_swing_neck_excursion.json``).  Axial
rotation is therefore the neck's likely binding joint for the 125 mph work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.shared.python.contracts import require

_KAPANDJI = (
    "Kapandji, I.A. (2008). The Physiology of the Joints, Vol. 3: The Spinal "
    "Column, Pelvic Girdle and Head, 6th ed. (edition to be confirmed), "
    "cervical spine chapter."
)
_FERRARIO = (
    "Ferrario, V.F. et al. (2002). Active range of motion of the head and "
    "cervical spine: a three-dimensional investigation in healthy young "
    "adults. J Orthop Res 20(1):122-129."
)
_AAOS = (
    "Greene, W.B. & Heckman, J.D. (eds) (1994). The Clinical Measurement of "
    "Joint Motion. American Academy of Orthopaedic Surgeons."
)


@dataclass(frozen=True)
class NeckRom:
    """Symmetric range of one neck coordinate, with provenance."""

    coordinate: str
    motion: str
    lower_deg: float
    upper_deg: float
    sources: tuple[str, ...]
    # Same convention as ``spine_coupling_laws.json``: the numbers were not
    # read from the primary sources, so they stay an assumption until a
    # reviewer with the PDFs flips this and records page/table.
    verified_from_primary: bool = False


NECK_ROM: dict[str, NeckRom] = {
    "NeckInputX": NeckRom(
        "NeckInputX", "lateral bending", -45.0, 45.0, (_KAPANDJI, _FERRARIO, _AAOS)
    ),
    "NeckInputY": NeckRom(
        "NeckInputY",
        "flexion/extension",
        -60.0,
        60.0,
        (_KAPANDJI, _FERRARIO, _AAOS),
    ),
    "NeckInputZ": NeckRom(
        "NeckInputZ", "axial rotation", -80.0, 80.0, (_KAPANDJI, _FERRARIO, _AAOS)
    ),
}


def neck_range_rad(coordinate: str) -> tuple[float, float]:
    """Lower and upper bound of a neck coordinate in radians."""
    require(coordinate in NECK_ROM, f"not a neck coordinate: {coordinate}")
    rom = NECK_ROM[coordinate]
    return math.radians(rom.lower_deg), math.radians(rom.upper_deg)

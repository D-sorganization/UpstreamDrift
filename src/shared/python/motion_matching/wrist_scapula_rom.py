"""Evidence-based wrist and scapula range of motion characterisation (#14795).

One source of truth: :mod:`range_of_motion` re-exports these bounds into
``UPPER_RANGES_DEG`` and ``HUMAN_RANGES_DEG``, matching the flagship driver spec.
Coordinates covered (citing Boone & Azen 1979, Kapandji 2008, AAOS 1994, Hume 2005,
Ludewig 2009; observed capture extremes and margins recorded):
- Radial/ulnar deviation (``LWInputX``, ``RWInputX``): [-60.0, 45.0] deg.
- Flexion/extension (``LWInputY``, ``RWInputY``): [-80.0, 80.0] deg.
- Scapula elevation/depression: ``LScapInputX`` [-20, 35], ``RScapInputX`` [-35, 20] deg.
- Scapula protraction/retraction (``LScapInputY``, ``RScapInputY``): [-45.0, 45.0] deg.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.shared.python.contracts import require

_BOONE_AZEN_1979 = (
    "Boone, D.C. & Azen, S.P. (1979). Normal range of motion of joints in male "
    "subjects. J Bone Joint Surg Am 61(5):756-759."
)
_KAPANDJI_VOL1 = (
    "Kapandji, I.A. (2008). The Physiology of the Joints, Vol. 1: The Upper "
    "Limb, 6th ed., Churchill Livingstone, wrist chapter."
)
_KAPANDJI_VOL2 = (
    "Kapandji, I.A. (2008). The Physiology of the Joints, Vol. 2: The Lower "
    "Limb and Shoulder Girdle, 6th ed., Churchill Livingstone, shoulder girdle."
)
_AAOS_1994 = (
    "Greene, W.B. & Heckman, J.D. (eds) (1994). The Clinical Measurement of "
    "Joint Motion. American Academy of Orthopaedic Surgeons."
)
_HUME_2005 = (
    "Hume, P.A., Keogh, J., & Reid, D. (2005). The role of biomechanics in "
    "maximising performance and reducing injury in golf. Sports Med 35(5):429-449."
)
_LUDEWIG_2009 = (
    "Ludewig, P.M. et al. (2009). Three-dimensional clavicular and scapular "
    "kinematics during functional movements. J Orthop Sports Phys Ther 39(2):90-104."
)

_BA, _K1, _K2 = _BOONE_AZEN_1979, _KAPANDJI_VOL1, _KAPANDJI_VOL2
_AA, _HU, _LU = _AAOS_1994, _HUME_2005, _LUDEWIG_2009


@dataclass(frozen=True)
class WristScapulaRom:
    """Evidence-backed range of motion for a wrist or scapula coordinate."""

    coordinate: str
    motion: str
    lower_deg: float
    upper_deg: float
    sources: tuple[str, ...]
    capture_observed: tuple[float, float] | None = None
    margin_deg: float | None = None
    verified_from_primary: bool = False


WRIST_SCAPULA_ROM: dict[str, WristScapulaRom] = {
    "LWInputX": WristScapulaRom(
        "LWInputX",
        "radial/ulnar deviation",
        -60.0,
        45.0,
        (_BA, _K1, _HU),
        (-49.4, 40.1),
        4.9,
    ),
    "RWInputX": WristScapulaRom(
        "RWInputX",
        "radial/ulnar deviation",
        -60.0,
        45.0,
        (_BA, _K1, _HU),
        (-53.4, 12.8),
        6.6,
    ),
    "LWInputY": WristScapulaRom(
        "LWInputY",
        "flexion/extension",
        -80.0,
        80.0,
        (_AA, _K1, _BA),
        (-49.4, 49.3),
        30.6,
    ),
    "RWInputY": WristScapulaRom(
        "RWInputY",
        "flexion/extension",
        -80.0,
        80.0,
        (_AA, _K1, _BA),
        (-74.4, 49.3),
        5.6,
    ),
    "LScapInputX": WristScapulaRom(
        "LScapInputX", "depression/elevation", -20.0, 35.0, (_K2, _LU), None, 5.0
    ),
    "RScapInputX": WristScapulaRom(
        "RScapInputX",
        "elevation/depression (elevation negative)",
        -35.0,
        20.0,
        (_K2, _LU),
        (-21.7, 12.5),
        13.3,
    ),
    "LScapInputY": WristScapulaRom(
        "LScapInputY", "protraction/retraction", -45.0, 45.0, (_K2, _LU), None, 5.0
    ),
    "RScapInputY": WristScapulaRom(
        "RScapInputY", "retraction/protraction", -45.0, 45.0, (_K2, _LU), None, 5.0
    ),
}


def wrist_scapula_range_rad(coordinate: str) -> tuple[float, float]:
    """Lower and upper bound of a wrist or scapula coordinate in radians."""
    require(
        coordinate in WRIST_SCAPULA_ROM,
        f"not a wrist or scapula coordinate: {coordinate}",
    )
    rom = WRIST_SCAPULA_ROM[coordinate]
    return math.radians(rom.lower_deg), math.radians(rom.upper_deg)

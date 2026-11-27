"""Evidence-based wrist and scapula range of motion characterisation (#14795).

One source of truth: re-exported into UPPER_RANGES_DEG and HUMAN_RANGES_DEG
from cited literature: Boone & Azen 1979, Kapandji 2008, AAOS 1994, Hume 2005,
Ludewig 2009.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.shared.python.contracts import require

_BA = "Boone & Azen (1979) J Bone Joint Surg Am 61(5):756-759"
_K1 = "Kapandji (2008) Physiology of the Joints, Vol. 1: The Upper Limb"
_K2 = "Kapandji (2008) Physiology of the Joints, Vol. 2: Lower Limb & Shoulder"
_AA = "Greene & Heckman (eds) (1994) Clinical Measurement of Joint Motion AAOS"
_HU = "Hume et al. (2005) Sports Med 35(5):429-449"
_LU = "Ludewig et al. (2009) J Orthop Sports Phys Ther 39(2):90-104"


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


_D, _FE, _ED = "radial/ulnar deviation", "flexion/extension", "elevation/depression"
_RAW = (
    ("LWInputX", _D, -60.0, 45.0, (_BA, _K1, _HU), (-49.4, 40.1), 4.9),
    ("RWInputX", _D, -60.0, 45.0, (_BA, _K1, _HU), (-53.4, 12.8), 6.6),
    ("LWInputY", _FE, -80.0, 80.0, (_AA, _K1, _BA), (-49.4, 49.3), 30.6),
    ("RWInputY", _FE, -80.0, 80.0, (_AA, _K1, _BA), (-74.4, 49.3), 5.6),
    ("LScapInputX", "depression/elevation", -20.0, 35.0, (_K2, _LU), None, 5.0),
    ("RScapInputX", _ED, -35.0, 20.0, (_K2, _LU), (-21.7, 12.5), 13.3),
    ("LScapInputY", "protraction/retraction", -45.0, 45.0, (_K2, _LU), None, 5.0),
    ("RScapInputY", "retraction/protraction", -45.0, 45.0, (_K2, _LU), None, 5.0),
)

WRIST_SCAPULA_ROM: dict[str, WristScapulaRom] = {
    r[0]: WristScapulaRom(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in _RAW
}


def wrist_scapula_range_rad(coordinate: str) -> tuple[float, float]:
    """Lower and upper bound of a wrist or scapula coordinate in radians."""
    require(
        coordinate in WRIST_SCAPULA_ROM,
        f"not a wrist or scapula coordinate: {coordinate}",
    )
    rom = WRIST_SCAPULA_ROM[coordinate]
    return math.radians(rom.lower_deg), math.radians(rom.upper_deg)

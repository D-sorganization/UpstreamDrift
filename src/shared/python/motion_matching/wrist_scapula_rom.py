"""Evidence-based wrist and scapula range of motion characterisation (#14795).

One source of truth: :mod:`range_of_motion` re-exports these bounds into
``UPPER_RANGES_DEG`` and ``HUMAN_RANGES_DEG``, and a test asserts that the
flagship driver spec matches them.

Coordinates covered:
- Wrist radial/ulnar deviation (``LWInputX``, ``RWInputX``): +X is radial deviation
  (cock toward elbow pit); -X is ulnar deviation. Bounds: [-60.0, 45.0] deg.
  Captures observed extremes (HRG Section 4): LWInputX [-49.4, 40.1] deg,
  RWInputX [-53.4, 12.8] deg. Margins: 6.6 deg beyond ulnar extreme, 4.9 deg
  beyond radial extreme. Supported by active ROM surveys (Boone & Azen 1979;
  Kapandji 2008) and dynamic golf swing kinematics (Hume et al. 2005).
- Wrist flexion/extension (``LWInputY``, ``RWInputY``): symmetric envelope
  [-80.0, 80.0] deg. Captures observed extreme: RWInputY reaches -74.4 deg
  extension. Margin: 5.6 deg beyond capture extreme. Supported by AAOS
  clinical standards (Greene & Heckman 1994), Kapandji (2008), and Boone &
  Azen (1979).
- Scapula elevation/depression (``LScapInputX``, ``RScapInputX``): native model
  sign conventions have LScapInputX positive for elevation and negative for
  depression ([-20.0, 35.0] deg), whereas RScapInputX is positive for
  depression and negative for elevation ([-35.0, 20.0] deg). Captures observed
  extreme: RScapInputX reaches -21.7 deg elevation. Margin: 13.3 deg beyond
  capture extreme. Supported by Kapandji (2008) Vol. 2 and Ludewig et al. (2009).
- Scapula protraction/retraction (``LScapInputY``, ``RScapInputY``): chest-wall
  excursion envelope [-45.0, 45.0] deg. Supported by Kapandji (2008) Vol. 2
  and Ludewig et al. (2009).
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
        (_BOONE_AZEN_1979, _KAPANDJI_VOL1, _HUME_2005),
        capture_observed=(-49.4, 40.1),
        margin_deg=4.9,
    ),
    "RWInputX": WristScapulaRom(
        "RWInputX",
        "radial/ulnar deviation",
        -60.0,
        45.0,
        (_BOONE_AZEN_1979, _KAPANDJI_VOL1, _HUME_2005),
        capture_observed=(-53.4, 12.8),
        margin_deg=6.6,
    ),
    "LWInputY": WristScapulaRom(
        "LWInputY",
        "flexion/extension",
        -80.0,
        80.0,
        (_AAOS_1994, _KAPANDJI_VOL1, _BOONE_AZEN_1979),
        capture_observed=(-49.4, 49.3),
        margin_deg=30.6,
    ),
    "RWInputY": WristScapulaRom(
        "RWInputY",
        "flexion/extension",
        -80.0,
        80.0,
        (_AAOS_1994, _KAPANDJI_VOL1, _BOONE_AZEN_1979),
        capture_observed=(-74.4, 49.3),
        margin_deg=5.6,
    ),
    "LScapInputX": WristScapulaRom(
        "LScapInputX",
        "depression/elevation",
        -20.0,
        35.0,
        (_KAPANDJI_VOL2, _LUDEWIG_2009),
        capture_observed=None,
        margin_deg=5.0,
    ),
    "RScapInputX": WristScapulaRom(
        "RScapInputX",
        "elevation/depression (elevation negative)",
        -35.0,
        20.0,
        (_KAPANDJI_VOL2, _LUDEWIG_2009),
        capture_observed=(-21.7, 12.5),
        margin_deg=13.3,
    ),
    "LScapInputY": WristScapulaRom(
        "LScapInputY",
        "protraction/retraction",
        -45.0,
        45.0,
        (_KAPANDJI_VOL2, _LUDEWIG_2009),
        capture_observed=None,
        margin_deg=5.0,
    ),
    "RScapInputY": WristScapulaRom(
        "RScapInputY",
        "retraction/protraction",
        -45.0,
        45.0,
        (_KAPANDJI_VOL2, _LUDEWIG_2009),
        capture_observed=None,
        margin_deg=5.0,
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

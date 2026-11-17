"""Settings for the implicit SDF hand. Numbers live in ``sdf_hand.json``."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SETTINGS_PATH = _REPO_ROOT / "scripts" / "config" / "sdf_hand.json"
_FINGERS = ("index", "middle", "ring", "little")
_BLENDS = ("joint", "palm", "mcp", "web", "nail", "crease")
_RADIUS_RANGE_M = (0.006, 0.012)


@dataclass(frozen=True)
class FingerProfile:
    """One digit's bone lengths and cone radii (metres) at the reference hand."""

    phalanges_m: tuple[float, float, float]
    metacarpal_m: float
    metacarpal_radii_m: tuple[float, float]
    radii_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class ThumbProfile:
    """Thumb lengths, radii, and shaft stations at the reference hand."""

    metacarpal_m: float
    metacarpal_radii_m: tuple[float, float]
    phalanges_m: tuple[float, float]
    radii_m: tuple[tuple[float, float], tuple[float, float]]
    stations_m: tuple[float, float, float]


@dataclass(frozen=True)
class SdfHandSettings:
    """Reference-hand sculpt settings. Lengths scale with ``GripSpec.hand_size``.

    Invariants (checked at load): blend radii are positive, the middle-finger
    chain equals ``hand_length_m``, finger radii taper, and Taubin
    ``lambda < -mu``.
    """

    hand_length_m: float
    palm_width_m: float
    default_grip_radius_m: float
    pad_clearance_m: float
    contact_tolerance_m: float
    penetration_tolerance_m: float
    mcp_from_v_deg: float
    full_contact_palm_finger_depth: float
    open_hand_lift_m: float
    mc_tangent_weight: float
    mc_radial_weight: float
    wrist_convergence: float
    transverse_arch_m: float
    longitudinal_arch_m: float
    palm_wrap_arc_rad: float
    palm_base_hover_m: float
    palm_axial_drop_m: float
    thumb_opposition_sweep_m: float
    blend_m: Mapping[str, float]
    resolution: int
    test_resolution: int
    min_resolution: int
    bounds_pad_m: float
    taubin_iterations: int
    taubin_lambda: float
    taubin_mu: float
    frame_px: int
    frame_fill: float
    test_frame_px: int
    skin_material: str
    grip_rgb: tuple[float, float, float]
    shaft_rgb: tuple[float, float, float]
    background_rgb: tuple[float, float, float]
    key_light: tuple[float, float, float]
    fill_light: tuple[float, float, float]
    fill_weight: float
    ambient: float
    shaft_radius_m: float
    shaft_extension_fraction: float
    knuckle_radius_fraction: float
    knuckle_dorsal_fraction: float
    crease_radius_fraction: float
    crease_depth_fraction: float
    crease_span_fraction: float
    nail_radii_m: tuple[float, float, float]
    nail_along_fraction: float
    thenar_radii_m: tuple[float, float, float]
    hypothenar_radii_m: tuple[float, float, float]
    web_radii_m: tuple[float, float, float]
    wrist_radii_m: tuple[float, float, float]
    palm_fill_radii_m: tuple[float, float, float]
    ellipsoid_support_margin_m: float
    legacy_club_cross_angle_deg: float
    cross_angle_mcp_tilt_gain: float
    overlap_little_lift_m: float
    palm_head_weight: float
    butt_extension_m: float
    finger_spread_deg: Mapping[str, float]
    transverse_arch_coeff: Mapping[str, float]
    fingers: Mapping[str, FingerProfile]
    thumb: ThumbProfile
    source_path: str

    def finger(self, name: str) -> FingerProfile:
        """Return one digit profile. Unknown names raise ``KeyError``."""
        try:
            return self.fingers[name]
        except KeyError as exc:
            raise KeyError(f"unknown finger {name!r}") from exc


def repo_root() -> Path:
    """Repository root inferred from this module's location."""
    return _REPO_ROOT


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive(name: str, value: object) -> float:
    number = _finite(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


def _vec3(name: str, value: object) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a length-3 list, got {value!r}")
    return (_finite(name, value[0]), _finite(name, value[1]), _finite(name, value[2]))


def _pair(name: str, value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a length-2 list, got {value!r}")
    return (_finite(name, value[0]), _finite(name, value[1]))


def _radius_band(name: str, value: object) -> tuple[float, float]:
    start, end = _pair(name, value)
    lo, hi = _RADIUS_RANGE_M
    if not (lo <= start <= hi and lo <= end <= hi):
        raise ValueError(f"{name} must lie in [{lo}, {hi}] m, got {(start, end)}")
    return (start, end)


def _radius_pair(name: str, value: object) -> tuple[float, float]:
    start, end = _radius_band(name, value)
    if end > start:
        raise ValueError(f"{name} must taper (start >= end), got {(start, end)}")
    return (start, end)


def _finger_map(name: str, raw: object) -> dict[str, float]:
    if not isinstance(raw, dict) or set(raw) != set(_FINGERS):
        raise ValueError(f"{name} must map exactly {_FINGERS}")
    return {finger: _finite(name, raw[finger]) for finger in _FINGERS}


def _load_finger(name: str, raw: object) -> FingerProfile:
    if not isinstance(raw, dict):
        raise TypeError(f"finger {name} must be an object")
    phalanges = raw.get("phalanges_m")
    radii = raw.get("radii_m")
    if not isinstance(phalanges, list) or len(phalanges) != 3:
        raise ValueError(f"{name} phalanges_m must have length 3")
    if not isinstance(radii, list) or len(radii) != 3:
        raise ValueError(f"{name} radii_m must have length 3")
    lengths = tuple(_positive(f"{name} phalanx", item) for item in phalanges)
    cones = tuple(_radius_pair(f"{name} radii", item) for item in radii)
    proximal, middle, distal = lengths
    if not (0.0 < proximal / 2.0 < middle < proximal / 2.0 + distal):
        raise ValueError(f"{name} phalanx lengths cannot wrap the grip: {lengths}")
    return FingerProfile(
        phalanges_m=(lengths[0], lengths[1], lengths[2]),
        metacarpal_m=_positive(f"{name} metacarpal", raw.get("metacarpal_m")),
        metacarpal_radii_m=_radius_band(
            f"{name} metacarpal radii", raw.get("metacarpal_radii_m")
        ),
        radii_m=(cones[0], cones[1], cones[2]),
    )


def _load_thumb(raw: object) -> ThumbProfile:
    if not isinstance(raw, dict):
        raise TypeError("thumb must be an object")
    phalanges = raw.get("phalanges_m")
    radii = raw.get("radii_m")
    stations = raw.get("stations_m")
    if not isinstance(phalanges, list) or len(phalanges) != 2:
        raise ValueError("thumb phalanges_m must have length 2")
    if not isinstance(radii, list) or len(radii) != 2:
        raise ValueError("thumb radii_m must have length 2")
    if not isinstance(stations, list) or len(stations) != 3:
        raise ValueError("thumb stations_m must have length 3")
    lengths = tuple(_positive("thumb phalanx", item) for item in phalanges)
    cones = tuple(_radius_pair("thumb radii", item) for item in radii)
    stops = tuple(_finite("thumb station", item) for item in stations)
    if abs((stops[1] - stops[0]) - lengths[0]) > 1e-6:
        raise ValueError("thumb stations must match the proximal phalanx length")
    if abs((stops[2] - stops[1]) - lengths[1]) > 1e-6:
        raise ValueError("thumb stations must match the distal phalanx length")
    return ThumbProfile(
        metacarpal_m=_positive("thumb metacarpal", raw.get("metacarpal_m")),
        metacarpal_radii_m=_radius_pair(
            "thumb metacarpal radii", raw.get("metacarpal_radii_m")
        ),
        phalanges_m=(lengths[0], lengths[1]),
        radii_m=(cones[0], cones[1]),
        stations_m=(stops[0], stops[1], stops[2]),
    )


_FIELD_VALIDATORS: tuple[tuple[str, Callable[[str, object], Any]], ...] = (
    ("palm_width_m", _positive),
    ("default_grip_radius_m", _positive),
    ("pad_clearance_m", _finite),
    ("contact_tolerance_m", _positive),
    ("penetration_tolerance_m", _positive),
    ("mcp_from_v_deg", _finite),
    ("full_contact_palm_finger_depth", _positive),
    ("open_hand_lift_m", _finite),
    ("mc_tangent_weight", _positive),
    ("mc_radial_weight", _positive),
    ("transverse_arch_m", _finite),
    ("longitudinal_arch_m", _finite),
    ("palm_wrap_arc_rad", _positive),
    ("palm_base_hover_m", _positive),
    ("palm_axial_drop_m", _positive),
    ("thumb_opposition_sweep_m", _finite),
    ("bounds_pad_m", _positive),
    ("grip_rgb", _vec3),
    ("shaft_rgb", _vec3),
    ("background_rgb", _vec3),
    ("key_light", _vec3),
    ("fill_light", _vec3),
    ("fill_weight", _finite),
    ("ambient", _finite),
    ("shaft_radius_m", _positive),
    ("shaft_extension_fraction", _finite),
    ("knuckle_radius_fraction", _positive),
    ("knuckle_dorsal_fraction", _finite),
    ("crease_radius_fraction", _positive),
    ("crease_depth_fraction", _finite),
    ("crease_span_fraction", _positive),
    ("nail_radii_m", _vec3),
    ("nail_along_fraction", _finite),
    ("thenar_radii_m", _vec3),
    ("hypothenar_radii_m", _vec3),
    ("web_radii_m", _vec3),
    ("wrist_radii_m", _vec3),
    ("palm_fill_radii_m", _vec3),
    ("ellipsoid_support_margin_m", _positive),
    ("legacy_club_cross_angle_deg", _finite),
    ("cross_angle_mcp_tilt_gain", _finite),
    ("overlap_little_lift_m", _finite),
    ("palm_head_weight", _finite),
    ("butt_extension_m", _positive),
    ("finger_spread_deg", _finger_map),
    ("transverse_arch_coeff", _finger_map),
)
"""Scalar fields validated in declaration order after the cross-field checks."""


def _cross_checked_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Fields whose validation spans keys or bounds beyond a single number."""
    blends_raw = raw.get("blend_m")
    if not isinstance(blends_raw, dict) or set(blends_raw) != set(_BLENDS):
        raise ValueError(f"blend_m must map exactly {_BLENDS}")
    blends = {key: _positive(f"blend_m.{key}", blends_raw[key]) for key in _BLENDS}
    fingers_raw = raw.get("fingers")
    if not isinstance(fingers_raw, dict) or set(fingers_raw) != set(_FINGERS):
        raise ValueError(f"fingers must map exactly {_FINGERS}")
    fingers = {name: _load_finger(name, fingers_raw[name]) for name in _FINGERS}
    middle = fingers["middle"]
    chain = middle.metacarpal_m + sum(middle.phalanges_m)
    hand_length = _positive("hand_length_m", raw.get("hand_length_m"))
    if abs(chain - hand_length) > 1e-4:
        raise ValueError(
            f"hand_length_m ({hand_length}) must equal the middle-finger chain ({chain})"
        )
    lam = _positive("taubin_lambda", raw.get("taubin_lambda"))
    mu = _finite("taubin_mu", raw.get("taubin_mu"))
    if not (mu < 0.0 and lam < -mu):
        raise ValueError(f"Taubin needs 0 < lambda < -mu, got lambda={lam}, mu={mu}")
    iterations = int(raw.get("taubin_iterations", 0))
    if iterations < 1:
        raise ValueError(f"taubin_iterations must be >= 1, got {iterations}")
    convergence = _finite("wrist_convergence", raw.get("wrist_convergence"))
    if not 0.0 < convergence <= 1.0:
        raise ValueError(f"wrist_convergence must be in (0, 1], got {convergence}")
    if _finite("pad_clearance_m", raw.get("pad_clearance_m")) < 0.0:
        raise ValueError("pad_clearance_m must be >= 0")
    test_frame = int(raw.get("test_frame_px", 0))
    if test_frame < 8:
        raise ValueError(f"test_frame_px must be >= 8, got {test_frame}")
    skin = str(raw.get("skin_material", ""))
    if not skin:
        raise ValueError("skin_material must be a non-empty id")
    resolution = int(raw.get("resolution", 0))
    minimum = int(raw.get("min_resolution", 0))
    test_resolution = int(raw.get("test_resolution", 0))
    if minimum < 8 or test_resolution < minimum or resolution < test_resolution:
        raise ValueError("resolution must be >= test_resolution >= min_resolution >= 8")
    frame_px = int(raw.get("frame_px", 0))
    if frame_px < 1024:
        raise ValueError(f"frame_px must be at least 1024, got {frame_px}")
    fill = _finite("frame_fill", raw.get("frame_fill"))
    if not 0.5 <= fill <= 0.9:
        raise ValueError(f"frame_fill must be in [0.5, 0.9], got {fill}")
    return {
        "blend_m": blends,
        "fingers": fingers,
        "hand_length_m": hand_length,
        "taubin_lambda": lam,
        "taubin_mu": mu,
        "taubin_iterations": iterations,
        "wrist_convergence": convergence,
        "test_frame_px": test_frame,
        "skin_material": skin,
        "resolution": resolution,
        "min_resolution": minimum,
        "test_resolution": test_resolution,
        "frame_px": frame_px,
        "frame_fill": fill,
    }


def load_sdf_hand_settings(path: Path | None = None) -> SdfHandSettings:
    """Load and validate the SDF hand settings file.

    Preconditions: the path exists and the JSON matches the schema.
    Postcondition: the returned settings satisfy the module invariants.
    """
    source = DEFAULT_SETTINGS_PATH if path is None else Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SDF hand settings not found: {source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SDF hand settings must be a JSON object")
    checked = _cross_checked_fields(raw)
    scalars = {name: check(name, raw.get(name)) for name, check in _FIELD_VALIDATORS}
    return SdfHandSettings(
        **checked,
        **scalars,
        thumb=_load_thumb(raw.get("thumb")),
        source_path=str(source),
    )


__all__ = [
    "DEFAULT_SETTINGS_PATH",
    "FingerProfile",
    "SdfHandSettings",
    "ThumbProfile",
    "load_sdf_hand_settings",
    "repo_root",
]

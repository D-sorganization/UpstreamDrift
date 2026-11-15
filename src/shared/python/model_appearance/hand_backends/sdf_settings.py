"""Settings for the implicit SDF hand. Numbers live in ``sdf_hand.json``."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

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
    return SdfHandSettings(
        hand_length_m=hand_length,
        palm_width_m=_positive("palm_width_m", raw.get("palm_width_m")),
        default_grip_radius_m=_positive(
            "default_grip_radius_m", raw.get("default_grip_radius_m")
        ),
        pad_clearance_m=_finite("pad_clearance_m", raw.get("pad_clearance_m")),
        contact_tolerance_m=_positive(
            "contact_tolerance_m", raw.get("contact_tolerance_m")
        ),
        penetration_tolerance_m=_positive(
            "penetration_tolerance_m", raw.get("penetration_tolerance_m")
        ),
        mcp_from_v_deg=_finite("mcp_from_v_deg", raw.get("mcp_from_v_deg")),
        full_contact_palm_finger_depth=_positive(
            "full_contact_palm_finger_depth", raw.get("full_contact_palm_finger_depth")
        ),
        open_hand_lift_m=_finite("open_hand_lift_m", raw.get("open_hand_lift_m")),
        mc_tangent_weight=_positive("mc_tangent_weight", raw.get("mc_tangent_weight")),
        mc_radial_weight=_positive("mc_radial_weight", raw.get("mc_radial_weight")),
        wrist_convergence=convergence,
        transverse_arch_m=_finite("transverse_arch_m", raw.get("transverse_arch_m")),
        longitudinal_arch_m=_finite(
            "longitudinal_arch_m", raw.get("longitudinal_arch_m")
        ),
        palm_wrap_arc_rad=_positive("palm_wrap_arc_rad", raw.get("palm_wrap_arc_rad")),
        palm_base_hover_m=_positive("palm_base_hover_m", raw.get("palm_base_hover_m")),
        palm_axial_drop_m=_positive("palm_axial_drop_m", raw.get("palm_axial_drop_m")),
        thumb_opposition_sweep_m=_finite(
            "thumb_opposition_sweep_m", raw.get("thumb_opposition_sweep_m")
        ),
        blend_m=blends,
        resolution=resolution,
        test_resolution=test_resolution,
        min_resolution=minimum,
        bounds_pad_m=_positive("bounds_pad_m", raw.get("bounds_pad_m")),
        taubin_iterations=iterations,
        taubin_lambda=lam,
        taubin_mu=mu,
        frame_px=frame_px,
        frame_fill=fill,
        test_frame_px=test_frame,
        skin_material=skin,
        grip_rgb=_vec3("grip_rgb", raw.get("grip_rgb")),
        shaft_rgb=_vec3("shaft_rgb", raw.get("shaft_rgb")),
        background_rgb=_vec3("background_rgb", raw.get("background_rgb")),
        key_light=_vec3("key_light", raw.get("key_light")),
        fill_light=_vec3("fill_light", raw.get("fill_light")),
        fill_weight=_finite("fill_weight", raw.get("fill_weight")),
        ambient=_finite("ambient", raw.get("ambient")),
        shaft_radius_m=_positive("shaft_radius_m", raw.get("shaft_radius_m")),
        shaft_extension_fraction=_finite(
            "shaft_extension_fraction", raw.get("shaft_extension_fraction")
        ),
        knuckle_radius_fraction=_positive(
            "knuckle_radius_fraction", raw.get("knuckle_radius_fraction")
        ),
        knuckle_dorsal_fraction=_finite(
            "knuckle_dorsal_fraction", raw.get("knuckle_dorsal_fraction")
        ),
        crease_radius_fraction=_positive(
            "crease_radius_fraction", raw.get("crease_radius_fraction")
        ),
        crease_depth_fraction=_finite(
            "crease_depth_fraction", raw.get("crease_depth_fraction")
        ),
        crease_span_fraction=_positive(
            "crease_span_fraction", raw.get("crease_span_fraction")
        ),
        nail_radii_m=_vec3("nail_radii_m", raw.get("nail_radii_m")),
        nail_along_fraction=_finite(
            "nail_along_fraction", raw.get("nail_along_fraction")
        ),
        thenar_radii_m=_vec3("thenar_radii_m", raw.get("thenar_radii_m")),
        hypothenar_radii_m=_vec3("hypothenar_radii_m", raw.get("hypothenar_radii_m")),
        web_radii_m=_vec3("web_radii_m", raw.get("web_radii_m")),
        wrist_radii_m=_vec3("wrist_radii_m", raw.get("wrist_radii_m")),
        palm_fill_radii_m=_vec3("palm_fill_radii_m", raw.get("palm_fill_radii_m")),
        ellipsoid_support_margin_m=_positive(
            "ellipsoid_support_margin_m", raw.get("ellipsoid_support_margin_m")
        ),
        legacy_club_cross_angle_deg=_finite(
            "legacy_club_cross_angle_deg", raw.get("legacy_club_cross_angle_deg")
        ),
        cross_angle_mcp_tilt_gain=_finite(
            "cross_angle_mcp_tilt_gain", raw.get("cross_angle_mcp_tilt_gain")
        ),
        overlap_little_lift_m=_finite(
            "overlap_little_lift_m", raw.get("overlap_little_lift_m")
        ),
        palm_head_weight=_finite("palm_head_weight", raw.get("palm_head_weight")),
        butt_extension_m=_positive("butt_extension_m", raw.get("butt_extension_m")),
        finger_spread_deg=_finger_map(
            "finger_spread_deg", raw.get("finger_spread_deg")
        ),
        transverse_arch_coeff=_finger_map(
            "transverse_arch_coeff", raw.get("transverse_arch_coeff")
        ),
        fingers=fingers,
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

"""Settings for the shared hand poser.

Numbers that decide contact live in ``scripts/config/hand_pose_contact.json``.
Finger radii and phalanx lengths stay on ``HandMeshParams`` so the poser and
the grip mesh cannot drift apart.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SETTINGS_PATH = _REPO_ROOT / "scripts" / "config" / "hand_pose_contact.json"
_FINGERS = ("index", "middle", "ring", "little")
_LIMITS = {
    "proximal": "mcp_limit_rad",
    "middle": "pip_limit_rad",
    "distal": "dip_limit_rad",
}


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive(name: str, value: object) -> float:
    number = _finite(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


@dataclass(frozen=True)
class PoseSettings:
    """Contact, curl, arch and palm numbers for one pose solve.

    Preconditions: the committed JSON schema, with
    ``base_clearance_m + just_below_pinky_m`` inside the gap tolerance, the
    palm standoff outside that tolerance, and depth 1 bringing the palm patch
    back inside it. Postcondition: every length is stored in metres or
    radians, never degrees.
    """

    contact_gap_tolerance_m: float
    penetration_tolerance_m: float
    base_clearance_m: float
    just_below_pinky_m: float
    open_gap_m: float
    curl_samples: int
    mcp_limit_rad: float
    pip_limit_rad: float
    dip_limit_rad: float
    arch_gain_rad: float
    arch_travel_m: float
    arch_weight: tuple[tuple[str, float], ...]
    cross_azimuth_scale: float
    thumb_lengths_m: tuple[float, float, float]
    thumb_flex_limit_rad: float
    palm_grid_nu: int
    palm_grid_nv: int
    palm_standoff_m: float
    palm_arch_travel_m: float
    palm_cell_m: float
    min_wrap_rad: float
    style_separation_m: float
    hinge_bisection_steps: int
    axis_samples: int
    metacarpal_length_m: tuple[tuple[str, float], ...]

    def arch_weight_for(self, finger: str) -> float:
        """Arch weight for ``finger`` (index light, little full)."""
        weights = dict(self.arch_weight)
        if finger not in weights:
            raise KeyError(f"no arch weight for {finger!r}")
        return weights[finger]

    def metacarpal_length_for(self, finger: str) -> float:
        """Nominal metacarpal length (metres) for ``finger``."""
        lengths = dict(self.metacarpal_length_m)
        if finger not in lengths:
            raise KeyError(f"no metacarpal length for {finger!r}")
        return lengths[finger]

    def flexion_limit_rad(self, segment: str) -> float:
        """Flexion limit for a phalanx segment name."""
        field_name = _LIMITS.get(segment)
        if field_name is None:
            raise KeyError(f"no flexion limit for segment {segment!r}")
        return float(getattr(self, field_name))


def _weight_table(raw: object, label: str) -> tuple[tuple[str, float], ...]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label} must be an object, got {type(raw).__name__}")
    if set(raw) != set(_FINGERS):
        raise ValueError(f"{label} must cover exactly {_FINGERS}, got {sorted(raw)}")
    pairs = []
    for finger in _FINGERS:
        pairs.append((finger, _positive(f"{label}.{finger}", raw[finger])))
    return tuple(pairs)


def _arch_travel(payload: Mapping[str, Any]) -> float:
    travel = _finite("arch_travel_m", payload["arch_travel_m"])
    if travel < 0.0:
        raise ValueError("arch_travel_m must be non-negative")
    return travel


def _arch_weights(raw: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"arch_weight must be an object, got {type(raw).__name__}")
    if set(raw) != set(_FINGERS):
        raise ValueError(f"arch_weight must cover exactly {_FINGERS}")
    pairs = []
    for finger in _FINGERS:
        weight = _finite(f"arch_weight.{finger}", raw[finger])
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"arch_weight.{finger} must be in [0, 1], got {weight}")
        pairs.append((finger, weight))
    return tuple(pairs)


def settings_from_dict(payload: Mapping[str, Any]) -> PoseSettings:
    """Build settings from a JSON object.

    Preconditions: ``payload`` is a mapping with schema version 1.
    Postcondition: pinky flat clearance fits in the gap band, and a fully
    cupped palm patch falls inside that band while a flat palm does not.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(f"settings must be a mapping, got {type(payload).__name__}")
    version = payload.get("schema_version")
    if version != 1:
        raise ValueError(f"unsupported hand pose settings schema {version!r}")
    gap = _positive("contact_gap_tolerance_m", payload["contact_gap_tolerance_m"])
    pen = _positive("penetration_tolerance_m", payload["penetration_tolerance_m"])
    base = _finite("base_clearance_m", payload["base_clearance_m"])
    just_below = _finite("just_below_pinky_m", payload["just_below_pinky_m"])
    if base < 0.0 or just_below < 0.0:
        raise ValueError("clearances must be non-negative")
    if base + just_below > gap + 1e-12:
        raise ValueError(
            "base_clearance_m + just_below_pinky_m must fit in "
            f"contact_gap_tolerance_m ({gap}); got {base + just_below}"
        )
    open_gap = _positive("open_gap_m", payload["open_gap_m"])
    if open_gap <= gap:
        raise ValueError("open_gap_m must be wider than the contact tolerance")
    samples = payload["curl_samples"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError(f"curl_samples must be an int >= 2, got {samples!r}")
    steps = payload["hinge_bisection_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 8:
        raise ValueError(f"hinge_bisection_steps must be an int >= 8, got {steps!r}")
    axis_samples = payload["axis_samples"]
    if (
        isinstance(axis_samples, bool)
        or not isinstance(axis_samples, int)
        or axis_samples < 2
    ):
        raise ValueError(f"axis_samples must be an int >= 2, got {axis_samples!r}")
    standoff = _positive("palm_standoff_m", payload["palm_standoff_m"])
    travel = _finite("palm_arch_travel_m", payload["palm_arch_travel_m"])
    if travel < 0.0:
        raise ValueError("palm_arch_travel_m must be non-negative")
    if standoff <= gap:
        raise ValueError("palm_standoff_m must sit outside the gap tolerance")
    cupped = standoff - travel
    if not -pen <= cupped <= gap:
        raise ValueError(
            "palm_standoff_m - palm_arch_travel_m must land inside the contact "
            f"band at depth 1, got {cupped}"
        )
    weights = _arch_weights(payload["arch_weight"])
    if dict(weights)["little"] < dict(weights)["index"]:
        raise ValueError("little arch weight must be at least the index weight")
    scale = _finite("cross_azimuth_scale", payload["cross_azimuth_scale"])
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"cross_azimuth_scale must be in (0, 1], got {scale}")
    thumb_raw = payload["thumb_lengths_m"]
    if not isinstance(thumb_raw, (list, tuple)) or len(thumb_raw) != 3:
        raise ValueError("thumb_lengths_m must be three lengths")
    thumb = tuple(
        _positive(f"thumb_lengths_m[{i}]", v) for i, v in enumerate(thumb_raw)
    )
    nu = payload["palm_grid_nu"]
    nv = payload["palm_grid_nv"]
    if isinstance(nu, bool) or not isinstance(nu, int) or nu < 1:
        raise ValueError(f"palm_grid_nu must be an int >= 1, got {nu!r}")
    if isinstance(nv, bool) or not isinstance(nv, int) or nv < 1:
        raise ValueError(f"palm_grid_nv must be an int >= 1, got {nv!r}")
    return PoseSettings(
        contact_gap_tolerance_m=gap,
        penetration_tolerance_m=pen,
        base_clearance_m=base,
        just_below_pinky_m=just_below,
        open_gap_m=open_gap,
        curl_samples=samples,
        mcp_limit_rad=math.radians(
            _positive("mcp_limit_deg", payload["mcp_limit_deg"])
        ),
        pip_limit_rad=math.radians(
            _positive("pip_limit_deg", payload["pip_limit_deg"])
        ),
        dip_limit_rad=math.radians(
            _positive("dip_limit_deg", payload["dip_limit_deg"])
        ),
        arch_gain_rad=_positive("arch_gain_rad", payload["arch_gain_rad"]),
        arch_travel_m=_arch_travel(payload),
        arch_weight=weights,
        cross_azimuth_scale=scale,
        thumb_lengths_m=(thumb[0], thumb[1], thumb[2]),
        thumb_flex_limit_rad=math.radians(
            _positive("thumb_flex_limit_deg", payload["thumb_flex_limit_deg"])
        ),
        palm_grid_nu=nu,
        palm_grid_nv=nv,
        palm_standoff_m=standoff,
        palm_arch_travel_m=travel,
        palm_cell_m=_positive("palm_cell_m", payload["palm_cell_m"]),
        min_wrap_rad=_positive("min_wrap_rad", payload["min_wrap_rad"]),
        style_separation_m=_positive(
            "style_separation_m", payload["style_separation_m"]
        ),
        hinge_bisection_steps=steps,
        axis_samples=axis_samples,
        metacarpal_length_m=_weight_table(
            payload["metacarpal_length_m"], "metacarpal_length_m"
        ),
    )


def load_pose_settings(path: Path | None = None) -> PoseSettings:
    """Load pose settings. ``path`` defaults to the committed JSON file."""
    source = DEFAULT_SETTINGS_PATH if path is None else Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"hand pose settings not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    return settings_from_dict(payload)


__all__ = [
    "DEFAULT_SETTINGS_PATH",
    "PoseSettings",
    "load_pose_settings",
    "settings_from_dict",
]

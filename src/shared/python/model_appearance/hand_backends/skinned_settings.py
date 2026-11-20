"""Settings of the skinned MakeHuman hand (epic #15150).

Every number the hand uses comes from ``scripts/config/skinned_human_hand.json``;
this module loads and validates it. Split out of :mod:`skinned_human` to keep
that module inside the file-size budget.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.shared.python.model_appearance.hand_backends.correctives import (
    FoldRelaxSettings,
    PadConformSettings,
    PressSettings,
)
from src.shared.python.model_appearance.hand_backends.golf_grip import load_golf_grip
from src.shared.python.model_appearance.hand_backends.knuckle_sideways import (
    ABDUCTED_FINGERS,
)
from src.shared.python.model_appearance.hand_backends.metacarpal_arch import (
    MetacarpalArchSettings,
)
from src.shared.python.model_appearance.hand_backends.palm_fold import (
    PalmFoldSettings,
)
from src.shared.python.model_appearance.hand_backends.rest_shape import (
    FullnessSettings,
)
from src.shared.python.model_appearance.hand_backends.joint_angles import FINGERS
from src.shared.python.model_appearance.hand_backends.posing import CurlSettings


SETTINGS_PATH = (
    Path(__file__).resolve().parents[5]
    / "scripts"
    / "config"
    / "skinned_human_hand.json"
)
_MAX_SUBDIVISION_LEVELS = 3
JOINT_OF_SEGMENT = {"prox": "mcp", "mid": "pip", "dist": "dip"}


@dataclass(frozen=True)
class SkinnedHumanSettings:
    """Everything tunable about the skinned hand (loaded from JSON)."""

    skin_rgb: tuple[float, float, float]
    breadth_band_m: float
    subdivision_levels: int
    fullness: tuple[FullnessSettings, ...]
    crease_bones: tuple[str, ...]
    crease_min_fold_deg: float
    scale_limits: tuple[float, float]
    palm_contact_azimuth_offset_deg: float
    palm_centre_wrap_deg: tuple[tuple[str, float], ...]
    grip_taper: float
    index_station_at_spec_zero: float
    seat_max_tilt_deg: float
    seat_max_travel_m: float
    seat_iterations: int
    finger_adduction_fraction: float
    index_knuckle_flexion_deg: float
    curl: CurlSettings
    limits_deg: tuple[tuple[str, tuple[float, float]], ...]
    pip_per_mcp: float
    dip_per_pip: float
    wrap_scan_samples: int
    arch_passes: int
    arch_gain: tuple[tuple[str, float], ...]
    arch_limit_deg: tuple[tuple[str, float], ...]
    thumb_max_swing_deg: float
    thumb_max_twist_deg: float
    thumb_press_range_deg: tuple[float, float]
    thumb_mcp_limits_deg: tuple[float, float]
    thumb_ip_limits_deg: tuple[float, float]
    thumb_straight_ip_limit_deg: float
    palm_cup_max_deg: float
    palm_cup_step_deg: float
    palm_cup_pivot_fraction: float
    thumb_toward_index_max_deg: tuple[tuple[str, float], ...]
    thumb_toward_index_steps: int
    thumb_toward_index_straightness: float
    thumb_index_gap_m: float
    thumb_knuckle_distance_m: float
    thumb_azimuth_close_rad: float
    thumb_toward_index_target: str
    thumb_target_of: tuple[tuple[str, str], ...]
    thumb_held_flexion_of: tuple[tuple[str, tuple[float, float]], ...]
    thumb_held_flexion_deg: tuple[float, float]
    side_close_max_deg: float
    side_close_gap_m: float
    side_close_index_gap_m: float
    side_close_scan_samples: int
    side_close_trigger_spread_deg: tuple[tuple[str, float], ...]
    side_close_trigger_back_off_deg: float
    side_close_knuckle_max_deg: float
    side_close_flexed_share: float
    side_close_flexed_at_deg: float
    side_close_limit_by_flexion: bool
    side_close_knuckle_abduction_deg: tuple[
        tuple[str, tuple[tuple[str, float], ...]], ...
    ]
    side_close_abduction_hold_margin_deg: float
    side_close_abduction_hold_passes: int
    pair_capsule_radius_scale: float
    pair_palm_capsule_radius_scale: float
    pair_thumb_radius_scale: float
    pair_grip_fingers: tuple[str, ...]
    pair_max_spacing_adjust_m: float
    pair_capsule_margin_m: float
    pair_obstacle: str
    pair_skin_margin_m: float
    pair_max_spacing_close_m: float
    pair_finger_gap_m: float
    pair_overlap_max_swing_deg: float
    pair_spacing_passes: int
    pair_spacing_pick: str
    pair_final_slide_max_m: float
    pair_overlap_bias_m: float
    pair_skin_reach_m: float
    pair_yield_max_depth_m: float
    pair_yield_passes: int
    pair_yield_clearance_m: float
    pair_palm_cup_max_deg: float
    pair_palm_wrap: PadConformSettings
    pair_palm_fold: PalmFoldSettings
    frame_excluded_bones: tuple[str, ...]
    fold_relax: FoldRelaxSettings
    palm_pad: PadConformSettings
    press: PressSettings
    squeeze_finger_depth_m: float
    squeeze_palm_depth_m: float
    crease_relax: FoldRelaxSettings
    crease_relax_bones: tuple[str, ...]
    crease_relax_max_added_fold_deg: float
    wrist_deg: tuple[tuple[str, tuple[float, float]], ...]
    metacarpal_arch: MetacarpalArchSettings

    def with_set_arch(self, side: str) -> SkinnedHumanSettings:
        """These settings for hand ``side``: a ray whose arch is set
        (``metacarpal_arch``) gives up the arch coupled to its knuckle.
        Postcondition: ``self`` itself if no ray of that hand is set."""
        held = self.metacarpal_arch.of(side)
        if not held:
            return self
        gains = tuple((ray, 0.0 if ray in held else g) for ray, g in self.arch_gain)
        return replace(self, arch_gain=gains)

    def with_thumb_target(self, side: str) -> SkinnedHumanSettings:
        """These settings for hand ``side``, whose thumb is laid to the target
        ``thumb.toward_index_target_of`` names for it. Precondition: ``side``
        is lead or trail (``ValueError``). Postcondition: ``self`` itself if
        that hand is named no target of its own."""
        if side not in _HANDS:
            raise ValueError(f"side must be one of {_HANDS}, got {side!r}")
        target = dict(self.thumb_target_of).get(side, self.thumb_toward_index_target)
        if target == self.thumb_toward_index_target:
            return self
        return replace(self, thumb_toward_index_target=target)

    def with_short_thumb(self, side: str) -> SkinnedHumanSettings:
        """These settings for hand ``side``, whose thumb is held as far flexed
        as ``thumb.held_flexion_deg_of`` names for it (a short thumb).
        Precondition: ``side`` is lead or trail (``ValueError``).
        Postcondition: ``self`` itself if that hand's thumb is held no further
        than these settings already hold it."""
        if side not in _HANDS:
            raise ValueError(f"side must be one of {_HANDS}, got {side!r}")
        held = dict(self.thumb_held_flexion_of).get(side, self.thumb_held_flexion_deg)
        if held == self.thumb_held_flexion_deg:
            return self
        return replace(self, thumb_held_flexion_deg=held)

    def beside_other_hand(self) -> SkinnedHumanSettings:
        """These settings for the trail hand of a pair. Its palm lies on the
        lead thumb, so it cups only as far as ``pair.palm_cup_max_deg``."""
        return replace(self, palm_cup_max_deg=self.pair_palm_cup_max_deg)

    def palm_pad_over(self, other_hand: bool) -> PadConformSettings:
        """How the palm pads settle: on a bare grip as ``correctives.palm_pad``
        says, over the other hand with the keys of ``pair.palm_wrap`` in place
        of those, so the palm can be made to cup round the thumb it lies on."""
        return self.pair_palm_wrap if other_hand else self.palm_pad

    def limit_rad(self, joint: str, finger: str = "") -> tuple[float, float]:
        """Flexion range of ``joint``; ``finger``'s own (``joint.finger``) if set."""
        limits = dict(self.limits_deg)
        low, high = limits.get(f"{joint}.{finger}", limits[joint])
        return math.radians(low), math.radians(high)

    def finger_limits_rad(self, finger: str) -> dict[str, tuple[float, float]]:
        """Flexion range of each segment's joint for ``finger``."""
        return {s: self.limit_rad(j, finger) for s, j in JOINT_OF_SEGMENT.items()}


def _check_limit_keys(limits: Mapping) -> None:
    """Every joint has a limit; other keys name ``joint.finger``."""
    for joint in JOINT_OF_SEGMENT.values():
        if joint not in limits:
            raise ValueError(f"limits_deg missing joint {joint!r}")
    for key in limits:
        joint, _, finger = key.partition(".")
        known = joint in JOINT_OF_SEGMENT.values() and finger in ("", *FINGERS)
        if not known or key.endswith("."):
            raise ValueError(f"limits_deg has no joint or finger for {key!r}")


def _pair(raw: Any, name: str) -> tuple[float, float]:
    values = tuple(float(x) for x in raw)
    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must be two finite numbers")
    if values[0] > values[1]:
        raise ValueError(f"{name} must be ordered low, high")
    return values[0], values[1]


def _unit_interval(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _fingers(raw: Any) -> tuple[str, ...]:
    names = tuple(str(x) for x in raw)
    if not set(names) <= set(FINGERS):
        raise ValueError(f"grip_fingers must be among {FINGERS}")
    return names


def _non_negative(value: Any, name: str) -> float:
    number = float(value)
    if not (math.isfinite(number) and number >= 0.0):
        raise ValueError(f"{name} must be non-negative and finite")
    return number


def _subdivision_levels(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("mesh.subdivision_levels must be an integer")
    if not 0 <= value <= _MAX_SUBDIVISION_LEVELS:
        raise ValueError(
            f"mesh.subdivision_levels must be 0..{_MAX_SUBDIVISION_LEVELS}"
        )
    return value


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not number > 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _fold_relax(raw: Mapping[str, Any]) -> FoldRelaxSettings:
    return FoldRelaxSettings(
        iterations=int(raw["iterations"]),
        rings=int(raw["rings"]),
        step=float(raw["step"]),
    )


def _fullness(raw: Mapping[str, Any]) -> FullnessSettings:
    return FullnessSettings(
        along_bone=str(raw["along_bone"]),
        bones=tuple(str(bone) for bone in raw["bones"]),
        height_m=float(raw["height_m"]),
        centre=float(raw["centre"]),
        half_span=float(raw["half_span"]),
        radius_m=float(raw["radius_m"]),
        facing_full=float(raw["facing_full"]),
    )


def _palm_pad(raw: Mapping[str, Any]) -> PadConformSettings:
    return PadConformSettings(
        bones=tuple(str(bone) for bone in raw["bones"]),
        reach_m=float(raw["reach_m"]),
        clearance_m=float(raw["clearance_m"]),
        facing_full=float(raw["facing_full"]),
        smooth_iterations=int(raw["smooth_iterations"]),
        passes=int(raw["passes"]),
        max_shift_m=float(raw["max_shift_m"]),
    )


def _corrective_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The ``correctives`` section as :class:`SkinnedHumanSettings` fields."""
    crease: Mapping[str, Any] = raw["crease_relax"]
    return {
        "fold_relax": _fold_relax(raw["fold_relax"]),
        "palm_pad": _palm_pad(raw["palm_pad"]),
        "crease_relax": _fold_relax(crease),
        "crease_relax_bones": tuple(str(b) for b in crease["bones"]),
        "crease_relax_max_added_fold_deg": _positive(
            crease["max_added_fold_deg"], "crease_relax.max_added_fold_deg"
        ),
        "press": PressSettings(
            max_depth_m=float(raw["lift_out"]["max_depth_m"]),
            bulge_gain=float(raw["lift_out"]["bulge_gain"]),
            bulge_rings=int(raw["lift_out"]["bulge_rings"]),
        ),
    }


def _flag(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError("side_close.limit_by_flexion must be true or false")
    return value


def _share(value: Any) -> float:
    share = float(value)
    if not 0.0 <= share <= 1.0:
        raise ValueError("side_close.flexed_share must be within 0..1")
    return share


def _knuckle_abduction(
    raw: Mapping[str, Any],
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    """``side_close.knuckle_abduction_deg``: per hand, the fingers abducted
    toward the little finger after the sideways closing, each by 0 to
    ``knuckle_max_deg`` degrees; a finger named is also held in its range."""
    name, most = "side_close.knuckle_abduction_deg", float(raw["knuckle_max_deg"])
    hands = raw["knuckle_abduction_deg"]
    if not isinstance(hands, Mapping) or set(hands) != set(_HANDS):
        raise ValueError(f"{name} must name the hands {_HANDS}")
    out = []
    for side in _HANDS:
        fingers = tuple((str(f), float(deg)) for f, deg in hands[side].items())
        for finger, deg in fingers:
            if finger not in ABDUCTED_FINGERS or not 0.0 <= deg <= most:
                raise ValueError(f"{name}.{side}: {finger!r} {deg} is not allowed")
        out.append((side, fingers))
    return tuple(out)


def _side_close_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The ``side_close`` section as :class:`SkinnedHumanSettings` fields."""
    return {
        "side_close_max_deg": _non_negative(raw["max_deg"], "side_close.max_deg"),
        "side_close_gap_m": _non_negative(raw["gap_m"], "side_close.gap_m"),
        "side_close_index_gap_m": _non_negative(
            raw["index_gap_m"], "side_close.index_gap_m"
        ),
        "side_close_scan_samples": int(
            _non_negative(raw["scan_samples"], "side_close.scan_samples")
        ),
        "side_close_trigger_spread_deg": tuple(
            (side, _non_negative(raw["trigger_spread_deg"][side], side))
            for side in ("lead", "trail")
        ),
        "side_close_trigger_back_off_deg": _positive(
            raw["trigger_back_off_deg"], "side_close.trigger_back_off_deg"
        ),
        "side_close_knuckle_max_deg": _non_negative(
            raw["knuckle_max_deg"], "side_close.knuckle_max_deg"
        ),
        "side_close_flexed_share": _share(raw["flexed_share"]),
        "side_close_flexed_at_deg": _positive(
            raw["flexed_at_deg"], "side_close.flexed_at_deg"
        ),
        "side_close_limit_by_flexion": _flag(raw["limit_by_flexion"]),
        "side_close_knuckle_abduction_deg": _knuckle_abduction(raw),
        "side_close_abduction_hold_margin_deg": _non_negative(
            raw["abduction_hold_margin_deg"], "side_close.abduction_hold_margin_deg"
        ),
        "side_close_abduction_hold_passes": int(
            _positive(raw["abduction_hold_passes"], "side_close.abduction_hold_passes")
        ),
    }


#: What stands in for the lead hand when the trail hand is posed around it.
#: What the thumb's turn brings it to (``thumb.toward_index_target``).
THUMB_TARGETS = ("azimuth", "knuckle", "fingertip")
_HANDS = ("lead", "trail")


def _thumb_target_of(raw: Any) -> tuple[tuple[str, str], ...]:
    """``thumb.toward_index_target_of``: a thumb target for a hand of its own."""
    if not isinstance(raw, Mapping):
        raise TypeError("thumb.toward_index_target_of must be a mapping")
    name = "thumb.toward_index_target_of"
    return tuple(
        (_one_of(side, _HANDS, name), _one_of(target, THUMB_TARGETS, name))
        for side, target in raw.items()
    )


def _thumb_held_flexion_of(
    thumb: Mapping[str, Any],
) -> tuple[tuple[str, tuple[float, float]], ...]:
    """``thumb.held_flexion_deg_of``: per hand, the least flexion of the
    thumb's knuckle and end joint, each within that joint's limits."""
    raw, name = thumb["held_flexion_deg_of"], "thumb.held_flexion_deg_of"
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name} must be a mapping")
    limits = [_pair(thumb[key], key) for key in ("mcp_limits_deg", "ip_limits_deg")]
    held = []
    for side, angles in raw.items():
        if isinstance(angles, (str, bytes)) or not isinstance(angles, Sequence):
            raise TypeError(f"{name} needs two angles for a hand")
        knuckle, end = (float(angle) for angle in angles)  # two, or ValueError
        for angle, (_, most) in zip((knuckle, end), limits, strict=True):
            if not 0.0 <= angle <= most:
                raise ValueError(f"{name}: {angle} is outside 0 to {most} degrees")
        held.append((_one_of(side, _HANDS, name), (knuckle, end)))
    return tuple(held)


PAIR_OBSTACLES = ("skin", "capsules")
PAIR_SPACING_PICKS = ("best", "last")


def _one_of(raw: Any, allowed: tuple[str, ...], name: str) -> str:
    if raw not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {raw!r}")
    return str(raw)


def _pair_fields(pair: Mapping[str, Any]) -> dict[str, Any]:
    """The ``pair`` section as :class:`SkinnedHumanSettings` fields."""
    return {
        "pair_capsule_radius_scale": _unit_interval(
            pair["capsule_radius_scale"], "capsule_radius_scale"
        ),
        "pair_palm_capsule_radius_scale": _positive(
            pair["palm_capsule_radius_scale"], "palm_capsule_radius_scale"
        ),
        "pair_thumb_radius_scale": _unit_interval(
            pair["thumb_radius_scale"], "thumb_radius_scale"
        ),
        "pair_grip_fingers": _fingers(pair["grip_fingers"]),
        "pair_max_spacing_adjust_m": _non_negative(
            pair["max_spacing_adjust_m"], "max_spacing_adjust_m"
        ),
        "pair_capsule_margin_m": _non_negative(
            pair["capsule_margin_m"], "capsule_margin_m"
        ),
        "pair_obstacle": _one_of(pair["obstacle"], PAIR_OBSTACLES, "pair.obstacle"),
        "pair_skin_margin_m": _non_negative(pair["skin_margin_m"], "skin_margin_m"),
        "pair_max_spacing_close_m": _non_negative(
            pair["max_spacing_close_m"], "max_spacing_close_m"
        ),
        "pair_finger_gap_m": _non_negative(pair["finger_gap_m"], "finger_gap_m"),
        "pair_skin_reach_m": _positive(pair["skin_reach_m"], "skin_reach_m"),
        "pair_yield_max_depth_m": _non_negative(
            pair["yield_max_depth_m"], "yield_max_depth_m"
        ),
        "pair_yield_passes": int(_non_negative(pair["yield_passes"], "yield_passes")),
        "pair_yield_clearance_m": _non_negative(
            pair["yield_clearance_m"], "yield_clearance_m"
        ),
        "pair_palm_cup_max_deg": _non_negative(
            pair["palm_cup_max_deg"], "pair.palm_cup_max_deg"
        ),
        "pair_overlap_bias_m": _finite(pair["overlap_bias_m"], "overlap_bias_m"),
        "pair_spacing_passes": int(
            _non_negative(pair["spacing_passes"], "spacing_passes")
        ),
        "pair_final_slide_max_m": _non_negative(
            pair["final_slide_max_m"], "pair.final_slide_max_m"
        ),
        "pair_spacing_pick": _one_of(
            pair["spacing_pick"], PAIR_SPACING_PICKS, "pair.spacing_pick"
        ),
        "pair_overlap_max_swing_deg": _non_negative(
            pair["overlap_max_swing_deg"], "overlap_max_swing_deg"
        ),
    }


GRIP_SHAPES = ("tapered", "cylinder")


def _grip_taper(shape: Any) -> float:
    """Radius change per metre of the grip the hands hold (``grip.shape``)."""
    if shape not in GRIP_SHAPES:
        raise ValueError(f"grip.shape must be one of {GRIP_SHAPES}")
    return load_golf_grip().taper if shape == "tapered" else 0.0


def _thumb_turn_fields(thumb: Mapping) -> dict[str, Any]:
    """The thumb's turn toward the forefinger tip (#15181)."""
    return {
        "thumb_toward_index_max_deg": tuple(
            (side, _non_negative(thumb["toward_index_max_deg"][side], side))
            for side in ("lead", "trail")
        ),
        "thumb_toward_index_steps": int(
            _positive(thumb["toward_index_steps"], "toward_index_steps")
        ),
        "thumb_toward_index_straightness": _unit_interval(
            thumb["toward_index_straightness"], "toward_index_straightness"
        ),
        "thumb_index_gap_m": _non_negative(thumb["index_gap_m"], "index_gap_m"),
        "thumb_knuckle_distance_m": _non_negative(
            thumb["knuckle_distance_m"], "knuckle_distance_m"
        ),
        "thumb_azimuth_close_rad": math.radians(
            _non_negative(thumb["azimuth_tolerance_deg"], "azimuth_tolerance_deg")
        ),
        "thumb_toward_index_target": _one_of(
            thumb["toward_index_target"], THUMB_TARGETS, "thumb.toward_index_target"
        ),
        "thumb_target_of": _thumb_target_of(thumb["toward_index_target_of"]),
        "thumb_held_flexion_of": _thumb_held_flexion_of(thumb),
        "thumb_held_flexion_deg": (0.0, 0.0),
    }


def settings_from_mapping(raw: Mapping[str, Any]) -> SkinnedHumanSettings:
    """Validate a settings mapping. Raises ``ValueError``/``KeyError`` if bad."""
    rgb = tuple(_unit_interval(x, "skin_rgb") for x in raw["skin_rgb"])
    if len(rgb) != 3:
        raise ValueError("skin_rgb must be three numbers in [0, 1]")
    anthro: Mapping = raw["anthropometry"]
    place: Mapping = raw["placement"]
    pre: Mapping = raw["preshape"]
    curl: Mapping = raw["curl"]
    thumb: Mapping = raw["thumb"]
    passes = int(curl["arch_passes"])
    if passes < 1:
        raise ValueError("arch_passes must be at least 1")
    for finger in FINGERS:
        if finger not in curl["arch_gain"] or finger not in curl["arch_limit_deg"]:
            raise ValueError(f"arch settings missing finger {finger!r}")
    _check_limit_keys(curl["limits_deg"])
    return SkinnedHumanSettings(
        skin_rgb=(rgb[0], rgb[1], rgb[2]),
        breadth_band_m=float(anthro["breadth_band_m"]),
        subdivision_levels=_subdivision_levels(raw["mesh"]["subdivision_levels"]),
        fullness=tuple(_fullness(pad) for pad in raw["mesh"]["fullness"]),
        crease_bones=tuple(str(b) for b in raw["mesh"]["creases"]["bones"]),
        crease_min_fold_deg=_positive(
            raw["mesh"]["creases"]["min_fold_deg"], "mesh.creases.min_fold_deg"
        ),
        scale_limits=_pair(anthro["scale_limits"], "scale_limits"),
        palm_contact_azimuth_offset_deg=float(place["palm_contact_azimuth_offset_deg"]),
        grip_taper=_grip_taper(raw["grip"]["shape"]),
        palm_centre_wrap_deg=tuple(
            (side, _finite(place["palm_centre_wrap_deg"][side], side))
            for side in ("lead", "trail")
        ),
        index_station_at_spec_zero=_unit_interval(
            place["index_station_at_spec_zero"], "index_station_at_spec_zero"
        ),
        seat_max_tilt_deg=float(place["seat_max_tilt_deg"]),
        seat_max_travel_m=float(place["seat_max_travel_m"]),
        seat_iterations=int(place["seat_iterations"]),
        finger_adduction_fraction=_unit_interval(
            pre["finger_adduction_fraction"], "finger_adduction_fraction"
        ),
        index_knuckle_flexion_deg=_non_negative(
            pre["index_knuckle_flexion_deg"], "index_knuckle_flexion_deg"
        ),
        curl=CurlSettings(
            step_rad=math.radians(float(curl["step_deg"])),
            bisection_steps=int(curl["bisection_steps"]),
            contact_tolerance_m=float(curl["contact_tolerance_m"]),
            penetration_allowance_m=float(curl["penetration_allowance_m"]),
        ),
        limits_deg=tuple(
            (k, _pair(v, f"limits_deg.{k}")) for k, v in curl["limits_deg"].items()
        ),
        pip_per_mcp=float(curl["synergy"]["pip_per_mcp"]),
        dip_per_pip=float(curl["synergy"]["dip_per_pip"]),
        wrap_scan_samples=int(curl["wrap_scan_samples"]),
        arch_passes=passes,
        arch_gain=tuple((k, float(v)) for k, v in curl["arch_gain"].items()),
        arch_limit_deg=tuple((k, float(v)) for k, v in curl["arch_limit_deg"].items()),
        thumb_max_swing_deg=float(thumb["max_swing_deg"]),
        thumb_max_twist_deg=float(thumb["max_twist_deg"]),
        thumb_press_range_deg=_pair(thumb["press_range_deg"], "press_range_deg"),
        thumb_mcp_limits_deg=_pair(thumb["mcp_limits_deg"], "mcp_limits_deg"),
        thumb_ip_limits_deg=_pair(thumb["ip_limits_deg"], "ip_limits_deg"),
        thumb_straight_ip_limit_deg=float(thumb["straight_ip_limit_deg"]),
        palm_cup_max_deg=_non_negative(thumb["palm_cup_max_deg"], "palm_cup_max_deg"),
        palm_cup_step_deg=_positive(thumb["palm_cup_step_deg"], "palm_cup_step_deg"),
        palm_cup_pivot_fraction=_non_negative(
            thumb["palm_cup_pivot_fraction"], "palm_cup_pivot_fraction"
        ),
        **_thumb_turn_fields(thumb),
        **_pair_fields(raw["pair"]),
        frame_excluded_bones=tuple(
            str(bone) for bone in raw["framing"]["excluded_bones"]
        ),
        **_corrective_fields(raw["correctives"]),
        pair_palm_wrap=_palm_pad(
            {**raw["correctives"]["palm_pad"], **raw["pair"]["palm_wrap"]}
        ),
        pair_palm_fold=PalmFoldSettings.from_mapping(raw["pair"]["palm_fold"]),
        **_side_close_fields(raw["side_close"]),
        squeeze_finger_depth_m=_non_negative(
            raw["squeeze"]["finger_depth_m"], "squeeze.finger_depth_m"
        ),
        squeeze_palm_depth_m=_non_negative(
            raw["squeeze"]["palm_depth_m"], "squeeze.palm_depth_m"
        ),
        wrist_deg=tuple(
            (
                side,
                (
                    float(raw["wrist"][side]["extension_deg"]),
                    float(raw["wrist"][side]["ulnar_deviation_deg"]),
                ),
            )
            for side in ("lead", "trail")
        ),
        metacarpal_arch=MetacarpalArchSettings.from_mapping(raw["metacarpal_arch"]),
    )


def load_settings_mapping() -> dict[str, Any]:
    """The raw settings JSON (for callers that override a few values)."""
    if not SETTINGS_PATH.is_file():
        raise ValueError(f"missing settings: {SETTINGS_PATH}")
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def default_settings() -> SkinnedHumanSettings:
    return settings_from_mapping(load_settings_mapping())

"""Canonical JSON and SHA-256 for HRG-GB1 grip specs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from src.shared.python.contracts import require
from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_spec.defaults import (
    DEFAULT_CONTACT_PROVIDER_NAME,
    DEFAULT_CURL_ORDER,
    DEFAULT_FORCE_PROVIDER_NAME,
    LEGACY_CLUB_CROSS_ANGLE_DEG,
    LEGACY_LEAD_THUMB_AZIMUTH_DEG,
    LEGACY_LEAD_V_AZIMUTH_DEG,
    SCHEMA_VERSION,
)
from src.shared.python.hand_size import DEFAULT_HAND_LENGTH_M, resolve_hand_breadth_m
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec, HandSize


def _encode(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    return value


def authoring_spec_to_canonical_dict(spec: GripSpec) -> dict[str, Any]:
    return _encode(spec)


def canonical_spec_sha256(spec: GripSpec) -> str:
    payload = json.dumps(
        authoring_spec_to_canonical_dict(spec),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hand_grip_from_dict(data: dict[str, Any]) -> HandGripSpec:
    overrides = data.get("joint_coupling_overrides") or []
    return HandGripSpec(
        club_cross_angle_deg=float(
            data.get("club_cross_angle_deg", LEGACY_CLUB_CROSS_ANGLE_DEG)
        ),
        index_contact_station=float(data.get("index_contact_station", 0.0)),
        pinky_side_palm_depth_m=float(data.get("pinky_side_palm_depth_m", 0.0)),
        palm_finger_depth=float(data.get("palm_finger_depth", 0.5)),
        v_azimuth_deg=float(data.get("v_azimuth_deg", LEGACY_LEAD_V_AZIMUTH_DEG)),
        thumb_azimuth_deg=float(
            data.get("thumb_azimuth_deg", LEGACY_LEAD_THUMB_AZIMUTH_DEG)
        ),
        thumb_straightness=float(data.get("thumb_straightness", 1.0)),
        joint_coupling_overrides=tuple(
            (str(a), float(b))
            for a, b in overrides  # type: ignore[misc]
        ),
    )


def authoring_spec_from_dict(payload: dict[str, Any]) -> GripSpec:
    version = payload.get("schema_version", SCHEMA_VERSION)
    require(
        version == SCHEMA_VERSION,
        f"unsupported schema_version {version!r}, expected {SCHEMA_VERSION}",
    )
    style_raw = payload.get("style", GripStyle.OVERLAP.value)
    style = style_raw if isinstance(style_raw, GripStyle) else GripStyle(str(style_raw))
    hand_size_data = payload.get("hand_size") or {}
    default_breadth = resolve_hand_breadth_m(DEFAULT_HAND_LENGTH_M)
    hand_size = HandSize(
        length_m=float(hand_size_data.get("length_m", DEFAULT_HAND_LENGTH_M)),
        breadth_m=float(hand_size_data.get("breadth_m", default_breadth)),
    )
    lead_data = payload.get("lead") or {}
    trail_data = payload.get("trail") or {}
    curl = payload.get("curl_order") or DEFAULT_CURL_ORDER
    return GripSpec(
        schema_version=int(version),
        handedness=payload.get("handedness", "right"),  # type: ignore[arg-type]
        style=style,
        trigger_finger=bool(payload.get("trigger_finger", False)),
        hand_size=hand_size,
        lead=_hand_grip_from_dict(lead_data),
        trail=_hand_grip_from_dict(trail_data),
        lead_below_butt_m=float(payload.get("lead_below_butt_m", 0.032)),
        hand_spacing_m=float(payload.get("hand_spacing_m", 0.0762)),
        curl_order=tuple(str(x) for x in curl),
        force_provider_name=str(
            payload.get(
                "force_provider_name",
                DEFAULT_FORCE_PROVIDER_NAME,
            )
        ),
        contact_provider_name=str(
            payload.get(
                "contact_provider_name",
                DEFAULT_CONTACT_PROVIDER_NAME,
            )
        ),
    )


__all__ = [
    "authoring_spec_from_dict",
    "authoring_spec_to_canonical_dict",
    "canonical_spec_sha256",
]

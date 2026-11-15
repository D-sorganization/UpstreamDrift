"""Versioned per-hand grip authoring types (HRG-GB1, #15123)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from src.shared.python.contracts import ensure, require
from src.shared.python.core.handedness import (
    DEFAULT_HANDEDNESS,
    Handedness,
    validate_handedness,
)
from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.hand_size import DEFAULT_HAND_LENGTH_M, resolve_hand_breadth_m
from src.shared.python.model_appearance.grip_pose import (
    LEAD_BELOW_BUTT_M,
    LEAD_V_AZIMUTH_MAX_DEG,
    LEAD_V_AZIMUTH_MIN_DEG,
    NATIVE_HAND_SPACING_M,
    TRAIL_V_OFFSET_RANGE_DEG,
)

from src.shared.python.grip_spec.defaults import (
    DEFAULT_CONTACT_PROVIDER_NAME,
    DEFAULT_CURL_ORDER,
    DEFAULT_FORCE_PROVIDER_NAME,
    LEGACY_CLUB_CROSS_ANGLE_DEG,
    LEGACY_LEAD_THUMB_AZIMUTH_DEG,
    LEGACY_LEAD_V_AZIMUTH_DEG,
    LEGACY_TRAIL_THUMB_AZIMUTH_DEG,
    LEGACY_TRAIL_V_AZIMUTH_DEG,
    NEUTRAL_LEAD_THUMB_AZIMUTH_DEG,
    NEUTRAL_TRAIL_THUMB_AZIMUTH_DEG,
    SCHEMA_VERSION,
)
from src.shared.python.grip_spec.validation import (
    require_fields_in_range,
    validate_grip_style_fields,
)


def _default_hand_breadth_m() -> float:
    return resolve_hand_breadth_m(DEFAULT_HAND_LENGTH_M)


@dataclass(frozen=True)
class HandSize:
    """Anthropometric hand scale (metres)."""

    length_m: float = DEFAULT_HAND_LENGTH_M
    breadth_m: float = field(default_factory=_default_hand_breadth_m)

    def __post_init__(self) -> None:
        require_fields_in_range(
            self,
            (
                ("length_m", 0.14, 0.24),
                ("breadth_m", 0.05, 0.12),
            ),
        )


@dataclass(frozen=True)
class HandGripSpec:
    """Per-hand grip geometry parameters (degrees at the spec boundary)."""

    club_cross_angle_deg: float = LEGACY_CLUB_CROSS_ANGLE_DEG
    index_contact_station: float = 0.0
    pinky_side_palm_depth_m: float = 0.0
    palm_finger_depth: float = 0.5
    v_azimuth_deg: float = LEGACY_LEAD_V_AZIMUTH_DEG
    thumb_azimuth_deg: float = LEGACY_LEAD_THUMB_AZIMUTH_DEG
    thumb_straightness: float = 1.0
    joint_coupling_overrides: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        require_fields_in_range(
            self,
            (
                ("club_cross_angle_deg", 0.0, 90.0),
                ("index_contact_station", 0.0, 1.0),
                ("palm_finger_depth", 0.0, 1.0),
                ("thumb_straightness", 0.0, 1.0),
                ("v_azimuth_deg", LEAD_V_AZIMUTH_MIN_DEG, LEAD_V_AZIMUTH_MAX_DEG),
                ("thumb_azimuth_deg", 0.0, 45.0),
            ),
        )
        require(
            isinstance(self.joint_coupling_overrides, tuple),
            "joint_coupling_overrides must be a tuple",
        )


@dataclass(frozen=True)
class GripSpec:
    """Versioned golf grip authoring contract (engine-free)."""

    schema_version: int = SCHEMA_VERSION
    handedness: Handedness = DEFAULT_HANDEDNESS
    style: GripStyle = GripStyle.OVERLAP
    trigger_finger: bool = False
    hand_size: HandSize = field(default_factory=HandSize)
    lead: HandGripSpec = field(default_factory=HandGripSpec)
    trail: HandGripSpec = field(
        default_factory=lambda: HandGripSpec(
            v_azimuth_deg=LEGACY_TRAIL_V_AZIMUTH_DEG,
            thumb_azimuth_deg=LEGACY_TRAIL_THUMB_AZIMUTH_DEG,
        )
    )
    lead_below_butt_m: float = LEAD_BELOW_BUTT_M
    hand_spacing_m: float = NATIVE_HAND_SPACING_M
    curl_order: tuple[str, ...] = DEFAULT_CURL_ORDER
    force_provider_name: str = DEFAULT_FORCE_PROVIDER_NAME
    contact_provider_name: str = DEFAULT_CONTACT_PROVIDER_NAME

    def __post_init__(self) -> None:
        require(self.schema_version == SCHEMA_VERSION, "unsupported schema_version")
        validate_grip_style_fields(self.style, self.trigger_finger)
        if not isinstance(self.hand_size, HandSize):
            raise TypeError("hand_size must be a HandSize")
        if not isinstance(self.lead, HandGripSpec) or not isinstance(
            self.trail, HandGripSpec
        ):
            raise TypeError("lead and trail must be HandGripSpec")
        validate_handedness(self.handedness)
        require_fields_in_range(
            self,
            (
                ("lead_below_butt_m", 0.0, 0.1),
                ("hand_spacing_m", 0.0, 0.2),
            ),
        )
        require(self.hand_spacing_m > 0.0, "hand_spacing_m must be positive")
        ensure(len(self.curl_order) >= 1, "curl_order must not be empty")
        self._validate_projectable_to_grip_pose()

    def _validate_projectable_to_grip_pose(self) -> None:
        """Ensure per-hand fields map to a valid :class:`GripPose` without silent fixes."""
        trail_v_offset_deg = self.trail.v_azimuth_deg - self.lead.v_azimuth_deg
        lo, hi = TRAIL_V_OFFSET_RANGE_DEG
        require(
            lo <= trail_v_offset_deg <= hi,
            (
                "trail V offset "
                f"{trail_v_offset_deg:.4g} deg (trail v_azimuth_deg minus lead "
                "v_azimuth_deg) is outside GripPose trail_v_offset_deg range "
                f"[{lo}, {hi}]"
            ),
        )

    @classmethod
    def legacy(cls, **overrides: Any) -> GripSpec:
        """Spec that projects to today's default :class:`GripPose` exactly."""
        base = cls(
            lead=HandGripSpec(
                club_cross_angle_deg=LEGACY_CLUB_CROSS_ANGLE_DEG,
                v_azimuth_deg=LEGACY_LEAD_V_AZIMUTH_DEG,
                thumb_azimuth_deg=LEGACY_LEAD_THUMB_AZIMUTH_DEG,
            ),
            trail=HandGripSpec(
                club_cross_angle_deg=LEGACY_CLUB_CROSS_ANGLE_DEG,
                v_azimuth_deg=LEGACY_TRAIL_V_AZIMUTH_DEG,
                thumb_azimuth_deg=LEGACY_TRAIL_THUMB_AZIMUTH_DEG,
            ),
        )
        return replace(base, **overrides) if overrides else base

    @classmethod
    def neutral(cls, *, handedness: Handedness = DEFAULT_HANDEDNESS) -> GripSpec:
        """Dieter's neutral thumb targets with legacy shaft cross defaults."""
        lead = HandGripSpec(thumb_azimuth_deg=NEUTRAL_LEAD_THUMB_AZIMUTH_DEG)
        trail = HandGripSpec(
            v_azimuth_deg=LEGACY_TRAIL_V_AZIMUTH_DEG,
            thumb_azimuth_deg=NEUTRAL_TRAIL_THUMB_AZIMUTH_DEG,
        )
        return cls(lead=lead, trail=trail, handedness=handedness)

    def resolved_thumb_azimuth_deg(self, role: Literal["lead", "trail"]) -> float:
        """Thumb azimuth with handedness mirroring (see package docstring)."""
        raw = (
            self.lead.thumb_azimuth_deg
            if role == "lead"
            else self.trail.thumb_azimuth_deg
        )
        if self.handedness == "left":
            return -raw
        return raw

    def to_grip_pose(self):
        from src.shared.python.grip_spec.projection import grip_pose_from_authoring_spec
        from src.shared.python.grip_spec.providers import resolve_grip_force_limits

        return grip_pose_from_authoring_spec(
            self,
            force_limits=resolve_grip_force_limits(self),
        )

    @property
    def spec_sha256(self) -> str:
        from src.shared.python.grip_spec.serialization import canonical_spec_sha256

        return canonical_spec_sha256(self)

    def to_dict(self) -> dict[str, Any]:
        from src.shared.python.grip_spec.serialization import (
            authoring_spec_to_canonical_dict,
        )

        return authoring_spec_to_canonical_dict(self)

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GripSpec:
        from src.shared.python.grip_spec.serialization import authoring_spec_from_dict

        return authoring_spec_from_dict(payload)

    @classmethod
    def from_json(cls, text: str) -> GripSpec:
        import json

        return cls.from_dict(json.loads(text))


__all__ = ["HandSize", "HandGripSpec", "GripSpec"]

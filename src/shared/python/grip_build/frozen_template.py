"""Frozen grip template export and fail-closed loader (HRG-GB5, #15127).

A frozen template is the *data only* capture of one procedural grip build: per-hand
capsule poses in the hand-grip frame, the joint table, the contact table and a
provenance stamp. It is deliberately not wired to any physics engine as collision or
mass, and loading refuses a template whose stamp does not match the caller's spec and
builder so a stale capture can never be drawn or consumed as current.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.shared.python.grip_build.full_builder import (
    BUILDER_VERSION as GB3_BUILDER_VERSION,
    build_full_grip,
)
from src.shared.python.grip_build.hand_chain import (
    _DEFAULT_DIP_OVER_PIP,
    FingerJointAngles,
    HandChain,
)
from src.shared.python.grip_build.receipt import BuildStatus, FullGripBuildResult
from src.shared.python.grip_build.thumb import ThumbJointAngles, thumb_capsules
from src.shared.python.grip_spec.types import GripSpec, HandSize
from src.shared.python.grip_build.finger_order import LEAD_FINGER_CURL_ORDER

FROZEN_TEMPLATE_SCHEMA_ID: Final[str] = "grip-frozen-template/v1"
FROZEN_TEMPLATE_SCHEMA_VERSION: Final[int] = 1
TEMPLATE_BUILDER_VERSION: Final[str] = "gb5-0.1.0"
TEMPLATE_HAND_ROLES: Final[tuple[str, ...]] = ("lead", "trail")


class FrozenTemplateError(ValueError):
    """Base class for every fail-closed frozen-template rejection."""


class FrozenTemplateStampError(FrozenTemplateError):
    """The template's provenance stamp does not match the caller's expectation."""


@dataclass(frozen=True)
class CapsulePose:
    """One frozen phalanx capsule core segment in the hand-grip frame."""

    owner: str
    segment: int
    p0: tuple[float, float, float]
    p1: tuple[float, float, float]
    radius_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "segment": self.segment,
            "p0": list(self.p0),
            "p1": list(self.p1),
            "radius_m": self.radius_m,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CapsulePose:
        if not isinstance(raw, dict):
            raise FrozenTemplateError("capsule entry must be a JSON object")
        owner = raw.get("owner")
        segment = raw.get("segment")
        radius = raw.get("radius_m")
        if not isinstance(owner, str) or not owner:
            raise FrozenTemplateError("capsule owner must be a non-empty str")
        if not isinstance(segment, int) or segment < 0:
            raise FrozenTemplateError(
                f"capsule segment must be a non-negative int, got {segment!r}"
            )
        if (
            not isinstance(radius, (int, float))
            or not np.isfinite(radius)
            or radius <= 0.0
        ):
            raise FrozenTemplateError(
                f"capsule radius_m must be a positive finite number, got {radius!r}"
            )
        return cls(
            owner=owner,
            segment=segment,
            p0=_vec3(raw.get("p0"), "p0", owner),
            p1=_vec3(raw.get("p1"), "p1", owner),
            radius_m=float(radius),
        )


@dataclass(frozen=True)
class HandTemplate:
    """Frozen pose of one hand: capsules, joint table and build parameters."""

    role: str
    curl_sign: int
    flat_lateral_offset_m: float
    hand_length_m: float
    joints: tuple[tuple[str, float, float, float], ...]
    thumb: tuple[float, float, float]
    thumb_azimuth_rad: float
    palm_wrap_rad: float
    grip_radius_m: float
    axial_shift_m: float
    dip_over_pip: tuple[tuple[str, float], ...]
    capsules: tuple[CapsulePose, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "curl_sign": self.curl_sign,
            "flat_lateral_offset_m": self.flat_lateral_offset_m,
            "hand_length_m": self.hand_length_m,
            "joints": [list(row) for row in self.joints],
            "thumb": list(self.thumb),
            "thumb_azimuth_rad": self.thumb_azimuth_rad,
            "palm_wrap_rad": self.palm_wrap_rad,
            "grip_radius_m": self.grip_radius_m,
            "axial_shift_m": self.axial_shift_m,
            "dip_over_pip": [list(row) for row in self.dip_over_pip],
            "capsules": [c.to_dict() for c in self.capsules],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> HandTemplate:
        if not isinstance(raw, dict):
            raise FrozenTemplateError("hand template must be a JSON object")
        role = raw.get("role")
        if role not in TEMPLATE_HAND_ROLES:
            raise FrozenTemplateError(
                f"hand role must be one of {TEMPLATE_HAND_ROLES}, got {role!r}"
            )
        curl_sign = raw.get("curl_sign")
        if curl_sign not in (-1, 1):
            raise FrozenTemplateError(
                f"{role} curl_sign must be -1 or 1, got {curl_sign!r}"
            )
        capsules_raw = raw.get("capsules")
        if not isinstance(capsules_raw, list) or not capsules_raw:
            raise FrozenTemplateError(f"{role} capsules must be a non-empty list")
        capsules = tuple(CapsulePose.from_dict(c) for c in capsules_raw)
        joints_raw = raw.get("joints")
        if not isinstance(joints_raw, list) or not joints_raw:
            raise FrozenTemplateError(f"{role} joints must be a non-empty list")
        joints: list[tuple[str, float, float, float]] = []
        for row in joints_raw:
            if (
                not isinstance(row, list)
                or len(row) != 4
                or not isinstance(row[0], str)
            ):
                raise FrozenTemplateError(
                    f"{role} joint row must be [name, mcp, pip, dip], got {row!r}"
                )
            joints.append((row[0], float(row[1]), float(row[2]), float(row[3])))
        thumb_raw = raw.get("thumb")
        if not isinstance(thumb_raw, list) or len(thumb_raw) != 3:
            raise FrozenTemplateError(
                f"{role} thumb must be [cmc, mcp, ip], got {thumb_raw!r}"
            )
        coupling_raw = raw.get("dip_over_pip")
        if not isinstance(coupling_raw, list):
            raise FrozenTemplateError(f"{role} dip_over_pip must be a list")
        coupling = tuple((str(row[0]), float(row[1])) for row in coupling_raw)
        for key in (
            "flat_lateral_offset_m",
            "hand_length_m",
            "thumb_azimuth_rad",
            "palm_wrap_rad",
            "grip_radius_m",
            "axial_shift_m",
        ):
            value = raw.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise FrozenTemplateError(
                    f"{role} {key} must be a finite number, got {value!r}"
                )
        return cls(
            role=role,
            curl_sign=int(curl_sign),
            flat_lateral_offset_m=float(raw["flat_lateral_offset_m"]),
            hand_length_m=float(raw["hand_length_m"]),
            joints=tuple(joints),
            thumb=(float(thumb_raw[0]), float(thumb_raw[1]), float(thumb_raw[2])),
            thumb_azimuth_rad=float(raw["thumb_azimuth_rad"]),
            palm_wrap_rad=float(raw["palm_wrap_rad"]),
            grip_radius_m=float(raw["grip_radius_m"]),
            axial_shift_m=float(raw["axial_shift_m"]),
            dip_over_pip=coupling,
            capsules=capsules,
        )


@dataclass(frozen=True)
class FrozenGripTemplate:
    """Data-only capture of a built grip, stamped with its provenance."""

    schema: str
    schema_version: int
    spec_sha256: str
    builder_version: str
    grip_build_builder_version: str
    handedness: str
    style: str
    status: str
    hands: tuple[HandTemplate, ...]
    contact_clearances_m: tuple[tuple[str, float], ...]
    active_contact_pairs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != FROZEN_TEMPLATE_SCHEMA_ID:
            raise FrozenTemplateError(
                f"unsupported frozen template schema {self.schema!r} (expected {FROZEN_TEMPLATE_SCHEMA_ID!r})"
            )
        if self.schema_version != FROZEN_TEMPLATE_SCHEMA_VERSION:
            raise FrozenTemplateError(
                f"unsupported frozen template schema_version {self.schema_version!r}"
                f" (expected {FROZEN_TEMPLATE_SCHEMA_VERSION})"
            )
        if len(self.spec_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.spec_sha256
        ):
            raise FrozenTemplateError(
                f"spec_sha256 must be 64 lowercase hex chars, got {self.spec_sha256!r}"
            )
        if not self.builder_version or not self.grip_build_builder_version:
            raise FrozenTemplateError(
                "builder_version and grip_build_builder_version must be non-empty"
            )
        roles = tuple(hand.role for hand in self.hands)
        if roles != TEMPLATE_HAND_ROLES:
            raise FrozenTemplateError(
                f"hands must be ordered {TEMPLATE_HAND_ROLES}, got {roles}"
            )

    def hand(self, role: str) -> HandTemplate:
        for hand in self.hands:
            if hand.role == role:
                return hand
        raise FrozenTemplateError(f"no {role!r} hand in frozen template")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "spec_sha256": self.spec_sha256,
            "builder_version": self.builder_version,
            "grip_build_builder_version": self.grip_build_builder_version,
            "handedness": self.handedness,
            "style": self.style,
            "status": self.status,
            "hands": [hand.to_dict() for hand in self.hands],
            "contact_clearances_m": [list(row) for row in self.contact_clearances_m],
            "active_contact_pairs": list(self.active_contact_pairs),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FrozenGripTemplate:
        if not isinstance(raw, dict):
            raise FrozenTemplateError("frozen template must be a JSON object")
        hands_raw = raw.get("hands")
        if not isinstance(hands_raw, list):
            raise FrozenTemplateError("frozen template hands must be a list")
        clearances_raw = raw.get("contact_clearances_m")
        if not isinstance(clearances_raw, list):
            raise FrozenTemplateError(
                "frozen template contact_clearances_m must be a list"
            )
        pairs_raw = raw.get("active_contact_pairs")
        if not isinstance(pairs_raw, list):
            raise FrozenTemplateError(
                "frozen template active_contact_pairs must be a list"
            )
        return cls(
            schema=str(raw.get("schema", "")),
            schema_version=int(raw.get("schema_version", -1)),
            spec_sha256=str(raw.get("spec_sha256", "")),
            builder_version=str(raw.get("builder_version", "")),
            grip_build_builder_version=str(raw.get("grip_build_builder_version", "")),
            handedness=str(raw.get("handedness", "")),
            style=str(raw.get("style", "")),
            status=str(raw.get("status", "")),
            hands=tuple(HandTemplate.from_dict(h) for h in hands_raw),
            contact_clearances_m=tuple(
                (str(row[0]), float(row[1])) for row in clearances_raw
            ),
            active_contact_pairs=tuple(str(p) for p in pairs_raw),
        )

    @classmethod
    def from_json(cls, blob: str) -> FrozenGripTemplate:
        return cls.from_dict(json.loads(blob))


def _vec3(raw: Any, name: str, owner: str) -> tuple[float, float, float]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise FrozenTemplateError(
            f"{owner} capsule {name} must be 3 numbers, got {raw!r}"
        )
    out = tuple(float(v) for v in raw)
    if not all(np.isfinite(v) for v in out):
        raise FrozenTemplateError(f"{owner} capsule {name} must be finite, got {raw!r}")
    return (out[0], out[1], out[2])  # type: ignore[return-value]


def _role_capsules(hand: HandTemplate, chain: HandChain) -> tuple[CapsulePose, ...]:
    capsules: list[CapsulePose] = []
    for finger, seg_idx, p0, p1, radius in chain.all_capsules():
        capsules.append(
            CapsulePose(
                owner=finger,
                segment=int(seg_idx),
                p0=_as_vec(p0),
                p1=_as_vec(p1),
                radius_m=float(radius),
            )
        )
    thumb_angles = ThumbJointAngles(
        cmc_rad=hand.thumb[0], mcp_rad=hand.thumb[1], ip_rad=hand.thumb[2]
    )
    for seg_idx, (p0, p1, radius) in enumerate(
        thumb_capsules(
            chain.mesh_params,
            hand.thumb_azimuth_rad,
            thumb_angles,
            grip_radius_m=hand.grip_radius_m,
            palm_wrap_rad=hand.palm_wrap_rad,
            axial_shift_m=hand.axial_shift_m,
        )
    ):
        capsules.append(
            CapsulePose(
                owner="thumb",
                segment=seg_idx,
                p0=_as_vec(p0),
                p1=_as_vec(p1),
                radius_m=float(radius),
            )
        )
    return tuple(capsules)


def _as_vec(point: Any) -> tuple[float, float, float]:
    arr = np.asarray(point, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise FrozenTemplateError(
            f"capsule endpoint must be a finite vec3, got {point!r}"
        )
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _rebuild_chain(hand: HandTemplate) -> HandChain:
    """Rebuild the GB-2/GB-3 hand chain that produced *hand* (frozen data only)."""
    chain = HandChain(
        hand_size=HandSize(length_m=hand.hand_length_m),
        curl_sign=hand.curl_sign,
        dip_over_pip=dict(hand.dip_over_pip),
        grip_radius_m=hand.grip_radius_m,
        axial_shift_m=hand.axial_shift_m,
    )
    chain.flat_lateral_offset_m = hand.flat_lateral_offset_m
    for finger in LEAD_FINGER_CURL_ORDER:
        chain.joint_angles[finger] = FingerJointAngles()
    angles_by_name = {row[0]: row for row in hand.joints}
    for finger in LEAD_FINGER_CURL_ORDER:
        for joint, slot in (("mcp", 1), ("pip", 2), ("dip", 3)):
            row = angles_by_name.get(f"{finger}_{joint}")
            if row is not None:
                setattr(chain.joint_angles[finger], f"{joint}_rad", float(row[slot]))
    chain.palm_wrap_rad = hand.palm_wrap_rad
    chain.thumb_azimuth_rad = hand.thumb_azimuth_rad
    chain.thumb = ThumbJointAngles(
        cmc_rad=hand.thumb[0], mcp_rad=hand.thumb[1], ip_rad=hand.thumb[2]
    )
    return chain


def _hand_template(
    spec: GripSpec,
    role: str,
    table: dict[str, float],
) -> HandTemplate:
    hand_spec = spec.lead if role == "lead" else spec.trail
    curl_sign = 1 if role == "lead" else (-1 if str(spec.handedness) == "right" else 1)
    coupling = dict(hand_spec.joint_coupling_overrides)
    joints = tuple(
        (
            name,
            float(table[f"{name}_mcp"]),
            float(table[f"{name}_pip"]),
            float(table[f"{name}_dip"]),
        )
        for name in LEAD_FINGER_CURL_ORDER
    )
    thumb = (
        float(table["thumb_cmc"]),
        float(table["thumb_mcp"]),
        float(table["thumb_ip"]),
    )
    thumb_azimuth_rad = float(table["thumb_azimuth_rad"])
    palm_wrap_rad = float(table["palm_wrap"])
    grip_radius_m = float(table["grip_radius_m"])
    axial_shift_m = float(table["axial_shift_m"])
    hand = HandTemplate(
        role=role,
        curl_sign=curl_sign,
        flat_lateral_offset_m=(hand_spec.palm_finger_depth - 0.5) * 0.012,
        hand_length_m=float(spec.hand_size.length_m),
        joints=joints,
        thumb=thumb,
        thumb_azimuth_rad=thumb_azimuth_rad,
        palm_wrap_rad=palm_wrap_rad,
        grip_radius_m=grip_radius_m,
        axial_shift_m=axial_shift_m,
        dip_over_pip=tuple(
            (finger, float(coupling.get(finger, _DEFAULT_DIP_OVER_PIP)))
            for finger in LEAD_FINGER_CURL_ORDER
        ),
        capsules=(),
    )
    chain = _rebuild_chain(hand)
    return replace(hand, capsules=_role_capsules(hand, chain))


def build_frozen_template(
    spec: GripSpec,
    *,
    result: FullGripBuildResult | None = None,
) -> FrozenGripTemplate:
    """Capture a built grip as frozen data stamped with its provenance.

    Preconditions: ``spec`` is a :class:`GripSpec`; ``result``, when given, was built
    from the same spec. Postconditions: the template carries both hands, a non-empty
    capsule set per hand, and a stamp that matches ``spec.spec_sha256`` and
    :data:`TEMPLATE_BUILDER_VERSION`.
    """
    if not isinstance(spec, GripSpec):
        raise TypeError(f"spec must be GripSpec, got {type(spec).__name__}")
    build = result if result is not None else build_full_grip(spec)
    hands = tuple(
        _hand_template(spec, role, build.joint_angles_rad[role])
        for role in TEMPLATE_HAND_ROLES
    )
    return FrozenGripTemplate(
        schema=FROZEN_TEMPLATE_SCHEMA_ID,
        schema_version=FROZEN_TEMPLATE_SCHEMA_VERSION,
        spec_sha256=build.receipt.spec_sha256,
        builder_version=TEMPLATE_BUILDER_VERSION,
        grip_build_builder_version=GB3_BUILDER_VERSION,
        handedness=build.receipt.handedness,
        style=str(spec.style.value),
        status=build.status.value
        if isinstance(build.status, BuildStatus)
        else str(build.status),
        hands=hands,
        contact_clearances_m=tuple(
            sorted((str(k), float(v)) for k, v in build.contact_clearances_m.items())
        ),
        active_contact_pairs=tuple(sorted(build.active_contact_pairs)),
    )


def write_frozen_template(template: FrozenGripTemplate, path: Path | str) -> Path:
    """Atomically write *template* as JSON and return the written path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(template.to_json(), encoding="utf-8")
    tmp.replace(target)
    return target


def load_frozen_template(
    path: Path | str,
    *,
    expected_spec_sha256: str | None = None,
    expected_builder_version: str | None = None,
) -> FrozenGripTemplate:
    """Load a frozen template, refusing any stamp the caller did not ask for.

    Fail closed: a missing file, unreadable JSON, an unknown schema, or a spec-hash or
    builder-version mismatch raises before any capsule data is handed back.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"frozen grip template not found: {source}")
    template = FrozenGripTemplate.from_json(source.read_text(encoding="utf-8"))
    verify_template_stamp(
        template,
        expected_spec_sha256=expected_spec_sha256,
        expected_builder_version=expected_builder_version,
    )
    return template


def verify_template_stamp(
    template: FrozenGripTemplate,
    *,
    expected_spec_sha256: str | None = None,
    expected_builder_version: str | None = None,
) -> None:
    """Raise :class:`FrozenTemplateStampError` when *template* is stale for the caller."""
    if (
        expected_spec_sha256 is not None
        and template.spec_sha256 != expected_spec_sha256
    ):
        raise FrozenTemplateStampError(
            f"frozen grip template spec_sha256 {template.spec_sha256[:12]} does not match "
            f"the requested spec {expected_spec_sha256[:12]}; rebuild the template"
        )
    if (
        expected_builder_version is not None
        and template.builder_version != expected_builder_version
    ):
        raise FrozenTemplateStampError(
            f"frozen grip template builder_version {template.builder_version!r} does not match "
            f"the requested {expected_builder_version!r}; rebuild the template"
        )


__all__ = [
    "FROZEN_TEMPLATE_SCHEMA_ID",
    "FROZEN_TEMPLATE_SCHEMA_VERSION",
    "TEMPLATE_BUILDER_VERSION",
    "TEMPLATE_HAND_ROLES",
    "CapsulePose",
    "FrozenGripTemplate",
    "FrozenTemplateError",
    "FrozenTemplateStampError",
    "HandTemplate",
    "build_frozen_template",
    "load_frozen_template",
    "verify_template_stamp",
    "write_frozen_template",
]

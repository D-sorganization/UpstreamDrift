"""Shared model grip binding contract and contact evaluator (HRG-GS-M1, #13323).

Connects any hand-bearing model across engines to the canonical
:class:`~src.shared.python.grip_contact.contact_topology.ContactTopology`,
the G6 :func:`~src.shared.python.grip_contact.topology_contact.topology_side_wrench`
physical evaluator, and the G3
:func:`~src.shared.python.grip_contact.hand_hand_allocation.allocate_hand_hand_from_closure`
allocated hand-hand reporter.

Pure NumPy module with no physics-engine imports.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any, Final, Literal, Protocol, runtime_checkable

import numpy as np

from src.shared.python.core.handedness import (
    DEFAULT_HANDEDNESS,
    Handedness,
    lead_side,
    trail_side,
    validate_handedness,
)
from src.shared.python.grip_contact.bushing_law import BushingState, cross3
from src.shared.python.grip_contact.contact_topology import (
    ALLOCATED_HAND_GRIP_MODES,
    GRIP,
    PHYSICAL_HAND_GRIP_MODES,
    PHYSICAL_HAND_HAND_MODES,
    ContactPair,
    ContactSite,
    ContactTopology,
)
from src.shared.python.grip_contact.hand_hand_allocation import (
    AllocatedHandContacts,
    AllocationPrior,
    allocate_hand_hand_from_closure,
)
from src.shared.python.grip_contact.pair_contact import PairWrench
from src.shared.python.grip_contact.prescribed_motion import RigidBodyState
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    TopologySideWrench,
    map_site_to_hand_frame,
    topology_side_wrench,
)
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)

__all__ = [
    "KNOWN_ATTACHMENT_MODES",
    "PAIR_LABELS",
    "ROTATION_TOLERANCE",
    "SITE_REPRESENTATIONS",
    "BodyStates",
    "GripCapabilities",
    "GripStepReport",
    "HandClubBinding",
    "ModelGripAdapter",
    "PairLabel",
    "SiteBinding",
    "SiteRepresentation",
    "apply_report",
    "bind_rigid_hand_sites",
    "evaluate_contacts",
    "site_transform_in_hand_frame",
    "validate_binding",
]

ROTATION_TOLERANCE: Final[float] = 1e-9
SiteRepresentation = Literal["rigid_proxy", "phalange"]
SITE_REPRESENTATIONS: Final[tuple[SiteRepresentation, ...]] = (
    "rigid_proxy",
    "phalange",
)
PairLabel = Literal["physical", "allocated"]
PAIR_LABELS: Final[tuple[PairLabel, ...]] = ("physical", "allocated")
KNOWN_ATTACHMENT_MODES: Final[tuple[str, ...]] = tuple(
    sorted(set(PHYSICAL_HAND_GRIP_MODES) | set(ALLOCATED_HAND_GRIP_MODES))
)
_CONSUMER: Final[str] = "grip_contact.binding"
_CHILD: Final[str] = "HRG-GS-M1"
_ISSUE: Final[int] = 13323
_NO_CLOSURE_REASON: Final[str] = (
    "hand-hand pair is allocated and no closure wrench was supplied"
)


def _require_non_empty_str(value: object, name: str) -> str:
    """Validate that ``value`` is a non-empty, non-whitespace string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_transform_4x4(matrix: object, name: str) -> np.ndarray:
    """Validate a 4x4 homogeneous transform with proper rotation within 1e-9."""
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (4, 4) or not bool(np.isfinite(arr).all()):
        raise ValueError(f"{name} must be a finite 4x4 homogeneous transform")
    bottom = arr[3, :]
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(bottom, expected_bottom, atol=ROTATION_TOLERANCE):
        raise ValueError(f"{name} must have bottom row [0, 0, 0, 1] within 1e-9")
    rot = arr[:3, :3]
    ortho_ok = np.allclose(rot.T @ rot, np.eye(3), atol=ROTATION_TOLERANCE)
    det_val = float(
        rot[0, 0] * (rot[1, 1] * rot[2, 2] - rot[1, 2] * rot[2, 1])
        - rot[0, 1] * (rot[1, 0] * rot[2, 2] - rot[1, 2] * rot[2, 0])
        + rot[0, 2] * (rot[1, 0] * rot[2, 1] - rot[1, 1] * rot[2, 0])
    )
    det_ok = math.isclose(det_val, 1.0, abs_tol=ROTATION_TOLERANCE)
    if not (ortho_ok and det_ok):
        raise ValueError(f"{name} must contain a proper rotation matrix within 1e-9")
    out = arr.copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class SiteBinding:
    """Rigid attachment of one ContactTopology site to a model body.

    Preconditions:
        ``site_id`` and ``body_id`` are non-empty strings;
        ``body_to_site`` is a 4x4 homogeneous transform with proper rotation
        within 1e-9 and bottom row ``[0, 0, 0, 1]``;
        ``representation`` is ``"rigid_proxy"`` or ``"phalange"``.
    """

    site_id: str
    body_id: str
    body_to_site: np.ndarray
    representation: SiteRepresentation = "rigid_proxy"

    def __post_init__(self) -> None:
        _require_non_empty_str(self.site_id, "site_id")
        _require_non_empty_str(self.body_id, "body_id")
        if self.representation not in SITE_REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {SITE_REPRESENTATIONS}, "
                f"got {self.representation!r}"
            )
        mat = _validate_transform_4x4(self.body_to_site, "body_to_site")
        object.__setattr__(self, "body_to_site", mat)

    @property
    def rotation(self) -> np.ndarray:
        """3x3 proper rotation block of ``body_to_site``."""
        return self.body_to_site[:3, :3]

    @property
    def translation_m(self) -> np.ndarray:
        """3-vector translation block (m) of ``body_to_site``."""
        return self.body_to_site[:3, 3]


@dataclass(frozen=True)
class HandClubBinding:
    """Complete binding between a ContactTopology and a specific model asset.

    Preconditions:
        Non-empty string identifiers; valid ``handedness``; unique ``site_id``
        values in ``sites``; non-empty reasons in ``unavailable_sites``; and
        no site present in both ``sites`` and ``unavailable_sites``.
    """

    model_id: str
    model_hash: str
    handedness: Handedness
    attachment_mode: str
    club_body: str
    lead_hand_body: str
    trail_hand_body: str
    sites: tuple[SiteBinding, ...] = ()
    unavailable_sites: Mapping[str, str] = field(default_factory=dict)
    club_grip_frames: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.model_id, "model_id")
        _require_non_empty_str(self.model_hash, "model_hash")
        validate_handedness(self.handedness)
        _require_non_empty_str(self.attachment_mode, "attachment_mode")
        _require_non_empty_str(self.club_body, "club_body")
        _require_non_empty_str(self.lead_hand_body, "lead_hand_body")
        _require_non_empty_str(self.trail_hand_body, "trail_hand_body")

        site_tuple = tuple(self.sites)
        seen_ids: set[str] = set()
        for sb in site_tuple:
            if not isinstance(sb, SiteBinding):
                raise TypeError(
                    f"sites must contain SiteBinding, got {type(sb).__name__}"
                )
            if sb.site_id in seen_ids:
                raise ValueError(
                    f"duplicate site_id in sites must be unique: {sb.site_id!r}"
                )
            seen_ids.add(sb.site_id)
        object.__setattr__(self, "sites", site_tuple)

        if not isinstance(self.unavailable_sites, Mapping):
            raise TypeError("unavailable_sites must be a Mapping[str, str]")
        unavail_clean: dict[str, str] = {}
        for k, v in self.unavailable_sites.items():
            key = _require_non_empty_str(k, "unavailable_sites key")
            reason = _require_non_empty_str(v, f"unavailable_sites[{k!r}] reason")
            unavail_clean[key] = reason

        overlap = seen_ids & set(unavail_clean.keys())
        if overlap:
            raise ValueError(
                f"sites cannot be both bound and unavailable: {sorted(overlap)}"
            )
        object.__setattr__(self, "unavailable_sites", MappingProxyType(unavail_clean))

        if not isinstance(self.club_grip_frames, Mapping):
            raise TypeError("club_grip_frames must be a Mapping[str, np.ndarray]")
        grip_clean: dict[str, np.ndarray] = {}
        for side_key, mat in self.club_grip_frames.items():
            if side_key not in ("L", "R"):
                raise ValueError(
                    f"club_grip_frames key must be 'L' or 'R', got {side_key!r}"
                )
            grip_clean[side_key] = _validate_transform_4x4(
                mat, f"club_grip_frames[{side_key!r}]"
            )
        object.__setattr__(self, "club_grip_frames", MappingProxyType(grip_clean))

    def site_binding_map(self) -> dict[str, SiteBinding]:
        """Return mapping from ``site_id`` to :class:`SiteBinding`."""
        return {sb.site_id: sb for sb in self.sites}

    def required_bodies(self) -> tuple[str, ...]:
        """Ordered unique body IDs referenced by this binding."""
        bodies: list[str] = [
            self.club_body,
            self.lead_hand_body,
            self.trail_hand_body,
        ]
        for sb in self.sites:
            if sb.body_id not in bodies:
                bodies.append(sb.body_id)
        return tuple(bodies)

    def hand_body_for_role(self, role: str) -> str:
        """Return ``lead_hand_body`` or ``trail_hand_body`` for ``role``."""
        if role == "lead":
            return self.lead_hand_body
        if role == "trail":
            return self.trail_hand_body
        raise ValueError(f"role must be 'lead' or 'trail', got {role!r}")


@dataclass(frozen=True)
class GripCapabilities:
    """Declared grip contact capabilities of a model adapter."""

    attachment_modes: tuple[str, ...]
    supports_physical_pairs: bool
    supports_wrench_application: bool
    native_contact: bool = False
    available_bodies: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.attachment_modes, Collection) or isinstance(
            self.attachment_modes, str
        ):
            raise TypeError("attachment_modes must be a collection of strings")
        modes = tuple(
            _require_non_empty_str(m, "attachment_mode") for m in self.attachment_modes
        )
        if not modes:
            raise ValueError("attachment_modes must be non-empty")
        object.__setattr__(self, "attachment_modes", modes)

        for field_name in (
            "supports_physical_pairs",
            "supports_wrench_application",
            "native_contact",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, bool):
                raise TypeError(
                    f"{field_name} must be a bool, got {type(val).__name__}"
                )

        if self.available_bodies is not None:
            bodies = tuple(
                _require_non_empty_str(b, "available_body")
                for b in self.available_bodies
            )
            object.__setattr__(self, "available_bodies", bodies)


class BodyStates(dict[str, np.ndarray]):
    """Mapping from ``body_id`` to 4x4 world pose with optional spatial velocities."""

    def __init__(
        self,
        poses: Mapping[str, np.ndarray],
        velocities: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        validated: dict[str, np.ndarray] = {}
        for body_id, pose in poses.items():
            key = _require_non_empty_str(body_id, "body_id")
            validated[key] = _validate_transform_4x4(pose, f"states[{key!r}]")
        super().__init__(validated)
        self._velocities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if velocities is not None:
            for body_id, (lin, ang) in velocities.items():
                v_arr = np.asarray(lin, dtype=float)
                w_arr = np.asarray(ang, dtype=float)
                if v_arr.shape != (3,) or not bool(np.isfinite(v_arr).all()):
                    raise ValueError(
                        f"linear velocity for {body_id!r} must be finite (3,)"
                    )
                if w_arr.shape != (3,) or not bool(np.isfinite(w_arr).all()):
                    raise ValueError(
                        f"angular velocity for {body_id!r} must be finite (3,)"
                    )
                self._velocities[body_id] = (v_arr.copy(), w_arr.copy())

    def velocity_of(self, body_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(velocity_m_s, omega_rad_s)`` in world axes for ``body_id``."""
        if body_id in self._velocities:
            v_lin, v_ang = self._velocities[body_id]
            return v_lin.copy(), v_ang.copy()
        return np.zeros(3), np.zeros(3)


@runtime_checkable
class ModelGripAdapter(Protocol):
    """Protocol for binding a physics model to the shared grip contact law."""

    def capabilities(self) -> GripCapabilities:
        """Return the adapter's declared grip contact capabilities."""
        ...

    def resolve_binding(self, topology: ContactTopology) -> HandClubBinding:
        """Resolve a :class:`HandClubBinding` for ``topology``."""
        ...

    def sample_states(self, binding: HandClubBinding) -> BodyStates:
        """Sample current 4x4 world poses (and optional velocities) for bound bodies."""
        ...

    def apply_wrenches(
        self, wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Apply ``(force_n, moment_about_body_origin_nm)`` wrenches in world axes."""
        ...


def site_transform_in_hand_frame(
    topology: ContactTopology,
    site: ContactSite | str,
    side: str | None = None,
) -> np.ndarray:
    """Return the 4x4 transform ``^{F_hand} T_site`` in the hand grip frame."""
    target = topology.site(site) if isinstance(site, str) else site
    side_char = side if side is not None else ("L" if target.hand == "lead" else "R")
    mapped = map_site_to_hand_frame(topology, target, side_char)
    mat = np.eye(4)
    mat[:3, 3] = np.asarray(mapped.centre_m, dtype=float)
    mat.setflags(write=False)
    return mat


def bind_rigid_hand_sites(
    topology: ContactTopology,
    *,
    lead_hand_body: str = "hand_l",
    trail_hand_body: str = "hand_r",
    lead_hand_to_grip: np.ndarray | None = None,
    trail_hand_to_grip: np.ndarray | None = None,
) -> tuple[SiteBinding, ...]:
    """Build ``rigid_proxy`` :class:`SiteBinding` tuples for all sites in ``topology``."""
    if not isinstance(topology, ContactTopology):
        raise TypeError(
            f"topology must be ContactTopology, got {type(topology).__name__}"
        )
    t_lead = (
        np.eye(4)
        if lead_hand_to_grip is None
        else _validate_transform_4x4(lead_hand_to_grip, "lead_hand_to_grip")
    )
    t_trail = (
        np.eye(4)
        if trail_hand_to_grip is None
        else _validate_transform_4x4(trail_hand_to_grip, "trail_hand_to_grip")
    )
    bindings: list[SiteBinding] = []
    for site in topology.sites:
        is_lead = site.hand == "lead"
        body_id = lead_hand_body if is_lead else trail_hand_body
        base_t = t_lead if is_lead else t_trail
        side_char = "L" if is_lead else "R"
        site_t = site_transform_in_hand_frame(topology, site, side_char)
        bindings.append(
            SiteBinding(
                site_id=site.name,
                body_id=body_id,
                body_to_site=base_t @ site_t,
                representation="rigid_proxy",
            )
        )
    return tuple(bindings)


def _raise_unavailable(model_id: str, feature: str) -> None:
    raise SchemaFeatureUnavailableError(
        model_id,
        consumer=_CONSUMER,
        feature=feature,
        child=_CHILD,
        issue=_ISSUE,
    )


def validate_binding(
    topology: ContactTopology,
    binding: HandClubBinding,
    capabilities: GripCapabilities,
    *,
    available_bodies: Collection[str] | None = None,
) -> None:
    """Validate ``binding`` against ``topology`` and ``capabilities`` before stepping.

    Raises:
        SchemaFeatureUnavailableError: when a topology physical site has no
            binding and is not listed unavailable, a bound body is missing from
            the model, or the attachment mode is unsupported.
    """
    if not isinstance(topology, ContactTopology):
        raise TypeError(
            f"topology must be ContactTopology, got {type(topology).__name__}"
        )
    if not isinstance(binding, HandClubBinding):
        raise TypeError(
            f"binding must be HandClubBinding, got {type(binding).__name__}"
        )
    if not isinstance(capabilities, GripCapabilities):
        raise TypeError(
            f"capabilities must be GripCapabilities, got {type(capabilities).__name__}"
        )

    mode = binding.attachment_mode
    model_id = binding.model_id
    if mode not in KNOWN_ATTACHMENT_MODES:
        _raise_unavailable(model_id, f"unknown attachment_mode {mode!r}")
    if mode not in capabilities.attachment_modes:
        _raise_unavailable(
            model_id,
            f"attachment_mode {mode!r} unsupported by capabilities {capabilities.attachment_modes}",
        )
    if mode in PHYSICAL_HAND_GRIP_MODES and not capabilities.supports_physical_pairs:
        _raise_unavailable(
            model_id,
            f"physical pairs in attachment_mode {mode!r}",
        )
    if capabilities.native_contact and capabilities.supports_wrench_application:
        _raise_unavailable(
            model_id,
            "simultaneous native_contact and supports_wrench_application (duplicate contact forces)",
        )
    if mode in PHYSICAL_HAND_HAND_MODES and (
        binding.lead_hand_body == binding.trail_hand_body
        or binding.lead_hand_body == binding.club_body
        or binding.trail_hand_body == binding.club_body
    ):
        _raise_unavailable(
            model_id,
            f"physical hand-hand mode {mode!r} requires separate club, lead, and trail bodies",
        )

    known_bodies: set[str] | None = None
    if available_bodies is not None:
        known_bodies = set(available_bodies)
    elif capabilities.available_bodies is not None:
        known_bodies = set(capabilities.available_bodies)

    if known_bodies is not None:
        for body_id in binding.required_bodies():
            if body_id not in known_bodies:
                _raise_unavailable(model_id, f"missing required model body {body_id!r}")

    site_map = binding.site_binding_map()
    topo_site_names = set(topology.site_names())
    for site_id, sb in site_map.items():
        if site_id not in topo_site_names:
            _raise_unavailable(model_id, f"unknown topology site {site_id!r}")
        if sb.representation == "rigid_proxy":
            topo_site = topology.site(site_id)
            expected_body = binding.hand_body_for_role(topo_site.hand)
            if sb.body_id != expected_body:
                _raise_unavailable(
                    model_id,
                    f"rigid_proxy site {site_id!r} bound to {sb.body_id!r} instead of {expected_body!r}",
                )

    for pair in topology.pairs:
        if mode not in pair.physical_modes:
            continue
        partner_sites = (
            (pair.site_a,) if pair.site_b == GRIP else (pair.site_a, pair.site_b)
        )
        for s_name in partner_sites:
            if s_name not in site_map and s_name not in binding.unavailable_sites:
                _raise_unavailable(
                    model_id,
                    f"physical site {s_name!r} (pair {pair.name!r}) has no binding and is not listed unavailable",
                )


@dataclass(frozen=True)
class GripStepReport:
    """Pure per-step contact evaluation report (HRG-GS-M1, #13323)."""

    normal_force_n: Mapping[str, float]
    tangential_force_n: Mapping[str, float]
    spin_moment_nm: Mapping[str, float]
    pair_labels: Mapping[str, PairLabel]
    unavailable_reasons: Mapping[str, str]
    pair_wrenches: Mapping[str, PairWrench]
    side_wrenches: Mapping[str, TopologySideWrench] = field(default_factory=dict)
    allocated_contacts: AllocatedHandContacts | None = None
    _body_wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False
    )

    @property
    def labels(self) -> Mapping[str, PairLabel]:
        """Alias for :attr:`pair_labels`."""
        return self.pair_labels

    @property
    def normal(self) -> Mapping[str, float]:
        """Alias for :attr:`normal_force_n`."""
        return self.normal_force_n

    @property
    def tangential(self) -> Mapping[str, float]:
        """Alias for :attr:`tangential_force_n`."""
        return self.tangential_force_n

    @property
    def spin(self) -> Mapping[str, float]:
        """Alias for :attr:`spin_moment_nm`."""
        return self.spin_moment_nm

    def body_wrenches(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Equal and opposite wrenches ``(force_n, moment_about_body_origin_nm)``
        for bound hand and club bodies from PHYSICAL pairs only.
        Allocated pairs are never included.
        """
        return {
            body_id: (force.copy(), moment.copy())
            for body_id, (force, moment) in self._body_wrenches.items()
        }

    def action_reaction_residual_n(self) -> float:
        """Euclidean norm (N) of the sum of all physical body forces."""
        net = np.zeros(3)
        for force, _ in self._body_wrenches.values():
            net = net + force
        return float(np.linalg.norm(net))


def _extract_rigid_state(states: Mapping[str, Any], body_id: str) -> RigidBodyState:
    """Convert ``states[body_id]`` into a validated :class:`RigidBodyState`."""
    raw = states[body_id]
    if isinstance(raw, RigidBodyState):
        return raw
    if isinstance(raw, BushingState):
        return RigidBodyState(
            rotation=raw.rotation,
            position_m=raw.position_m,
            velocity_m_s=raw.velocity_m_s,
            omega_rad_s=raw.omega_rad_s,
        )
    pose = _validate_transform_4x4(raw, f"states[{body_id!r}]")
    vel_lin = np.zeros(3)
    vel_ang = np.zeros(3)
    if isinstance(states, BodyStates):
        vel_lin, vel_ang = states.velocity_of(body_id)
    return RigidBodyState(
        rotation=pose[:3, :3],
        position_m=pose[:3, 3],
        velocity_m_s=vel_lin,
        omega_rad_s=vel_ang,
    )


def _implied_hand_frame_for_site(
    topology: ContactTopology,
    site: ContactSite,
    side_char: str,
    sb: SiteBinding,
    body_state: RigidBodyState,
) -> BushingState:
    """Recover the world :class:`BushingState` of the hand grip frame for ``site``."""
    mapped = map_site_to_hand_frame(topology, site, side_char)
    p_hand = np.asarray(mapped.centre_m, dtype=float)
    rot_bs = sb.rotation
    trans_bs = sb.translation_m - rot_bs @ p_hand
    offset = np.eye(4)
    offset[:3, :3] = rot_bs
    offset[:3, 3] = trans_bs
    return body_state.frame(offset)


def _spin_magnitude_nm(pw: PairWrench) -> float:
    """Extract spin friction moment magnitude (N*m) from :class:`PairWrench`."""
    arm = pw.contact_point_m - pw.point_a_m
    spin_vec = pw.moment_a_nm - cross3(arm, pw.force_a_n)
    return float(np.linalg.norm(spin_vec))


def _evaluate_physical_side(
    topology_model: TopologyContactModel,
    binding: HandClubBinding,
    body_states: Mapping[str, RigidBodyState],
    club_frame: BushingState,
    role: str,
    side_char: str,
) -> tuple[TopologySideWrench, dict[str, PairWrench]]:
    """Evaluate physical hand-grip pairs for one hand via ``topology_side_wrench``."""
    topology = topology_model.topology
    site_map = binding.site_binding_map()
    hand_body = binding.hand_body_for_role(role)
    default_hand_state = body_states[hand_body]
    default_hand_frame = BushingState(
        default_hand_state.rotation,
        default_hand_state.position_m,
        default_hand_state.velocity_m_s,
        default_hand_state.omega_rad_s,
    )

    pairs = topology_model.physical_grip_pairs(side_char)
    bound_frames: dict[str, BushingState] = {}
    for pair in pairs:
        if pair.site_a in site_map:
            sb = site_map[pair.site_a]
            site = topology.site(pair.site_a)
            b_state = body_states[sb.body_id]
            bound_frames[pair.name] = _implied_hand_frame_for_site(
                topology, site, side_char, sb, b_state
            )

    if not bound_frames:
        empty_sw = topology_side_wrench(
            topology_model, side_char, default_hand_frame, club_frame
        )
        return empty_sw, {}

    first_frame = next(iter(bound_frames.values()))
    uniform = all(
        np.allclose(f.rotation, first_frame.rotation, atol=1e-12)
        and np.allclose(f.position_m, first_frame.position_m, atol=1e-12)
        and np.allclose(f.velocity_m_s, first_frame.velocity_m_s, atol=1e-12)
        and np.allclose(f.omega_rad_s, first_frame.omega_rad_s, atol=1e-12)
        for f in bound_frames.values()
    )
    if uniform:
        sw = topology_side_wrench(
            topology_model,
            side_char,
            first_frame,
            club_frame,
            mode=binding.attachment_mode,
        )
        active = {
            name: pw for name, pw in sw.pair_wrenches.items() if name in bound_frames
        }
        return sw, active

    active_wrenches: dict[str, PairWrench] = {}
    force_sum = np.zeros(3)
    moment_sum = np.zeros(3)
    norm_list: list[float] = []
    tang_list: list[float] = []
    for pair in pairs:
        if pair.name not in bound_frames:
            continue
        sw_i = topology_side_wrench(
            topology_model,
            side_char,
            bound_frames[pair.name],
            club_frame,
            mode=binding.attachment_mode,
        )
        if pair.name in sw_i.pair_wrenches:
            pw = sw_i.pair_wrenches[pair.name]
            active_wrenches[pair.name] = pw
            force_sum = force_sum + pw.force_b_n
            moment_sum = moment_sum + pw.moment_b_nm
            norm_list.append(pw.normal_force_mag_n)
            tang_list.append(pw.tangential_force_mag_n)

    norm_arr = np.asarray(norm_list, dtype=float)
    tang_arr = np.asarray(tang_list, dtype=float)
    combined_sw = TopologySideWrench(
        force_n=force_sum,
        moment_nm=moment_sum,
        pair_wrenches=active_wrenches,
        normal_force_n=norm_arr,
        tangential_force_n=tang_arr,
        total_normal_force_n=float(np.sum(norm_arr)) if len(norm_arr) > 0 else 0.0,
    )
    return combined_sw, active_wrenches


def _assemble_body_wrenches(
    topology: ContactTopology,
    binding: HandClubBinding,
    body_states: Mapping[str, RigidBodyState],
    physical_pair_wrenches: Mapping[str, PairWrench],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Assemble equal-and-opposite body wrenches from physical pairs only."""
    site_map = binding.site_binding_map()
    forces: dict[str, np.ndarray] = {b: np.zeros(3) for b in binding.required_bodies()}
    moments: dict[str, np.ndarray] = {b: np.zeros(3) for b in binding.required_bodies()}
    club_body = binding.club_body
    p_club = body_states[club_body].position_m

    for pair_name, pw in physical_pair_wrenches.items():
        pair = topology.pair(pair_name)
        sb_a = site_map[pair.site_a]
        body_a = sb_a.body_id
        p_body_a = body_states[body_a].position_m

        forces[body_a] = forces[body_a] + pw.force_a_n
        moments[body_a] = (
            moments[body_a]
            + pw.moment_a_nm
            + cross3(pw.point_a_m - p_body_a, pw.force_a_n)
        )

        forces[club_body] = forces[club_body] + pw.force_b_n
        moments[club_body] = (
            moments[club_body]
            + pw.moment_b_nm
            + cross3(pw.point_b_m - p_club, pw.force_b_n)
        )

    return {b: (forces[b], moments[b]) for b in forces}


def _record_hand_grip_channels(
    topology: ContactTopology,
    binding: HandClubBinding,
    physical_pair_wrenches: Mapping[str, PairWrench],
    channels: tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, PairLabel],
        dict[str, str],
    ],
) -> None:
    """Populate per-pair channels and labels for all hand-grip pairs."""
    (
        normal_force_n,
        tangential_force_n,
        spin_moment_nm,
        pair_labels,
        unavailable_reasons,
    ) = channels
    mode = binding.attachment_mode
    unavail = binding.unavailable_sites
    for pair in topology.hand_grip_pairs():
        is_phys = mode in pair.physical_modes
        pair_labels[pair.name] = "physical" if is_phys else "allocated"
        if pair.site_a in unavail:
            normal_force_n[pair.name] = float("nan")
            tangential_force_n[pair.name] = float("nan")
            spin_moment_nm[pair.name] = float("nan")
            unavailable_reasons[pair.name] = unavail[pair.site_a]
        elif is_phys and pair.name in physical_pair_wrenches:
            pw = physical_pair_wrenches[pair.name]
            normal_force_n[pair.name] = float(pw.normal_force_mag_n)
            tangential_force_n[pair.name] = float(pw.tangential_force_mag_n)
            spin_moment_nm[pair.name] = _spin_magnitude_nm(pw)
        else:
            normal_force_n[pair.name] = float("nan")
            tangential_force_n[pair.name] = float("nan")
            spin_moment_nm[pair.name] = float("nan")
            unavailable_reasons[pair.name] = (
                f"pair {pair.name!r} is not physical in attachment_mode {mode!r}"
            )


def _record_allocated_channels(
    topology_model: TopologyContactModel,
    binding: HandClubBinding,
    club_state: RigidBodyState,
    closure: Any | None,
    prior: AllocationPrior | None,
    channels: tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, PairLabel],
        dict[str, str],
    ],
) -> AllocatedHandContacts:
    """Evaluate and record allocated hand-hand pairs via ``hand_hand_allocation``."""
    normal_force_n, tangential_force_n, spin_moment_nm, pair_labels, reasons = channels
    closure_input = (
        closure
        if closure is not None
        else SimpleNamespace(available=False, reason=_NO_CLOSURE_REASON, wrenches=[])
    )
    topology = topology_model.topology
    law = topology_model.law
    allocated = allocate_hand_hand_from_closure(
        closure_input,
        club_rotation=club_state.rotation,
        club_origin=club_state.position_m,
        topology=topology,
        static_friction=law.static_friction,
        prior=prior,
    )
    alloc_pairs = allocated.pairs
    unavail = binding.unavailable_sites
    for pair in topology.hand_hand_pairs():
        pair_labels[pair.name] = "allocated"
        site_reason = unavail.get(pair.site_a) or unavail.get(pair.site_b)
        pf = alloc_pairs[pair.name]
        if site_reason is not None or not pf.available:
            normal_force_n[pair.name] = float("nan")
            tangential_force_n[pair.name] = float("nan")
            spin_moment_nm[pair.name] = float("nan")
            reasons[pair.name] = site_reason or pf.reason or _NO_CLOSURE_REASON
        else:
            normal_force_n[pair.name] = (
                float(pf.normal_force_n)
                if pf.normal_force_n is not None
                else float("nan")
            )
            tangential_force_n[pair.name] = (
                float(pf.tangential_force_n)
                if pf.tangential_force_n is not None
                else float("nan")
            )
            spin_moment_nm[pair.name] = (
                float(np.linalg.norm(pf.couple_nm))
                if pf.couple_nm is not None
                else float("nan")
            )
    return allocated


def _build_club_frames(
    club_state: RigidBodyState,
    binding: HandClubBinding,
) -> dict[str, BushingState]:
    """Build per-side club grip BushingState frames from club state and binding."""
    grip_frames = binding.club_grip_frames
    club_frames: dict[str, BushingState] = {}
    for side_char in ("L", "R"):
        if side_char in grip_frames:
            club_frames[side_char] = club_state.frame(grip_frames[side_char])
        else:
            club_frames[side_char] = BushingState(
                club_state.rotation,
                club_state.position_m,
                club_state.velocity_m_s,
                club_state.omega_rad_s,
            )
    return club_frames


def evaluate_contacts(
    topology_model: TopologyContactModel,
    binding: HandClubBinding,
    states: Mapping[str, Any],
    *,
    closure: Any | None = None,
    prior: AllocationPrior | None = None,
) -> GripStepReport:
    """Evaluate physical and allocated grip contacts for one step.

    Preconditions:
        ``topology_model`` is a TopologyContactModel; ``binding`` is valid for
        ``topology_model``; ``states`` contains valid poses for required bodies.
    Postconditions:
        Returns GripStepReport with equal-and-opposite physical wrenches
        (residual <= 1e-12 N) and NaN plus a reason for unavailable channels.
    """
    if not isinstance(topology_model, TopologyContactModel):
        raise TypeError("topology_model must be a TopologyContactModel")
    if not isinstance(states, Mapping):
        raise TypeError("states must be a Mapping")

    topology = topology_model.topology
    avail_bodies = tuple(states.keys())
    default_caps = GripCapabilities(
        attachment_modes=KNOWN_ATTACHMENT_MODES,
        supports_physical_pairs=True,
        supports_wrench_application=True,
        native_contact=False,
        available_bodies=avail_bodies,
    )
    validate_binding(topology, binding, default_caps, available_bodies=avail_bodies)

    body_states = {
        b: _extract_rigid_state(states, b) for b in binding.required_bodies()
    }
    club_state = body_states[binding.club_body]
    club_frames = _build_club_frames(club_state, binding)

    side_wrenches: dict[str, TopologySideWrench] = {}
    physical_pair_wrenches: dict[str, PairWrench] = {}
    for role, side_char in (("lead", "L"), ("trail", "R")):
        sw, active = _evaluate_physical_side(
            topology_model,
            binding,
            body_states,
            club_frames[side_char],
            role,
            side_char,
        )
        side_wrenches[side_char] = sw
        physical_pair_wrenches.update(active)

    normal_force_n: dict[str, float] = {}
    tangential_force_n: dict[str, float] = {}
    spin_moment_nm: dict[str, float] = {}
    pair_labels: dict[str, PairLabel] = {}
    unavailable_reasons: dict[str, str] = {}
    channels = (
        normal_force_n,
        tangential_force_n,
        spin_moment_nm,
        pair_labels,
        unavailable_reasons,
    )
    _record_hand_grip_channels(topology, binding, physical_pair_wrenches, channels)
    allocated = _record_allocated_channels(
        topology_model, binding, club_state, closure, prior, channels
    )
    body_wrenches = _assemble_body_wrenches(
        topology, binding, body_states, physical_pair_wrenches
    )

    return GripStepReport(
        normal_force_n=MappingProxyType(normal_force_n),
        tangential_force_n=MappingProxyType(tangential_force_n),
        spin_moment_nm=MappingProxyType(spin_moment_nm),
        pair_labels=MappingProxyType(pair_labels),
        unavailable_reasons=MappingProxyType(unavailable_reasons),
        pair_wrenches=MappingProxyType(physical_pair_wrenches),
        side_wrenches=MappingProxyType(side_wrenches),
        allocated_contacts=allocated,
        _body_wrenches=MappingProxyType(body_wrenches),
    )


def apply_report(adapter: ModelGripAdapter, report: GripStepReport) -> None:
    """Apply physical body wrenches from ``report`` to ``adapter``.

    Preconditions:
        ``adapter`` implements :class:`ModelGripAdapter`;
        ``report`` is a :class:`GripStepReport`;
        ``adapter.capabilities()`` supports wrench application and does not
        have ``native_contact=True``.
    Postconditions:
        Forwards ``report.body_wrenches()`` (physical pairs only) to
        ``adapter.apply_wrenches``.
    """
    if not isinstance(report, GripStepReport):
        raise TypeError(f"report must be GripStepReport, got {type(report).__name__}")
    if not isinstance(adapter, ModelGripAdapter):
        raise TypeError(
            f"adapter must satisfy ModelGripAdapter, got {type(adapter).__name__}"
        )
    caps = adapter.capabilities()
    if caps.native_contact:
        _raise_unavailable(
            "native_contact",
            "cannot apply shared contact wrenches when native_contact is enabled",
        )
    if not caps.supports_wrench_application:
        _raise_unavailable(
            "wrench_application",
            "adapter does not support external wrench application",
        )
    adapter.apply_wrenches(report.body_wrenches())

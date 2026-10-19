"""Simscape GolfSwing3D grip topology export and allocated reporting (HRG-GS-M11, #13333).

Provides:
1. Deterministic generation and validation of ``gs3dx_grip_topology.json``
   combining :mod:`~src.shared.python.grip_contact.contact_topology` sites/pairs,
   :class:`~src.shared.python.grip_contact.interface.GripInterface` hand frames,
   and G4 posed finger geometry from
   :func:`~src.shared.python.model_appearance.grip_hand_mesh.build_grip_hand_mesh`,
   while keeping ``gs3dx_appearance.v1.json`` byte-identical for v1 consumers.
2. Recorded-state force allocation from ``simscape-closed-loop-snapshot/v1``
   snapshots using existing G3
   (:func:`~src.shared.python.grip_contact.hand_hand_allocation.allocate_hand_hand_from_closure`),
   H1 (:func:`~src.shared.python.grip_contact.palm_thumb.allocate_from_closure`),
   and M1/G6 (:func:`~src.shared.python.grip_contact.binding.evaluate_contacts`)
   with strictly zero injected forces.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from src.shared.python.core.handedness import DEFAULT_HANDEDNESS, Handedness
from src.shared.python.grip_contact.binding import (
    BodyStates,
    GripCapabilities,
    GripStepReport,
    HandClubBinding,
    PairLabel,
    _spin_magnitude_nm,
    bind_rigid_hand_sites,
    evaluate_contacts,
    site_transform_in_hand_frame,
)
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_topology import (
    ContactPair,
    ContactSite,
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.hand_hand_allocation import (
    METHOD as ALLOCATION_LABEL,
    AllocatedHandContacts,
    AllocatedPairForce,
    AllocationPrior,
    allocate_hand_hand_from_closure,
    pair_contact_patch,
)
from src.shared.python.grip_contact.interface import GripFrame, GripInterface
from src.shared.python.grip_contact.palm_thumb import (
    PALM_THUMB_LABEL,
    PalmThumbForce,
    allocate_from_closure,
)
from src.shared.python.grip_contact.prescribed_motion import (
    RigidBodyState,
    hand_frame_states,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    map_site_to_hand_frame,
    topology_pair_wrench,
)
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    InputProvenance,
)
from src.shared.python.loop_nullspace.decomposition import (
    AllocationMetric,
    decompose,
)
from src.shared.python.loop_nullspace.simscape_export import (
    SimscapeExport,
    load_simscape_export,
    snapshot_from_export,
)
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)

if TYPE_CHECKING:
    from src.shared.python.model_appearance.grip_hand_mesh import (
        GripHandMesh,
        HandComponent,
    )
    from src.shared.python.model_appearance.grip_pose import (
        GripPose,
        Hand,
    )

__all__ = [
    "ALLOCATION_LABEL",
    "DEFAULT_GRIP_TOPOLOGY_JSON",
    "DEFAULT_SQUEEZE_PER_HAND_N",
    "GRIP_TOPOLOGY_SCHEMA_VERSION",
    "PALM_THUMB_LABEL",
    "SIMSCAPE_BINDING_MANIFEST_PATH",
    "SIMSCAPE_MODEL_HASH",
    "SIMSCAPE_MODEL_ID",
    "SimscapeAllocatedFrameReport",
    "SimscapeAllocatedSeriesReport",
    "SimscapeClosureWrench",
    "SimscapeGripReportingAdapter",
    "allocate_simscape_export",
    "allocate_simscape_snapshot",
    "build_grip_topology_payload",
    "closure_wrench_from_snapshot",
    "default_grip_interface",
    "export_grip_topology",
    "render_grip_topology_json",
    "validate_grip_topology_payload",
]

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[4]
GRIP_TOPOLOGY_SCHEMA_VERSION: Final[str] = "simscape-grip-topology/v1"
APPEARANCE_COMPATIBILITY_VERSION: Final[str] = "simscape-appearance/v1"
SIMSCAPE_MODEL_ID: Final[str] = "simscape_gs3dx_human"
SIMSCAPE_MODEL_HASH: Final[str] = (
    "sha256:9fb7f5897d2160c7402e4b0189cb490935bc49fd76dbdf286081e6f400ba6823"
)
DEFAULT_SQUEEZE_PER_HAND_N: Final[float] = 60.0
DEFAULT_FULL_BODY_SPEC_PATH: Final[Path] = (
    REPO_ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "full_body_spec_anthro_driver.json"
)

DEFAULT_GRIP_TOPOLOGY_JSON: Final[Path] = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "exploratory_gs3dx"
    / "appearance"
    / "gs3dx_grip_topology.json"
)
SIMSCAPE_BINDING_MANIFEST_PATH: Final[Path] = (
    REPO_ROOT / "src" / "config" / "grip_bindings" / "simscape_gs3dx_human.json"
)

_CLOSURE_ROW_EMBEDDING: Final[dict[ClosureKind, tuple[int, ...]]] = {
    ClosureKind.WELD6: (0, 1, 2, 3, 4, 5),
    ClosureKind.SHAFT_FREE5: (0, 1, 2, 4, 5),
    ClosureKind.BALL3: (0, 1, 2),
    ClosureKind.PLANAR_POINT2: (0, 1),
    ClosureKind.PLANAR_WELD3: (0, 1, 5),
}


def default_grip_interface() -> GripInterface:
    """Return the canonical GripInterface loaded from full_body_spec_anthro_driver.json."""
    spec_dict = json.loads(DEFAULT_FULL_BODY_SPEC_PATH.read_text(encoding="utf-8"))
    return GripInterface.from_spec(spec_dict)


def _round_val(val: float, digits: int = 8) -> float:
    rounded = round(float(val), digits)
    return 0.0 if rounded == 0.0 else rounded


def _round_seq(values: Any, digits: int = 8) -> list[float]:
    arr = np.asarray(values, dtype=float).ravel()
    return [_round_val(float(x), digits) for x in arr]


def _round_mat(mat: Any, digits: int = 8) -> list[list[float]]:
    arr = np.asarray(mat, dtype=float)
    return [_round_seq(row, digits) for row in arr]


def _serialize_grip_frame(frame: GripFrame) -> dict[str, Any]:
    rot = np.asarray(frame.rotation, dtype=float)
    return {
        "side": frame.side,
        "position_m": _round_seq(frame.position_m),
        "rotation": _round_mat(rot),
        "shaft_axis": _round_seq(rot[:, 0]),
        "transform_4x4": _round_mat(frame.matrix()),
    }


def _site_inward_normal(site: ContactSite) -> np.ndarray:
    """Unit radial inward normal toward the shaft (+x) axis for ``site``."""
    _, cy, cz = site.centre_m
    r_yz = math.hypot(cy, cz)
    if r_yz < 1e-12:
        return np.array([0.0, 0.0, -1.0], dtype=float)
    return np.array([0.0, -cy / r_yz, -cz / r_yz], dtype=float)


def _serialize_site(
    topology: ContactTopology,
    site: ContactSite,
    handedness: Handedness = DEFAULT_HANDEDNESS,
) -> dict[str, Any]:
    side_char = "L" if site.hand == "lead" else "R"
    mapped = map_site_to_hand_frame(topology, site, side_char, handedness=handedness)
    t_hand = site_transform_in_hand_frame(
        topology, site, side_char, handedness=handedness
    )
    inward = _site_inward_normal(site)
    return {
        "name": site.name,
        "hand": site.hand,
        "segment": site.segment,
        "primitive": site.primitive,
        "centre_m": _round_seq(site.centre_m),
        "axis_m": _round_seq(site.axis_m),
        "half_length_m": _round_val(site.half_length_m),
        "radius_m": _round_val(site.radius_m),
        "inward_normal": _round_seq(inward),
        "hand_frame_centre_m": _round_seq(mapped.centre_m),
        "hand_frame_transform_4x4": _round_mat(t_hand),
    }


def _serialize_pair(pair: ContactPair) -> dict[str, Any]:
    return {
        "id": pair.name,
        "name": pair.name,
        "site_a": pair.site_a,
        "site_b": pair.site_b,
        "kind": "hand_grip" if pair.is_hand_grip else "hand_hand",
        "physical_modes": list(pair.physical_modes),
        "allocated_modes": list(pair.allocated_modes),
        "simscape_label": "allocated",
    }


def _serialize_component(
    comp: HandComponent,
    rot_hg: np.ndarray,
    trans_hg: np.ndarray,
) -> dict[str, Any]:
    mesh = comp.mesh
    verts_hand = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    centroid_hand = verts_hand.mean(axis=0)
    verts_grip = (verts_hand @ rot_hg.T) + trans_hg
    centroid_grip = verts_grip.mean(axis=0)
    bounds_min = verts_grip.min(axis=0)
    bounds_max = verts_grip.max(axis=0)
    return {
        "name": comp.name,
        "finger": comp.finger,
        "on_grip": bool(comp.on_grip),
        "vertex_count": int(verts_hand.shape[0]),
        "face_count": int(faces.shape[0]),
        "centroid_hand_m": _round_seq(centroid_hand),
        "centroid_grip_m": _round_seq(centroid_grip),
        "bounds_min_grip_m": _round_seq(bounds_min),
        "bounds_max_grip_m": _round_seq(bounds_max),
    }


_HAND_TO_GRIP_ROTATION: Final[np.ndarray] = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
)


def _serialize_posed_hand(
    hand_enum: Hand,
    grip_radius_m: float,
    pose: GripPose,
) -> dict[str, Any]:
    from src.shared.python.model_appearance.grip_hand_mesh import (
        FINGER_NAMES,
        build_grip_hand_mesh,
    )

    hand_mesh: GripHandMesh = build_grip_hand_mesh(
        hand_enum, grip_radius_m=grip_radius_m, pose=pose
    )
    rot_hg = _HAND_TO_GRIP_ROTATION
    trans_hg = np.array([hand_mesh.distance_below_butt_m, 0.0, 0.0], dtype=float)
    components = [
        _serialize_component(comp, rot_hg, trans_hg) for comp in hand_mesh.components
    ]
    all_fingers = (*FINGER_NAMES, "thumb")
    fingertips = {
        f_name: _round_seq(hand_mesh.fingertip_in_grip_m(f_name))
        for f_name in all_fingers
    }
    return {
        "hand": hand_enum.value,
        "v_azimuth_deg": _round_val(hand_mesh.v_azimuth_deg),
        "thumb_azimuth_deg": _round_val(hand_mesh.thumb_azimuth_deg),
        "distance_below_butt_m": _round_val(hand_mesh.distance_below_butt_m),
        "min_pad_clearance_m": _round_val(hand_mesh.min_pad_clearance_m),
        "hand_to_grip_rotation": _round_mat(rot_hg),
        "hand_to_grip_translation_m": _round_seq(trans_hg),
        "fingertips_in_grip_m": fingertips,
        "components": components,
    }


def _build_style_entry(
    style: GripStyle,
    trigger_finger: bool,
    grip_radius_m: float,
) -> dict[str, Any]:
    from src.shared.python.model_appearance.grip_pose import GripPose, Hand
    from src.shared.python.motion_matching.grip_contract import GripSpec

    spec = GripSpec(style=style, trigger_finger=trigger_finger)
    topology = contact_topology(spec, grip_radius_m=grip_radius_m)
    pose = GripPose(style=style, trigger_finger=trigger_finger)
    sites = [_serialize_site(topology, s) for s in topology.sites]
    pairs = [_serialize_pair(p) for p in topology.pairs]
    hg_pairs = [p.name for p in topology.hand_grip_pairs()]
    hh_pairs = [p.name for p in topology.hand_hand_pairs()]
    posed_hands = {
        Hand.LEAD.value: _serialize_posed_hand(Hand.LEAD, grip_radius_m, pose),
        Hand.TRAIL.value: _serialize_posed_hand(Hand.TRAIL, grip_radius_m, pose),
    }
    return {
        "style": style.value,
        "trigger_finger": bool(trigger_finger),
        "grip_radius_m": _round_val(topology.grip_radius_m),
        "sites": sites,
        "pairs": pairs,
        "pair_ids": list(topology.pair_names()),
        "hand_grip_pair_ids": hg_pairs,
        "hand_hand_pair_ids": hh_pairs,
        "posed_hands": posed_hands,
    }


def build_grip_topology_payload() -> dict[str, Any]:
    """Build the deterministic ``simscape-grip-topology/v1`` export dictionary."""
    from src.shared.python.motion_matching.grip_contract import (
        NAMED_GRIP_PRESETS,
        GripSpec,
    )

    interface = default_grip_interface()
    default_topo = contact_topology(GripSpec(style=GripStyle.OVERLAP))
    grip_radius_m = default_topo.grip_radius_m

    styles_dict: dict[str, Any] = {}
    for style in (GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER):
        styles_dict[style.value] = _build_style_entry(
            style, trigger_finger=False, grip_radius_m=grip_radius_m
        )
        styles_dict[f"{style.value}_trigger"] = _build_style_entry(
            style, trigger_finger=True, grip_radius_m=grip_radius_m
        )

    presets_dict: dict[str, Any] = {}
    for preset_name, spec in NAMED_GRIP_PRESETS.items():
        variant_key = (
            f"{spec.style.value}_trigger" if spec.trigger_finger else spec.style.value
        )
        presets_dict[preset_name] = {
            "preset_name": preset_name,
            "style": spec.style.value,
            "trigger_finger": bool(spec.trigger_finger),
            "style_key": variant_key,
        }

    return {
        "schema_version": GRIP_TOPOLOGY_SCHEMA_VERSION,
        "appearance_compatibility": APPEARANCE_COMPATIBILITY_VERSION,
        "model_id": SIMSCAPE_MODEL_ID,
        "model_hash": SIMSCAPE_MODEL_HASH,
        "handedness": DEFAULT_HANDEDNESS,
        "side_roles": {"L": "lead", "R": "trail"},
        "attachment_mode": "weld",
        "force_injection_policy": "zero_injected_forces_allocated_reporting_only",
        "units": {
            "length": "m",
            "angle": "rad",
            "force": "N",
            "moment": "N*m",
        },
        "grip_radius_m": _round_val(grip_radius_m),
        "hand_frames": {
            "L": _serialize_grip_frame(interface.left),
            "R": _serialize_grip_frame(interface.right),
        },
        "presets": presets_dict,
        "styles": styles_dict,
    }


def validate_grip_topology_payload(payload: Mapping[str, Any]) -> None:
    """Validate a ``simscape-grip-topology/v1`` payload dictionary.

    Raises:
        TypeError: if ``payload`` is not a Mapping.
        ValueError: if required keys, schema version, styles, or pair IDs are invalid.
    """
    from src.shared.python.motion_matching.grip_contract import GripSpec

    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a Mapping, got {type(payload).__name__}")
    if payload.get("schema_version") != GRIP_TOPOLOGY_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {GRIP_TOPOLOGY_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    if payload.get("attachment_mode") not in ("weld", "closure"):
        raise ValueError("attachment_mode must be 'weld' or 'closure'")

    styles = payload.get("styles")
    if not isinstance(styles, Mapping) or not styles:
        raise ValueError("styles must be a non-empty Mapping")

    for style in (GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER):
        if style.value not in styles:
            raise ValueError(f"styles missing required style {style.value!r}")
        entry = styles[style.value]
        topo = contact_topology(GripSpec(style=style, trigger_finger=False))
        if list(entry.get("pair_ids", [])) != list(topo.pair_names()):
            raise ValueError(f"pair_ids mismatch for style {style.value!r}")
        posed = entry.get("posed_hands", {})
        if "lead" not in posed or "trail" not in posed:
            raise ValueError(f"posed_hands missing lead/trail for {style.value!r}")


def render_grip_topology_json(payload: Mapping[str, Any]) -> str:
    """Render deterministic JSON text with sorted keys and trailing newline."""
    validate_grip_topology_payload(payload)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def export_grip_topology(output_path: Path, *, check: bool = False) -> int:
    """Write or check ``gs3dx_grip_topology.json``; return 0 on success, 1 on drift."""
    payload = build_grip_topology_payload()
    rendered = render_grip_topology_json(payload)
    if check:
        if not output_path.exists():
            return 1
        current = output_path.read_text(encoding="utf-8")
        return 0 if current == rendered else 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return 0


@dataclass(frozen=True)
class SimscapeClosureWrench:
    """Single-frame closure wrench sample on the lead/club side in world axes."""

    t_s: float
    point_m: np.ndarray
    force_n: np.ndarray
    torque_nm: np.ndarray


@dataclass(frozen=True)
class _PalmThumbPatch:
    """Patch descriptor in club/grip coordinates for H1 allocate_from_closure."""

    point_club_m: np.ndarray
    normal_club: np.ndarray


@dataclass(frozen=True)
class _ClosureWrenchCarrier:
    """Minimal closure carrier compatible with G3 and H1 allocation entry points."""

    available: bool
    wrenches: tuple[SimscapeClosureWrench, ...]
    reason: str | None = None


@dataclass(frozen=True)
class SimscapeAllocatedFrameReport:
    """Per-frame allocated pad and hand-hand report from a Simscape snapshot.

    Invariants:
        - ``label == "allocation"``;
        - ``injected_body_wrenches`` is empty (``{}``);
        - ``max_injected_force_n == 0.0`` and ``max_injected_moment_nm == 0.0``.
    """

    time_s: float
    label: str
    normal_force_n: Mapping[str, float]
    tangential_force_n: Mapping[str, float]
    spin_moment_nm: Mapping[str, float]
    pair_labels: Mapping[str, PairLabel]
    hand_hand_contacts: AllocatedHandContacts
    palm_thumb_contact: PalmThumbForce
    step_report: GripStepReport
    closure_wrench_grip_6: np.ndarray
    injected_body_wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]]
    max_injected_force_n: float = 0.0
    max_injected_moment_nm: float = 0.0


@dataclass(frozen=True)
class SimscapeAllocatedSeriesReport:
    """Allocated contact force report across a recorded Simscape trajectory."""

    model_name: str
    style: GripStyle
    frames: tuple[SimscapeAllocatedFrameReport, ...]
    injected_force_count: int = 0
    max_injected_force_n: float = 0.0
    max_injected_moment_nm: float = 0.0


class SimscapeGripReportingAdapter:
    """Read-only reporting adapter for Simscape closed-loop recordings.

    Explicitly declares ``supports_wrench_application=False`` and
    ``supports_physical_pairs=False`` so no force can ever be fed back.
    """

    def __init__(self, handedness: Handedness = DEFAULT_HANDEDNESS) -> None:
        self._handedness = handedness

    def capabilities(self) -> GripCapabilities:
        """Return read-only reporting capabilities with wrench application disabled."""
        return GripCapabilities(
            attachment_modes=("weld", "closure"),
            supports_physical_pairs=False,
            supports_wrench_application=False,
            native_contact=False,
            available_bodies=("club", "hand_l", "hand_r"),
        )

    def resolve_binding(self, topology: ContactTopology) -> HandClubBinding:
        """Resolve a welded reporting-only HandClubBinding for ``topology``."""
        interface = default_grip_interface()
        sites = bind_rigid_hand_sites(
            topology,
            lead_hand_body="hand_l",
            trail_hand_body="hand_r",
            handedness=self._handedness,
        )
        club_frames = {
            "L": interface.left.matrix(),
            "R": interface.right.matrix(),
        }
        return HandClubBinding(
            model_id=SIMSCAPE_MODEL_ID,
            model_hash=SIMSCAPE_MODEL_HASH,
            handedness=self._handedness,
            attachment_mode="weld",
            club_body="club",
            lead_hand_body="hand_l",
            trail_hand_body="hand_r",
            sites=sites,
            unavailable_sites={},
            club_grip_frames=club_frames,
        )

    def sample_states(self, binding: HandClubBinding) -> BodyStates:
        """Return nominal identity poses for ``binding`` bodies."""
        poses = {body_id: np.eye(4) for body_id in binding.required_bodies()}
        return BodyStates(poses)

    def apply_wrenches(
        self, wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Always raise SchemaFeatureUnavailableError: zero forces may be injected."""
        del wrenches
        raise SchemaFeatureUnavailableError(
            SIMSCAPE_MODEL_ID,
            consumer="SimscapeGripReportingAdapter",
            feature="external wrench injection into recorded Simscape states",
            child="HRG-GS-M11",
            issue=13333,
        )


def closure_wrench_from_snapshot(
    snapshot: ClosedLoopSnapshot,
    *,
    time_s: float = 0.0,
) -> tuple[SimscapeClosureWrench, np.ndarray]:
    """Extract the 6-DOF closure wrench on the lead/club side from ``snapshot``.

    Solves ``A x = b`` via :func:`decompose` using the identity metric on
    actuated torques, embeds ``lambda`` into the 6-component grip-frame wrench
    ``[fx, fy, fz, tx, ty, tz]``, and maps ``-lambda`` (force/torque of closing
    trail hand on lead/club side) into world axes.
    """
    if not isinstance(snapshot, ClosedLoopSnapshot):
        raise TypeError(
            f"snapshot must be ClosedLoopSnapshot, got {type(snapshot).__name__}"
        )
    topo = snapshot.topology
    metric = AllocationMetric.identity(topo.n_actuated)
    decomp = decompose(snapshot, weights=metric)
    n_a = decomp.n_actuated
    m = decomp.m
    lam_rows = decomp.x_p[n_a : n_a + m]

    row_indices = _CLOSURE_ROW_EMBEDDING[topo.closure_kind]
    lam_6 = np.zeros(6, dtype=float)
    for row_val, idx_6 in zip(lam_rows, row_indices, strict=True):
        lam_6[idx_6] = float(row_val)

    # Wrench of closing (trail) hand ON the lead/club side is -lambda
    wrench_on_lead_grip = -lam_6
    wrench_on_lead_grip.setflags(write=False)

    grip_frame = snapshot.grip_frame
    rot_g = grip_frame.rotation
    origin_g = grip_frame.origin
    force_world = rot_g @ wrench_on_lead_grip[:3]
    torque_world = rot_g @ wrench_on_lead_grip[3:6]

    sample = SimscapeClosureWrench(
        t_s=float(time_s),
        point_m=origin_g.copy(),
        force_n=force_world,
        torque_nm=torque_world,
    )
    return sample, wrench_on_lead_grip


def _build_body_poses_at_grip(
    grip_rotation: np.ndarray,
    grip_origin: np.ndarray,
    interface: GripInterface,
) -> tuple[BodyStates, dict[str, BushingState], dict[str, BushingState]]:
    club_state = RigidBodyState(
        rotation=grip_rotation,
        position_m=grip_origin,
        velocity_m_s=np.zeros(3),
        omega_rad_s=np.zeros(3),
    )
    hand_states = hand_frame_states(club_state, interface)
    club_pose = np.eye(4)
    club_pose[:3, :3] = grip_rotation
    club_pose[:3, 3] = grip_origin

    lead_pose = np.eye(4)
    lead_pose[:3, :3] = hand_states["L"].rotation
    lead_pose[:3, 3] = hand_states["L"].position_m

    trail_pose = np.eye(4)
    trail_pose[:3, :3] = hand_states["R"].rotation
    trail_pose[:3, 3] = hand_states["R"].position_m

    body_states = BodyStates(
        {
            "club": club_pose,
            "hand_l": lead_pose,
            "hand_r": trail_pose,
        }
    )
    hand_bushing = {
        side: BushingState(
            rotation=hand_states[side].rotation,
            position_m=hand_states[side].position_m,
            velocity_m_s=np.zeros(3),
            omega_rad_s=np.zeros(3),
        )
        for side in ("L", "R")
    }
    club_bushing = {
        "L": club_state.frame(interface.left.matrix()),
        "R": club_state.frame(interface.right.matrix()),
    }
    return body_states, hand_bushing, club_bushing


def _allocate_pad_channels(
    topo_model: TopologyContactModel,
    hand_bushing: Mapping[str, BushingState],
    club_bushing: Mapping[str, BushingState],
    wrench_on_lead_grip: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Allocate non-negative pad normal, tangential, and spin loads for hand-grip pairs."""
    topology = topo_model.topology
    normal_out: dict[str, float] = {}
    tangent_out: dict[str, float] = {}
    spin_out: dict[str, float] = {}

    hg_pairs = topology.hand_grip_pairs()
    n_pads = max(1, len(hg_pairs))
    closure_f_mag = float(np.linalg.norm(wrench_on_lead_grip[:3]))
    closure_t_mag = float(np.linalg.norm(wrench_on_lead_grip[3:6]))
    pad_tangent_share = closure_f_mag / float(n_pads)
    pad_spin_share = closure_t_mag / float(n_pads)

    for pair in hg_pairs:
        site = topology.site(pair.site_a)
        side_char = "L" if site.hand == "lead" else "R"
        pw = topology_pair_wrench(
            topo_model,
            pair,
            hand_bushing[side_char],
            club_bushing[side_char],
            mode="contact",
        )
        # Normal inward component of closure force on this pad's inward normal
        inward = _site_inward_normal(site)
        sign_hand = 1.0 if site.hand == "trail" else -1.0
        extra_normal = max(
            0.0, float(sign_hand * np.dot(inward, wrench_on_lead_grip[:3]))
        ) / float(n_pads)
        normal_out[pair.name] = float(pw.normal_force_mag_n) + extra_normal
        tangent_out[pair.name] = float(pw.tangential_force_mag_n) + pad_tangent_share
        spin_out[pair.name] = _spin_magnitude_nm(pw) + pad_spin_share

    return normal_out, tangent_out, spin_out


def allocate_simscape_snapshot(
    snapshot: ClosedLoopSnapshot,
    *,
    topology: ContactTopology | None = None,
    time_s: float = 0.0,
    squeeze_per_hand_n: float = DEFAULT_SQUEEZE_PER_HAND_N,
    prior: AllocationPrior | None = None,
) -> SimscapeAllocatedFrameReport:
    """Allocate pad and hand-hand contact forces from a single ``ClosedLoopSnapshot``.

    Preconditions:
        ``snapshot`` is a :class:`ClosedLoopSnapshot`;
        ``squeeze_per_hand_n > 0``.
    Postconditions:
        All topology pairs are labelled ``"allocated"``;
        ``injected_body_wrenches`` is empty and zero forces are fed back.
    """
    if not isinstance(snapshot, ClosedLoopSnapshot):
        raise TypeError(
            f"snapshot must be ClosedLoopSnapshot, got {type(snapshot).__name__}"
        )
    if not math.isfinite(squeeze_per_hand_n) or squeeze_per_hand_n <= 0.0:
        raise ValueError("squeeze_per_hand_n must be finite and positive")

    topo = topology if topology is not None else contact_topology()
    interface = default_grip_interface()
    topo_model = build_topology_model(
        interface, topology=topo, squeeze_per_hand_n=squeeze_per_hand_n
    )
    adapter = SimscapeGripReportingAdapter()
    binding = adapter.resolve_binding(topo)

    sample, wrench_grip_6 = closure_wrench_from_snapshot(snapshot, time_s=time_s)
    carrier = _ClosureWrenchCarrier(available=True, wrenches=(sample,))

    grip_frame = snapshot.grip_frame
    rot_g = grip_frame.rotation
    origin_g = grip_frame.origin
    body_states, hand_bushing, club_bushing = _build_body_poses_at_grip(
        rot_g, origin_g, interface
    )

    step_report = evaluate_contacts(
        topo_model,
        binding,
        body_states,
        closure=carrier,
        prior=prior,
    )
    law = topo_model.law
    hand_hand = allocate_hand_hand_from_closure(
        carrier,
        club_rotation=rot_g,
        club_origin=origin_g,
        topology=topo,
        static_friction=law.static_friction,
        prior=prior,
    )
    pt_point, pt_normal = pair_contact_patch(topo, topo.pair("trail_palm_lead_thumb"))
    pt_patch = _PalmThumbPatch(point_club_m=pt_point, normal_club=pt_normal)
    palm_thumb = allocate_from_closure(
        carrier,
        club_rotation=rot_g,
        club_origin=origin_g,
        patch=pt_patch,
        static_friction=law.static_friction,
    )

    normal_map, tangent_map, spin_map = _allocate_pad_channels(
        topo_model, hand_bushing, club_bushing, wrench_grip_6
    )
    for pair_name, val in step_report.normal_force_n.items():
        if pair_name not in normal_map:
            normal_map[pair_name] = float(val)
            tangent_map[pair_name] = float(step_report.tangential_force_n[pair_name])
            spin_map[pair_name] = float(step_report.spin_moment_nm[pair_name])

    pair_labels: dict[str, PairLabel] = dict.fromkeys(topo.pair_names(), "allocated")
    empty_wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]] = MappingProxyType({})

    return SimscapeAllocatedFrameReport(
        time_s=float(time_s),
        label=ALLOCATION_LABEL,
        normal_force_n=MappingProxyType(normal_map),
        tangential_force_n=MappingProxyType(tangent_map),
        spin_moment_nm=MappingProxyType(spin_map),
        pair_labels=MappingProxyType(pair_labels),
        hand_hand_contacts=hand_hand,
        palm_thumb_contact=palm_thumb,
        step_report=step_report,
        closure_wrench_grip_6=wrench_grip_6,
        injected_body_wrenches=empty_wrenches,
        max_injected_force_n=0.0,
        max_injected_moment_nm=0.0,
    )


def allocate_simscape_export(
    export_or_path: SimscapeExport | str | Path,
    *,
    topology: ContactTopology | None = None,
    squeeze_per_hand_n: float = DEFAULT_SQUEEZE_PER_HAND_N,
    prior: AllocationPrior | None = None,
    test_only: bool = False,
) -> SimscapeAllocatedSeriesReport:
    """Allocate pad and hand-hand forces across all frames of a Simscape export.

    Preconditions:
        ``export_or_path`` is a :class:`SimscapeExport` or path to a
        ``simscape-closed-loop-snapshot/v1`` JSON file.
    Postconditions:
        Returns :class:`SimscapeAllocatedSeriesReport` with
        ``injected_force_count == 0`` and ``max_injected_force_n == 0.0``.
    """
    if isinstance(export_or_path, str | Path):
        export_obj = load_simscape_export(export_or_path)
    elif isinstance(export_or_path, SimscapeExport):
        export_obj = export_or_path
    else:
        raise TypeError(
            "export_or_path must be SimscapeExport, str, or Path, got "
            f"{type(export_or_path).__name__}"
        )

    topo = topology if topology is not None else contact_topology()
    prov = export_obj.provenance
    model_name = prov.model_file or prov.provenance_kind
    in_prov = InputProvenance(
        kind="simscape_closed_loop_snapshot_v1",
        source=model_name,
        note="allocated reporting from recorded Simscape closed-loop states",
    )

    frame_reports: list[SimscapeAllocatedFrameReport] = []
    for idx in range(len(export_obj.states)):
        snap = snapshot_from_export(
            export_obj,
            idx,
            input_provenance=in_prov,
            test_only=test_only,
        )
        rep = allocate_simscape_snapshot(
            snap,
            topology=topo,
            time_s=float(idx),
            squeeze_per_hand_n=squeeze_per_hand_n,
            prior=prior,
        )
        frame_reports.append(rep)

    return SimscapeAllocatedSeriesReport(
        model_name=model_name,
        style=topo.style,
        frames=tuple(frame_reports),
        injected_force_count=0,
        max_injected_force_n=0.0,
        max_injected_moment_nm=0.0,
    )

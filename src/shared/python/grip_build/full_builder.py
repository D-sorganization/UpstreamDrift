"""Full procedural grip build (HRG-GB3) on top of GB-2 lead-finger curl."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from src.shared.python.grip_build.builder import (
    BONE_TOLERANCE_M,
    PAD_TOLERANCE_M,
    FlatHandCurlBuilder,
    _JOINT_NAMES,
    _merge_status,
)
from src.shared.python.grip_build.club_sdf import ClubSDF
from src.shared.python.grip_build.contact_protocol import ContactDistanceModel
from src.shared.python.grip_build.hand_chain import HandChain
from src.shared.python.grip_build.receipt import (
    BUILD_RECEIPT_SCHEMA_VERSION,
    BuildReceipt,
    BuildStageRecord,
    BuildStatus,
    FullGripBuildResult,
)
from src.shared.python.grip_build.stages_gb3 import (
    active_pairs_from_clearances,
    away_target_azimuth_rad,
    grip_pose_azimuth_from_direction,
    grip_pose_thumb_direction_rad,
    palm_wrap_stage,
    thumb_press_stage,
    trail_link_clearances,
)
from src.shared.python.grip_build.thumb import (
    thumb_capsules,
    thumb_ip_cap_rad,
    thumb_radial_direction,
)
from src.shared.python.grip_build.finger_order import lead_finger_curl_order
from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_contact.grip_geometry import GripGeometry, get_grip_preset
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec
from src.shared.python.model_appearance.grip_hand_mesh import GRIP_RADIUS_RANGE_M

BUILDER_VERSION: Final[str] = "gb3-0.1.0"


@dataclass
class _Gb3HandPipeline:
    chain: HandChain
    club: ClubSDF
    stages: list[BuildStageRecord]
    overall: BuildStatus
    palm_stage: BuildStageRecord
    thumb_stage: BuildStageRecord
    blocks: list[tuple[np.ndarray, np.ndarray, float]]


@dataclass
class _GripFinalize:
    stages: list[BuildStageRecord]
    overall: BuildStatus
    lead_chain: HandChain
    trail_chain: HandChain
    clearances: dict[str, float]
    critical_stages: tuple[
        BuildStageRecord, BuildStageRecord, BuildStageRecord, BuildStageRecord
    ]


def _validate_grip_geometry(geometry: GripGeometry) -> None:
    r = geometry.radius_at(0.05)
    lo, hi = GRIP_RADIUS_RANGE_M
    if not (lo <= r <= hi):
        raise ValueError(f"grip radius {r:.4f} m outside {GRIP_RADIUS_RANGE_M}")


def _chain_for_hand(
    spec: GripSpec,
    hand_spec: HandGripSpec,
    *,
    geometry: GripGeometry,
    curl_sign: int,
    axial_shift_m: float,
) -> tuple[HandChain, ClubSDF]:
    grip_radius = geometry.radius_at(spec.lead_below_butt_m)
    chain = HandChain(
        hand_size=spec.hand_size,
        curl_sign=curl_sign,
        dip_over_pip=dict(hand_spec.joint_coupling_overrides),
        grip_radius_m=grip_radius,
        axial_shift_m=axial_shift_m,
    )
    chain.flat_lateral_offset_m = (hand_spec.palm_finger_depth - 0.5) * 0.012
    club = ClubSDF(geometry=geometry, axial_offset_m=hand_spec.index_contact_station)
    return chain, club


def _curl_fingers(
    builder: FlatHandCurlBuilder,
    chain: HandChain,
    club: ClubSDF,
    model: ContactDistanceModel,
    *,
    prefix: str,
    lead_blocks: list[tuple[np.ndarray, np.ndarray, float]] | None = None,
) -> tuple[
    list[BuildStageRecord], BuildStatus, list[tuple[np.ndarray, np.ndarray, float]]
]:
    stages: list[BuildStageRecord] = []
    overall = BuildStatus.OK
    placed: list[tuple[np.ndarray, np.ndarray, float]] = []
    if lead_blocks:
        placed.extend(lead_blocks)
    for finger in ("index", "middle", "ring", "little"):
        for joint in _JOINT_NAMES:
            status, flex_rad, clearance = builder._curl_joint(
                chain,
                finger,
                joint,
                club,
                model,
                placed,
                bone_tol=BONE_TOLERANCE_M,
                pad_tol=PAD_TOLERANCE_M,
            )
            tol = PAD_TOLERANCE_M if joint == "dip" else BONE_TOLERANCE_M
            stages.append(
                BuildStageRecord(
                    stage_id=f"{prefix}_{finger}_{joint}",
                    finger=finger,
                    joint=joint,
                    flexion_rad=flex_rad,
                    status=status,
                    min_clearance_m=clearance,
                    target_tolerance_m=tol,
                )
            )
            overall = _merge_status(overall, status)
            if joint == "dip":
                placed.extend(chain.finger_capsules(finger))
    return stages, overall, placed


def _joint_table(chain: HandChain) -> dict[str, float]:
    table: dict[str, float] = {}
    for finger, angles in chain.joint_angles.items():
        table[f"{finger}_mcp"] = angles.mcp_rad
        table[f"{finger}_pip"] = angles.pip_rad
        table[f"{finger}_dip"] = angles.dip_rad
    table["palm_wrap"] = chain.palm_wrap_rad
    table["thumb_cmc"] = chain.thumb.cmc_rad
    table["thumb_mcp"] = chain.thumb.mcp_rad
    table["thumb_ip"] = chain.thumb.ip_rad
    table["thumb_azimuth_rad"] = chain.thumb_azimuth_rad
    table["grip_radius_m"] = chain.grip_radius_m
    table["axial_shift_m"] = chain.axial_shift_m
    return table


def _topology_clearances_m(
    spec: GripSpec,
    *,
    lead_chain: HandChain,
    trail_chain: HandChain,
    lead_club: ClubSDF,
    trail_club: ClubSDF,
    model: ContactDistanceModel,
) -> dict[str, float]:
    clearances = trail_link_clearances(
        spec, lead_chain, trail_chain, model, trail_club=trail_club
    )
    lead_caps = thumb_capsules(
        lead_chain.mesh_params,
        lead_chain.thumb_azimuth_rad,
        lead_chain.thumb,
        lead_chain.grip_radius_m,
        palm_wrap_rad=lead_chain.palm_wrap_rad,
    )
    trail_caps = thumb_capsules(
        trail_chain.mesh_params,
        trail_chain.thumb_azimuth_rad,
        trail_chain.thumb,
        trail_chain.grip_radius_m,
        palm_wrap_rad=trail_chain.palm_wrap_rad,
        axial_shift_m=trail_chain.axial_shift_m,
    )
    clearances["_lead_thumb_club_clearance"] = min(
        model.club_clearance(p0, p1, r, lead_club) for p0, p1, r in lead_caps
    )
    clearances["_trail_thumb_club_clearance"] = min(
        model.club_clearance(p0, p1, r, trail_club) for p0, p1, r in trail_caps
    )
    heel = lead_chain.palm_heel_point()
    clearances["lead_heel_pad_butt"] = model.club_clearance(
        heel, heel, lead_chain.palm_heel_radius_m(), lead_club
    )
    return clearances


def _full_grip_result(spec: GripSpec, assembly: _GripFinalize) -> FullGripBuildResult:
    joint_angles = {
        "lead": _joint_table(assembly.lead_chain),
        "trail": _joint_table(assembly.trail_chain),
    }
    overall = assembly.overall
    if overall not in (BuildStatus.INFEASIBLE, BuildStatus.JOINT_LIMIT):
        if all(stage.status == BuildStatus.OK for stage in assembly.critical_stages):
            overall = BuildStatus.OK
    receipt = BuildReceipt(
        schema_version=BUILD_RECEIPT_SCHEMA_VERSION,
        builder_version=BUILDER_VERSION,
        spec_sha256=spec.spec_sha256,
        curl_order=spec.curl_order,
        handedness=str(spec.handedness),
        bone_tolerance_m=BONE_TOLERANCE_M,
        pad_tolerance_m=PAD_TOLERANCE_M,
        stages=tuple(assembly.stages),
        overall_status=overall,
        gaps_m=(),
    )
    active = active_pairs_from_clearances(
        assembly.clearances, pad_tol=PAD_TOLERANCE_M, bone_tol=BONE_TOLERANCE_M
    )
    return FullGripBuildResult(
        receipt=receipt,
        joint_angles_rad=joint_angles,
        contact_clearances_m=assembly.clearances,
        active_contact_pairs=active,
        status=overall,
    )


def _gb3_lead_hand(
    builder: FlatHandCurlBuilder,
    spec: GripSpec,
    geometry: GripGeometry,
    model: ContactDistanceModel,
) -> _Gb3HandPipeline:
    chain, club = _chain_for_hand(
        spec, spec.lead, geometry=geometry, curl_sign=1, axial_shift_m=0.0
    )
    finger_stages, overall, blocks = _curl_fingers(
        builder, chain, club, model, prefix="lead"
    )
    palm_stage, palm_status = palm_wrap_stage(
        chain, club, model, bone_tol=BONE_TOLERANCE_M, pad_tol=PAD_TOLERANCE_M
    )
    stages = list(finger_stages) + [palm_stage]
    overall = _merge_status(overall, palm_status)
    thumb_rad = grip_pose_thumb_direction_rad(
        spec.resolved_thumb_azimuth_deg("lead"), str(spec.handedness)
    )
    thumb_stage, thumb_status, _ = thumb_press_stage(
        chain,
        club,
        model,
        thumb_radial=thumb_rad,
        ip_cap_rad=thumb_ip_cap_rad(spec.lead),
        bone_tol=BONE_TOLERANCE_M,
        pad_tol=PAD_TOLERANCE_M,
        stage_id="lead_thumb_press",
    )
    stages.append(thumb_stage)
    overall = _merge_status(overall, thumb_status)
    return _Gb3HandPipeline(
        chain=chain,
        club=club,
        stages=stages,
        overall=overall,
        palm_stage=palm_stage,
        thumb_stage=thumb_stage,
        blocks=blocks,
    )


def _gb3_trail_hand(
    builder: FlatHandCurlBuilder,
    spec: GripSpec,
    geometry: GripGeometry,
    model: ContactDistanceModel,
    *,
    lead_blocks: list[tuple[np.ndarray, np.ndarray, float]],
) -> _Gb3HandPipeline:
    shift = -(
        spec.hand_spacing_m + (0.012 if spec.style is GripStyle.TEN_FINGER else 0.0)
    )
    chain, club = _chain_for_hand(
        spec,
        spec.trail,
        geometry=geometry,
        curl_sign=-1 if spec.handedness == "right" else 1,
        axial_shift_m=shift,
    )
    finger_stages, overall, _ = _curl_fingers(
        builder, chain, club, model, prefix="trail", lead_blocks=lead_blocks
    )
    if spec.style is GripStyle.INTERLOCK and spec.hand_size.length_m <= 0.145:
        overall = BuildStatus.INFEASIBLE
    palm_stage, palm_status = palm_wrap_stage(
        chain, club, model, bone_tol=BONE_TOLERANCE_M, pad_tol=PAD_TOLERANCE_M
    )
    stages = list(finger_stages) + [palm_stage]
    overall = _merge_status(overall, palm_status)
    thumb_rad = grip_pose_thumb_direction_rad(
        spec.resolved_thumb_azimuth_deg("trail"), str(spec.handedness)
    )
    thumb_stage, thumb_status, _ = thumb_press_stage(
        chain,
        club,
        model,
        thumb_radial=thumb_rad,
        ip_cap_rad=thumb_ip_cap_rad(spec.trail),
        bone_tol=BONE_TOLERANCE_M,
        pad_tol=PAD_TOLERANCE_M,
        stage_id="trail_thumb_press",
    )
    stages.append(thumb_stage)
    overall = _merge_status(overall, thumb_status)
    return _Gb3HandPipeline(
        chain=chain,
        club=club,
        stages=stages,
        overall=overall,
        palm_stage=palm_stage,
        thumb_stage=thumb_stage,
        blocks=[],
    )


def build_full_grip(
    spec: GripSpec,
    *,
    geometry: GripGeometry | None = None,
    contact_model: ContactDistanceModel | None = None,
) -> FullGripBuildResult:
    if not isinstance(spec, GripSpec):
        raise TypeError(f"spec must be GripSpec, got {type(spec).__name__}")
    geometry = geometry if geometry is not None else get_grip_preset()
    _validate_grip_geometry(geometry)
    builder = FlatHandCurlBuilder(contact_model=contact_model)
    model = builder.contact_model
    if model is None:
        from src.shared.python.grip_build.registry import get_contact_distance_model

        model = get_contact_distance_model()
    lead = _gb3_lead_hand(builder, spec, geometry, model)
    trail = _gb3_trail_hand(builder, spec, geometry, model, lead_blocks=lead.blocks)
    stages = lead.stages + trail.stages
    overall = _merge_status(lead.overall, trail.overall)
    clearances = _topology_clearances_m(
        spec,
        lead_chain=lead.chain,
        trail_chain=trail.chain,
        lead_club=lead.club,
        trail_club=trail.club,
        model=model,
    )
    assembly = _GripFinalize(
        stages=stages,
        overall=overall,
        lead_chain=lead.chain,
        trail_chain=trail.chain,
        clearances=clearances,
        critical_stages=(
            lead.palm_stage,
            lead.thumb_stage,
            trail.palm_stage,
            trail.thumb_stage,
        ),
    )
    return _full_grip_result(spec, assembly)


def thumb_radial_azimuth_deg(result: FullGripBuildResult, *, role: str) -> float:
    """Measured thumb azimuth in grip_pose convention (0 = away from target)."""
    from src.shared.python.grip_build.thumb import ThumbJointAngles
    from dataclasses import replace

    from src.shared.python.model_appearance.grip_hand_mesh import (
        DEFAULT_HAND_MESH_PARAMS,
    )

    data = result.joint_angles_rad[role]
    mesh = replace(DEFAULT_HAND_MESH_PARAMS)
    angles = ThumbJointAngles(
        cmc_rad=data["thumb_cmc"],
        mcp_rad=data["thumb_mcp"],
        ip_rad=data["thumb_ip"],
    )
    thumb_rad = np.array(
        [
            math.cos(data["thumb_azimuth_rad"]),
            math.sin(data["thumb_azimuth_rad"]),
            0.0,
        ],
        dtype=np.float64,
    )
    caps = thumb_capsules(
        mesh,
        0.0,
        angles,
        grip_radius_m=data["grip_radius_m"],
        radial_unit=thumb_rad,
        palm_wrap_rad=data["palm_wrap"],
        axial_shift_m=0.0 if role == "lead" else data.get("axial_shift_m", 0.0),
    )
    base = caps[0][0]
    direction = base.astype(np.float64)
    direction[2] = 0.0
    return grip_pose_azimuth_from_direction(
        direction, handedness=result.receipt.handedness
    )


def build_grip(
    spec: GripSpec,
    *,
    geometry: GripGeometry | None = None,
    contact_model: ContactDistanceModel | None = None,
) -> FullGripBuildResult:
    return build_full_grip(spec, geometry=geometry, contact_model=contact_model)

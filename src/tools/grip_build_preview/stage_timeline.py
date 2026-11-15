"""Stage keyframes reconstructed from GB-2 GripBuildResult (no builder edits)."""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.python.grip_build.hand_chain import HandChain
from src.shared.python.grip_build.receipt import (
    BuildStageRecord,
    BuildStatus,
    GripBuildResult,
)
from src.shared.python.grip_contact.grip_geometry import GripGeometry, get_grip_preset
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec


@dataclass(frozen=True)
class StageKeyframe:
    """One deterministic pose in the procedural build order."""

    stage_id: str
    joint_angles_rad: dict[str, dict[str, float]]
    status: BuildStatus
    active_stage: BuildStageRecord | None
    clearance_m: float | None
    tolerance_m: float | None


def hand_chain_for_spec(
    spec: GripSpec, *, geometry: GripGeometry | None = None
) -> HandChain:
    if not isinstance(spec, GripSpec):
        raise TypeError(f"spec must be GripSpec, got {type(spec).__name__}")
    geometry = geometry if geometry is not None else get_grip_preset()
    hand_spec = spec.lead
    if not isinstance(hand_spec, HandGripSpec):
        raise TypeError("spec.lead must be HandGripSpec")
    grip_radius = geometry.radius_at(spec.lead_below_butt_m)
    chain = HandChain(
        hand_size=spec.hand_size,
        curl_sign=1,
        dip_over_pip=dict(hand_spec.joint_coupling_overrides),
        grip_radius_m=grip_radius,
    )
    depth_shift = (hand_spec.palm_finger_depth - 0.5) * 0.012
    chain.flat_lateral_offset_m = depth_shift
    return chain


def _snapshot_angles(chain: HandChain) -> dict[str, dict[str, float]]:
    return {
        finger: {
            "mcp": chain.joint_angles[finger].mcp_rad,
            "pip": chain.joint_angles[finger].pip_rad,
            "dip": chain.joint_angles[finger].dip_rad,
        }
        for finger in chain.joint_angles
    }


def build_stage_keyframes(
    spec: GripSpec,
    result: GripBuildResult,
    *,
    geometry: GripGeometry | None = None,
) -> tuple[StageKeyframe, ...]:
    if not isinstance(result, GripBuildResult):
        raise TypeError("result must be GripBuildResult")
    chain = hand_chain_for_spec(spec, geometry=geometry)
    keyframes: list[StageKeyframe] = [
        StageKeyframe(
            stage_id="flat_hand",
            joint_angles_rad=_snapshot_angles(chain),
            status=BuildStatus.OK,
            active_stage=None,
            clearance_m=None,
            tolerance_m=None,
        )
    ]
    for record in result.receipt.stages:
        angles = chain.joint_angles[record.finger]
        setattr(angles, f"{record.joint}_rad", record.flexion_rad)
        chain.apply_dip_coupling(record.finger)
        keyframes.append(
            StageKeyframe(
                stage_id=record.stage_id,
                joint_angles_rad=_snapshot_angles(chain),
                status=record.status,
                active_stage=record,
                clearance_m=record.min_clearance_m,
                tolerance_m=record.target_tolerance_m,
            )
        )
    return tuple(keyframes)

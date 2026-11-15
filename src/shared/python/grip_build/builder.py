"""Ordered flat-hand curl-to-contact builder (HRG-GB2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from src.shared.python.grip_build.club_sdf import ClubSDF
from src.shared.python.grip_build.contact_protocol import ContactDistanceModel
from src.shared.python.grip_build.hand_chain import HandChain
from src.shared.python.grip_build.receipt import (
    BUILD_RECEIPT_SCHEMA_VERSION,
    BuildReceipt,
    BuildStageRecord,
    BuildStatus,
    GripBuildResult,
)
from src.shared.python.grip_build.registry import (
    DEFAULT_BUILDER_NAME,
    register_grip_builder,
)
from src.shared.python.grip_contact.grip_geometry import GripGeometry, get_grip_preset
from src.shared.python.grip_spec.spec import GripSpec, HandGripSpec
from src.shared.python.model_appearance.grip_hand_mesh import GRIP_RADIUS_RANGE_M

BUILDER_VERSION: Final[str] = "gb2-0.1.0"
BONE_TOLERANCE_M: Final[float] = 0.0005
PAD_TOLERANCE_M: Final[float] = 0.002
_JOINT_NAMES: Final[tuple[str, ...]] = ("mcp", "pip", "dip")


def _validate_grip_geometry(geometry: GripGeometry) -> None:
    r = geometry.radius_at(0.05)
    lo, hi = GRIP_RADIUS_RANGE_M
    if not (lo <= r <= hi):
        raise ValueError(f"grip radius {r:.4f} m outside {GRIP_RADIUS_RANGE_M}")


@dataclass
class FlatHandCurlBuilder:
    """Lead-hand finger curl builder (palm/thumb/trail hand are GB-3)."""

    name: str = DEFAULT_BUILDER_NAME
    contact_model: ContactDistanceModel | None = None

    def build_lead_fingers(
        self,
        spec: GripSpec,
        *,
        geometry: GripGeometry | None = None,
        contact_model: ContactDistanceModel | None = None,
    ) -> GripBuildResult:
        if not isinstance(spec, GripSpec):
            raise TypeError(f"spec must be GripSpec, got {type(spec).__name__}")
        geometry = geometry if geometry is not None else get_grip_preset()
        _validate_grip_geometry(geometry)
        model = contact_model or self.contact_model
        if model is None:
            from src.shared.python.grip_build.registry import get_contact_distance_model

            model = get_contact_distance_model()
        hand_spec = spec.lead
        if not isinstance(hand_spec, HandGripSpec):
            raise TypeError("spec.lead must be HandGripSpec")
        curl_sign = 1
        grip_radius = geometry.radius_at(spec.lead_below_butt_m)
        chain = HandChain(
            hand_size=spec.hand_size,
            curl_sign=curl_sign,
            dip_over_pip=dict(hand_spec.joint_coupling_overrides.dip_over_pip),
            grip_radius_m=grip_radius,
        )
        depth_shift = (hand_spec.palm_finger_depth - 0.5) * 0.012
        chain.flat_lateral_offset_m = depth_shift
        club = ClubSDF(
            geometry=geometry, axial_offset_m=hand_spec.index_contact_station
        )
        stages: list[BuildStageRecord] = []
        gaps: list[float] = []
        overall = BuildStatus.OK
        placed_capsules: list[tuple[np.ndarray, np.ndarray, float]] = []
        for finger in spec.curl_order:
            for joint in _JOINT_NAMES:
                status, flex_rad, clearance = self._curl_joint(
                    chain,
                    finger,
                    joint,
                    club,
                    model,
                    placed_capsules,
                    bone_tol=BONE_TOLERANCE_M,
                    pad_tol=PAD_TOLERANCE_M,
                )
                stage_id = f"lead_{finger}_{joint}"
                tol = PAD_TOLERANCE_M if joint == "dip" else BONE_TOLERANCE_M
                stages.append(
                    BuildStageRecord(
                        stage_id=stage_id,
                        finger=finger,
                        joint=joint,
                        flexion_rad=flex_rad,
                        status=status,
                        min_clearance_m=clearance,
                        target_tolerance_m=tol,
                    )
                )
                if status != BuildStatus.OK:
                    overall = _merge_status(overall, status)
                    if clearance < 0.0:
                        gaps.append(abs(clearance))
                if joint == "dip":
                    placed_capsules.extend(chain.finger_capsules(finger))
        joint_angles = {
            finger: {
                "mcp": chain.joint_angles[finger].mcp_rad,
                "pip": chain.joint_angles[finger].pip_rad,
                "dip": chain.joint_angles[finger].dip_rad,
            }
            for finger in spec.curl_order
        }
        receipt = BuildReceipt(
            schema_version=BUILD_RECEIPT_SCHEMA_VERSION,
            builder_version=BUILDER_VERSION,
            spec_sha256=spec.spec_sha256(),
            curl_order=spec.curl_order,
            handedness=str(spec.handedness),
            bone_tolerance_m=BONE_TOLERANCE_M,
            pad_tolerance_m=PAD_TOLERANCE_M,
            stages=tuple(stages),
            overall_status=overall,
            gaps_m=tuple(gaps),
        )
        return GripBuildResult(
            receipt=receipt, joint_angles_rad=joint_angles, status=overall
        )

    def _curl_joint(
        self,
        chain: HandChain,
        finger: str,
        joint: str,
        club: ClubSDF,
        model: ContactDistanceModel,
        placed_capsules: list[tuple[np.ndarray, np.ndarray, float]],
        *,
        bone_tol: float,
        pad_tol: float,
    ) -> tuple[BuildStatus, float, float]:
        angles = chain.joint_angles[finger]
        limit = {
            "mcp": math.radians(90.0),
            "pip": math.radians(100.0),
            "dip": math.radians(80.0),
        }[joint]
        lo = 0.0
        hi = limit
        best_clearance = math.inf
        contact_angle = 0.0
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            setattr(angles, f"{joint}_rad", mid)
            chain.apply_dip_coupling(finger)
            p0, p1, radius = chain.segment_capsule(finger, joint)
            clearance = model.club_clearance(p0, p1, radius, club)
            for other in placed_capsules:
                clearance = min(
                    clearance,
                    model.finger_clearance(
                        p0, p1, radius, other[0], other[1], other[2]
                    ),
                )
            tol = pad_tol if joint == "dip" else bone_tol
            best_clearance = min(best_clearance, clearance)
            if clearance >= -tol:
                contact_angle = mid
                hi = mid
            else:
                lo = mid
        setattr(angles, f"{joint}_rad", contact_angle)
        chain.apply_dip_coupling(finger)
        p0, p1, radius = chain.segment_capsule(finger, joint)
        final_clearance = model.club_clearance(p0, p1, radius, club)
        for other in placed_capsules:
            final_clearance = min(
                final_clearance,
                model.finger_clearance(p0, p1, radius, other[0], other[1], other[2]),
            )
        tol = pad_tol if joint == "dip" else bone_tol
        if final_clearance < -tol:
            if contact_angle >= limit - 1e-6:
                return BuildStatus.JOINT_LIMIT, contact_angle, final_clearance
            return BuildStatus.GAP, contact_angle, final_clearance
        if final_clearance > tol + 1e-3:
            return BuildStatus.GAP, contact_angle, final_clearance
        if final_clearance > tol:
            return BuildStatus.GAP, contact_angle, final_clearance
        return BuildStatus.OK, contact_angle, final_clearance


def _merge_status(current: BuildStatus, new: BuildStatus) -> BuildStatus:
    priority = {
        BuildStatus.INFEASIBLE: 4,
        BuildStatus.JOINT_LIMIT: 3,
        BuildStatus.GAP: 2,
        BuildStatus.OK: 1,
    }
    return new if priority[new] > priority[current] else current


def build_lead_fingers(
    spec: GripSpec,
    *,
    geometry: GripGeometry | None = None,
    contact_model: ContactDistanceModel | None = None,
) -> GripBuildResult:
    """Convenience entrypoint using the registered default builder."""
    from src.shared.python.grip_build.registry import get_grip_builder

    return get_grip_builder().build_lead_fingers(
        spec, geometry=geometry, contact_model=contact_model
    )


_default_builder = FlatHandCurlBuilder()
register_grip_builder(DEFAULT_BUILDER_NAME, _default_builder)

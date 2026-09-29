"""Unit tests for Head, Trunk, and Grip Diagnostic Receipts (MMR-06-I, #11106).

Verifies:
1. Pure axial turn exhibits zero head-centre translation error while SO(3) orientation error is non-zero.
2. Marker-centroid to body-centre comparison fails closed without an explicit attachment transform.
3. Left/right frame conventions and rigid attachment transform round-trips (T^-1 * T == I).
4. Trunk observable: joint-centre frame remains invariant under axial turn while C7 proxy swings.
5. ObservabilityDiagnosticReceipt serialization, content hashes, and save/load round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.diagnostics.observability_receipts import (
    OBSERVABILITY_RECEIPT_SCHEMA,
    AttachmentTransform,
    GripAndClubfaceCalibration,
    HeadDiagnosticCalculator,
    HeadDiagnosticResult,
    ObservabilityDiagnosticReceipt,
    TrunkDiagnosticCalculator,
    TrunkObservableKind,
)

pytestmark = pytest.mark.unit


def _make_receipt(marker_residuals_mm: dict[str, float] | None = None) -> Any:
    """Build a representative receipt for serialization/tamper-evidence tests."""
    head_attachment = AttachmentTransform(
        from_frame="head_centre",
        to_frame="marker_centroid",
        translation_m=(0.03, 0.02, 0.07),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    lead_grip = AttachmentTransform(
        from_frame="LH_wrist",
        to_frame="club_grip",
        translation_m=(0.0, 0.0, -0.05),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    trail_grip = AttachmentTransform(
        from_frame="RH_wrist",
        to_frame="club_grip",
        translation_m=(0.0, 0.0, -0.05),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    clubface = AttachmentTransform(
        from_frame="club_grip",
        to_frame="club_face",
        translation_m=(0.0, 0.0, -0.20),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    calibration = GripAndClubfaceCalibration(
        lead_hand_frame="LH_wrist",
        trail_hand_frame="RH_wrist",
        grip_frame="club_grip",
        face_frame="club_face",
        lead_grip_transform=lead_grip,
        trail_grip_transform=trail_grip,
        clubface_transform=clubface,
        convention="lead_left_trail_right",
    )
    return ObservabilityDiagnosticReceipt(
        schema_version=OBSERVABILITY_RECEIPT_SCHEMA,
        head_attachment=head_attachment,
        grip_calibration=calibration,
        trunk_observable=TrunkObservableKind.JOINT_CENTRE,
        head_centre_error_m=0.002,
        head_orientation_error_deg=3.4,
        marker_residuals_mm=(
            {"HeadFront": 2.1, "HeadTop": 1.8, "HeadSide": 2.4}
            if marker_residuals_mm is None
            else marker_residuals_mm
        ),
        trunk_com_residual_m=0.004,
    )


def test_schema_version_is_stable() -> None:
    assert OBSERVABILITY_RECEIPT_SCHEMA == "observability-diagnostics/1.0.0"


def test_pure_axial_turn_head_centre_translation_zero_while_orientation_error_nonzero() -> (
    None
):
    # Head centre sits at (0, 0, 1.5) m
    head_centre_true = np.array([0.0, 0.0, 1.5])
    # Three surface markers placed around skull (radius 0.1 m): front (+x), top (+z), side (+y)
    offset_front = np.array([0.1, 0.0, 0.0])
    offset_top = np.array([0.0, 0.0, 0.12])
    offset_side = np.array([0.0, 0.09, 0.0])

    centroid_offset = (offset_front + offset_top + offset_side) / 3.0

    t_head = AttachmentTransform(
        from_frame="head_centre",
        to_frame="marker_centroid",
        translation_m=(
            float(centroid_offset[0]),
            float(centroid_offset[1]),
            float(centroid_offset[2]),
        ),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )

    calc = HeadDiagnosticCalculator(attachment_transform=t_head)

    # Orientation 1: Identity
    r_identity = np.eye(3)
    markers_t0 = {
        "HeadFront": head_centre_true + offset_front,
        "HeadTop": head_centre_true + offset_top,
        "HeadSide": head_centre_true + offset_side,
    }

    # Orientation 2: 45 degree (pi/4) pure axial yaw rotation about z
    theta = np.pi / 4.0
    r_yaw = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # Head centre does NOT move: pure axial turn on neck axis
    markers_t1 = {
        "HeadFront": head_centre_true + r_yaw @ offset_front,
        "HeadTop": head_centre_true + r_yaw @ offset_top,
        "HeadSide": head_centre_true + r_yaw @ offset_side,
    }

    # Evaluate diagnostics when model head centre is kept at head_centre_true, but orientation is still identity
    diag = calc.evaluate(
        model_head_centre=head_centre_true,
        model_head_rotation=r_identity,
        observed_markers=markers_t1,
    )

    assert isinstance(diag, HeadDiagnosticResult)
    # Head centre translation error is near zero
    assert diag.head_centre_error_m == pytest.approx(0.0, abs=1e-6)
    # Orientation error is 45 degrees (pi/4 radians)
    assert diag.orientation_error_deg == pytest.approx(45.0, abs=1e-3)
    assert diag.orientation_error_rad == pytest.approx(np.pi / 4.0, abs=1e-4)
    # Marker-level 3D residuals are non-zero due to rotated markers
    assert diag.marker_residuals_m["HeadFront"] > 0.05
    assert diag.marker_residuals_m["HeadSide"] > 0.05


def test_marker_centroid_without_attachment_transform_fails_closed() -> None:
    calc = HeadDiagnosticCalculator(attachment_transform=None)
    markers = {
        "HeadFront": np.array([0.1, 0.0, 1.5]),
        "HeadTop": np.array([0.0, 0.0, 1.62]),
        "HeadSide": np.array([0.0, 0.09, 1.5]),
    }

    with pytest.raises(ValueError, match="attachment transform"):
        calc.evaluate(
            model_head_centre=np.array([0.0, 0.0, 1.5]),
            model_head_rotation=np.eye(3),
            observed_markers=markers,
        )


def test_left_right_frame_conventions_and_rigid_transform_roundtrips() -> None:
    # Rigid transform with translation and 3D rotation
    theta = 0.5
    r = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    t = np.array([0.12, -0.05, 0.35])

    transform = AttachmentTransform.from_matrix_and_translation(
        from_frame="LH_wrist",
        to_frame="club_grip",
        rotation=r,
        translation=t,
    )

    # Point round-trip: T^-1(T(p)) == p
    p_test = np.array([0.2, -0.1, 0.8])
    p_forward = transform.apply(p_test)
    inv_transform = transform.inverse()
    p_back = inv_transform.apply(p_forward)

    np.testing.assert_allclose(p_back, p_test, atol=1e-12)
    assert transform.roundtrip_identity(p_test) is True

    # Frame convention validation: grip is calibrated from each hand, face from grip.
    trail_transform = AttachmentTransform.from_matrix_and_translation(
        from_frame="RH_wrist",
        to_frame="club_grip",
        rotation=r,
        translation=t,
    )
    clubface_transform = AttachmentTransform.from_matrix_and_translation(
        from_frame="club_grip",
        to_frame="club_face",
        rotation=np.eye(3),
        translation=np.zeros(3),
    )
    calib = GripAndClubfaceCalibration(
        lead_hand_frame="LH_wrist",
        trail_hand_frame="RH_wrist",
        grip_frame="club_grip",
        face_frame="club_face",
        lead_grip_transform=transform,
        trail_grip_transform=trail_transform,
        clubface_transform=clubface_transform,
        convention="lead_left_trail_right",
    )
    assert calib.convention == "lead_left_trail_right"
    assert bool(calib.calibration_sha256.strip()) is True


def test_trunk_joint_centre_vs_c7_proxy_diagnostics() -> None:
    # True trunk joint centre at (0, 0, 1.2)
    joint_centre = np.array([0.0, 0.0, 1.2])
    # C7 marker is on posterior surface of neck/trunk: offset -0.1 m along x
    c7_offset = np.array([-0.1, 0.0, 0.15])
    c7_pos_t0 = joint_centre + c7_offset

    calc_joint = TrunkDiagnosticCalculator(
        observable_kind=TrunkObservableKind.JOINT_CENTRE
    )
    calc_c7 = TrunkDiagnosticCalculator(
        observable_kind=TrunkObservableKind.C7_PROXY,
        c7_nominal_offset_m=(
            float(c7_offset[0]),
            float(c7_offset[1]),
            float(c7_offset[2]),
        ),
    )

    # Pure axial turn by 60 degrees (pi/3) about z
    theta = np.pi / 3.0
    r_z = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # During axial turn, joint centre remains stationary:
    jc_t1 = joint_centre.copy()
    # C7 skin marker swings in a circular arc:
    c7_pos_t1 = joint_centre + r_z @ c7_offset

    res_jc = calc_joint.compute_residual(
        model_trunk_centre=joint_centre, observed_point=jc_t1
    )
    # Joint centre residual is 0 under pure turn
    assert res_jc == pytest.approx(0.0, abs=1e-9)

    # C7 proxy evaluated without rotation compensation exhibits large apparent translation
    res_c7 = calc_c7.compute_residual(
        model_trunk_centre=joint_centre, observed_point=c7_pos_t1
    )
    # The displacement of C7 from address is ||(R - I) * offset||
    arc_displacement = np.linalg.norm((r_z - np.eye(3)) @ c7_offset)
    assert res_c7 == pytest.approx(arc_displacement, abs=1e-4)
    assert res_c7 > 0.08  # Over 80 mm false trunk shift!


def test_marker_residuals_mm_frozen_against_mutation() -> None:
    """Receipt residuals must be frozen: mutating them after construction is rejected."""
    residuals = {"HeadFront": 2.1, "HeadTop": 1.8, "HeadSide": 2.4}
    receipt = _make_receipt(marker_residuals_mm=residuals)

    sha_before = receipt.receipt_sha256

    # Mutating the caller-owned mapping must not change the receipt's values or hash.
    residuals["HeadFront"] = 999.0
    assert receipt.marker_residuals_mm["HeadFront"] == 2.1
    assert receipt.receipt_sha256 == sha_before

    # Mutating the stored field itself must fail closed.
    with pytest.raises(TypeError):
        receipt.marker_residuals_mm["HeadTop"] = 9.9  # type: ignore[index]


def test_attachment_rotation_composed_into_orientation_residual() -> None:
    """Calibrated attachment rotation must be composed into r_obs before the SO(3) residual.

    Scenario: marker triad glued to the skull rotated by a persistent 30 degree yaw
    relative to the anatomical head frame; physical head pose is a pure axial 45 degree
    turn while the model still believes identity. The reported SO(3) residual must be
    geodesic(r_model, r_head_true) = 45 deg (not the raw marker-frame orientation
    error 75 deg that ignores the attachment mounting).
    """

    def rz(phi: float) -> np.ndarray:
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    nominal = {
        "HeadFront": (0.10, 0.0, 0.0),
        "HeadTop": (0.0, 0.0, 0.12),
        "HeadSide": (0.0, 0.09, 0.0),
    }
    head_centre = np.array([0.0, 0.0, 1.5])
    centroid = np.mean(np.array(list(nominal.values())), axis=0)

    # Declared anatomical -> marker mounting rotation of 30 degrees yaw.
    r_attach = rz(np.pi / 6.0)
    t_head = AttachmentTransform(
        from_frame="head_centre",
        to_frame="marker_set",
        translation_m=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        rotation_matrix=tuple(tuple(float(x) for x in row) for row in r_attach),  # type: ignore[arg-type]
    )
    calc = HeadDiagnosticCalculator(attachment_transform=t_head)

    r_true = rz(np.pi / 4.0)  # physical head pose: pure axial 45 degree turn
    observed = {
        name: head_centre + r_true @ (r_attach @ np.asarray(off))
        for name, off in nominal.items()
    }

    diag = calc.evaluate(
        model_head_centre=head_centre,
        model_head_rotation=np.eye(3),
        observed_markers=observed,
    )

    # Attachment translation keeps the marker-centre estimate exact.
    assert diag.head_centre_error_m == pytest.approx(0.0, abs=1e-6)
    # The residual must reflect the anatomical orientation, not the rotated mounting.
    assert diag.orientation_error_deg == pytest.approx(45.0, abs=1e-3)
    assert diag.orientation_error_rad == pytest.approx(np.pi / 4.0, abs=1e-4)


def test_load_rejects_receipt_missing_receipt_sha256(tmp_path: Path) -> None:
    """Tamper-evident load fails closed when the stored receipt_sha256 is absent."""
    receipt = _make_receipt()
    save_path = tmp_path / "receipt_without_digest.json"
    receipt.save_json(save_path)

    data = json.loads(save_path.read_text(encoding="utf-8"))
    del data["receipt_sha256"]
    save_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt_sha256"):
        ObservabilityDiagnosticReceipt.load_json(save_path)


def test_calibration_transform_frame_endpoints_validated() -> None:
    """Grip/clubface transforms must connect the calibration's declared frames."""
    lead_grip = AttachmentTransform(
        from_frame="LH_wrist",
        to_frame="club_grip",
        translation_m=(0.0, 0.0, 0.1),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    trail_grip = AttachmentTransform(
        from_frame="RH_wrist",
        to_frame="club_grip",
        translation_m=(0.0, 0.0, 0.1),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    clubface = AttachmentTransform(
        from_frame="club_grip",
        to_frame="club_face",
        translation_m=(0.0, 0.0, 0.2),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )

    # Consistent calibration is accepted.
    GripAndClubfaceCalibration(
        lead_hand_frame="LH_wrist",
        trail_hand_frame="RH_wrist",
        grip_frame="club_grip",
        face_frame="club_face",
        lead_grip_transform=lead_grip,
        trail_grip_transform=trail_grip,
        clubface_transform=clubface,
        convention="lead_left_trail_right",
    )

    # Trail transform mislabeled with the lead-hand source frame must fail closed.
    with pytest.raises(ValueError, match="trail_grip_transform"):
        GripAndClubfaceCalibration(
            lead_hand_frame="LH_wrist",
            trail_hand_frame="RH_wrist",
            grip_frame="club_grip",
            face_frame="club_face",
            lead_grip_transform=lead_grip,
            trail_grip_transform=lead_grip,  # from 'LH_wrist', not the trail hand
            clubface_transform=clubface,
            convention="lead_left_trail_right",
        )

    # Clubface transform whose target is not the declared face frame must fail closed.
    with pytest.raises(ValueError, match="clubface_transform"):
        GripAndClubfaceCalibration(
            lead_hand_frame="LH_wrist",
            trail_hand_frame="RH_wrist",
            grip_frame="club_grip",
            face_frame="club_face",
            lead_grip_transform=lead_grip,
            trail_grip_transform=trail_grip,
            clubface_transform=trail_grip,  # to 'club_grip', not 'club_face'
            convention="lead_left_trail_right",
        )


def test_observability_receipt_serialization_roundtrip(tmp_path: Path) -> None:
    t_head = AttachmentTransform(
        from_frame="head_centre",
        to_frame="marker_centroid",
        translation_m=(0.03, 0.02, 0.07),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    grip_tf = AttachmentTransform(
        from_frame="LH_wrist",
        to_frame="club_grip",
        translation_m=(0.0, 0.0, -0.05),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    trail_grip_tf = AttachmentTransform(
        from_frame="RH_wrist",
        to_frame="club_grip",
        translation_m=(0.0, 0.0, -0.05),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    face_tf = AttachmentTransform(
        from_frame="club_grip",
        to_frame="club_face",
        translation_m=(0.0, 0.0, -0.2),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    calib = GripAndClubfaceCalibration(
        lead_hand_frame="LH_wrist",
        trail_hand_frame="RH_wrist",
        grip_frame="club_grip",
        face_frame="club_face",
        lead_grip_transform=grip_tf,
        trail_grip_transform=trail_grip_tf,
        clubface_transform=face_tf,
        convention="lead_left_trail_right",
    )
    receipt = ObservabilityDiagnosticReceipt(
        schema_version=OBSERVABILITY_RECEIPT_SCHEMA,
        head_attachment=t_head,
        grip_calibration=calib,
        trunk_observable=TrunkObservableKind.JOINT_CENTRE,
        head_centre_error_m=0.002,
        head_orientation_error_deg=3.4,
        marker_residuals_mm={"HeadFront": 2.1, "HeadTop": 1.8, "HeadSide": 2.4},
        trunk_com_residual_m=0.004,
    )

    save_path = tmp_path / "observability_receipt.json"
    receipt.save_json(save_path)

    loaded = ObservabilityDiagnosticReceipt.load_json(save_path)
    assert loaded.schema_version == receipt.schema_version
    assert loaded.head_centre_error_m == pytest.approx(receipt.head_centre_error_m)
    assert loaded.head_orientation_error_deg == pytest.approx(
        receipt.head_orientation_error_deg
    )
    assert loaded.marker_residuals_mm == receipt.marker_residuals_mm
    assert loaded.trunk_com_residual_m == pytest.approx(receipt.trunk_com_residual_m)
    assert loaded.receipt_sha256 == receipt.receipt_sha256

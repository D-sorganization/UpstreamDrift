"""Unit and behavioral regression tests for Head, Trunk, and Grip Diagnostic Receipts.

Specified in MMR-06-I (#11106):
1. Pure axial turn: head-centre translation remains zero while SO(3) orientation error is non-zero.
2. Marker-centroid / body-centre comparison fails closed without an explicit attachment transform.
3. Left/right frame conventions and rigid-transform roundtrips (T^-1 * T == I).
4. Emission of head-marker residuals, SO(3) angle residuals, and grip/clubface transforms with units and SHA-256 hashes.
5. Trunk observable: joint-centre frame vs C7 skin-marker proxy.
6. Receipt serialization, content hashes, and tamper verification.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def _make_receipt(
    marker_residuals_mm: dict[str, float] | None = None,
) -> ObservabilityDiagnosticReceipt:
    """Build a canonical diagnostic receipt for persistence tests."""
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


class TestHeadDiagnosticReceipts:
    """Tests head diagnostic receipts and pure axial turn contracts."""

    def test_pure_axial_turn_head_centre_translation_zero_while_orientation_error_nonzero(
        self,
    ) -> None:
        """Pure axial yaw turn produces zero head-centre translation error and exact SO(3) orientation error."""
        head_centre_true = np.array([0.0, 0.0, 1.5])
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
        r_identity = np.eye(3)

        theta = np.pi / 4.0  # 45 deg axial yaw turn
        r_yaw = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        markers_t1 = {
            "HeadFront": head_centre_true + r_yaw @ offset_front,
            "HeadTop": head_centre_true + r_yaw @ offset_top,
            "HeadSide": head_centre_true + r_yaw @ offset_side,
        }

        diag = calc.evaluate(
            model_head_centre=head_centre_true,
            model_head_rotation=r_identity,
            observed_markers=markers_t1,
        )

        assert isinstance(diag, HeadDiagnosticResult)
        assert diag.head_centre_error_m == pytest.approx(0.0, abs=1e-6)
        assert diag.orientation_error_deg == pytest.approx(45.0, abs=1e-3)
        assert diag.orientation_error_rad == pytest.approx(np.pi / 4.0, abs=1e-4)
        assert diag.marker_residuals_m["HeadFront"] > 0.05
        assert diag.marker_residuals_m["HeadSide"] > 0.05

    def test_marker_centroid_without_attachment_transform_fails_closed(self) -> None:
        """Comparing marker centroid to body centre without attachment transform fails closed."""
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

    def test_attachment_rotation_composed_into_orientation_residual(self) -> None:
        """Attachment rotation is composed into r_obs before calculating the SO(3) residual."""

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

        # 30 deg mounting yaw
        r_attach = rz(np.pi / 6.0)
        t_head = AttachmentTransform(
            from_frame="head_centre",
            to_frame="marker_set",
            translation_m=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            rotation_matrix=tuple(tuple(float(x) for x in row) for row in r_attach),  # type: ignore[arg-type]
        )
        calc = HeadDiagnosticCalculator(attachment_transform=t_head)

        # 45 deg physical head turn
        r_true = rz(np.pi / 4.0)
        observed = {
            name: head_centre + r_true @ (r_attach @ np.asarray(off))
            for name, off in nominal.items()
        }

        diag = calc.evaluate(
            model_head_centre=head_centre,
            model_head_rotation=np.eye(3),
            observed_markers=observed,
        )

        assert diag.head_centre_error_m == pytest.approx(0.0, abs=1e-6)
        assert diag.orientation_error_deg == pytest.approx(45.0, abs=1e-3)
        assert diag.orientation_error_rad == pytest.approx(np.pi / 4.0, abs=1e-4)


class TestGripAndRigidTransformRoundtrips:
    """Tests SE(3) transform algebra, left/right frame conventions, and roundtrips."""

    def test_left_right_frame_conventions_and_rigid_transform_roundtrips(self) -> None:
        """AttachmentTransform roundtrip identity holds: T^-1 * T == I."""
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

        p_test = np.array([0.2, -0.1, 0.8])
        p_forward = transform.apply(p_test)
        inv_transform = transform.inverse()
        p_back = inv_transform.apply(p_forward)

        np.testing.assert_allclose(p_back, p_test, atol=1e-12)
        assert transform.roundtrip_identity(p_test) is True

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

    def test_calibration_transform_frame_endpoints_validated(self) -> None:
        """GripAndClubfaceCalibration validates frame endpoints and fails closed."""
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

        # Consistent calibration accepted
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

        # Mismatched trail grip frame fails closed
        with pytest.raises(ValueError, match="trail_grip_transform"):
            GripAndClubfaceCalibration(
                lead_hand_frame="LH_wrist",
                trail_hand_frame="RH_wrist",
                grip_frame="club_grip",
                face_frame="club_face",
                lead_grip_transform=lead_grip,
                trail_grip_transform=lead_grip,
                clubface_transform=clubface,
            )


class TestTrunkDiagnosticCalculator:
    """Tests trunk observable comparisons: joint centre vs C7 proxy."""

    def test_trunk_joint_centre_vs_c7_proxy_diagnostics(self) -> None:
        """C7 skin marker proxy undergoes large displacement under axial turn, while joint-centre observable remains invariant."""
        joint_centre = np.array([0.0, 0.0, 1.2])
        c7_offset = np.array([-0.1, 0.0, 0.15])

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

        theta = np.pi / 3.0  # 60 deg axial turn
        r_z = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        jc_t1 = joint_centre.copy()
        c7_pos_t1 = joint_centre + r_z @ c7_offset

        res_jc = calc_joint.compute_residual(
            model_trunk_centre=joint_centre, observed_point=jc_t1
        )
        assert res_jc == pytest.approx(0.0, abs=1e-9)

        res_c7 = calc_c7.compute_residual(
            model_trunk_centre=joint_centre, observed_point=c7_pos_t1
        )
        arc_displacement = np.linalg.norm((r_z - np.eye(3)) @ c7_offset)
        assert res_c7 == pytest.approx(arc_displacement, abs=1e-4)
        assert res_c7 > 0.08


class TestObservabilityReceiptPersistence:
    """Tests serialization, SHA-256 provenance hashes, and immutability."""

    def test_receipt_serialization_roundtrip_and_tamper_evidence(
        self, tmp_path: Path
    ) -> None:
        """Receipt serializes to JSON, validates SHA-256 digest, and detects tampering."""
        receipt = _make_receipt()
        save_path = tmp_path / "receipt.json"

        receipt.save_json(save_path)
        assert save_path.exists()

        loaded = ObservabilityDiagnosticReceipt.load_json(save_path)
        assert loaded.schema_version == OBSERVABILITY_RECEIPT_SCHEMA
        assert loaded.receipt_sha256 == receipt.receipt_sha256
        assert loaded.head_centre_error_m == receipt.head_centre_error_m
        assert loaded.head_orientation_error_deg == receipt.head_orientation_error_deg

        with open(save_path, encoding="utf-8") as f:
            data = json.load(f)
        data["head_centre_error_m"] = 0.0999
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="Receipt SHA256 mismatch"):
            ObservabilityDiagnosticReceipt.load_json(save_path)

    def test_receipt_rejects_missing_hash(self, tmp_path: Path) -> None:
        """Loading a receipt missing receipt_sha256 fails closed."""
        receipt = _make_receipt()
        save_path = tmp_path / "no_hash_receipt.json"
        receipt.save_json(save_path)

        data = json.loads(save_path.read_text(encoding="utf-8"))
        del data["receipt_sha256"]
        save_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        with pytest.raises(ValueError, match="receipt_sha256"):
            ObservabilityDiagnosticReceipt.load_json(save_path)

    def test_marker_residuals_mm_frozen_against_mutation(self) -> None:
        """Receipt residuals mapping is frozen against mutation."""
        residuals = {"HeadFront": 2.1, "HeadTop": 1.8, "HeadSide": 2.4}
        receipt = _make_receipt(marker_residuals_mm=residuals)

        sha_before = receipt.receipt_sha256
        residuals["HeadFront"] = 999.0
        assert receipt.marker_residuals_mm["HeadFront"] == 2.1
        assert receipt.receipt_sha256 == sha_before

        with pytest.raises(TypeError):
            receipt.marker_residuals_mm["HeadTop"] = 9.9  # type: ignore[index]

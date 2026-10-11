"""Unit tests for COV-9 3D comparison receipts and depth-error isolation (#11277)."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.motion_capture.reference.comparison_3d import (
    AnthropometryAblationResult,
    Comparison3DLevel,
    Comparison3DReceipt,
    JointErrorSummary,
    L1_3DComparisonResult,
    L3ComparisonResult,
    align_trajectories_rigid_fixed_scale,
    build_3d_comparison_receipt,
    compute_anthropometry_ablation,
    compute_depth_and_image_plane_errors,
    compute_l1_3d_envelope_comparison,
    compute_l3_paired_comparison,
    compute_mpjpe_and_pa_mpjpe,
    validate_laterality,
)


@pytest.fixture
def synthetic_landmark_names() -> tuple[str, ...]:
    """Canonical test joint names."""
    return (
        "pelvis",
        "thorax",
        "shoulder_left",
        "shoulder_right",
        "elbow_left",
        "elbow_right",
        "wrist_left",
        "wrist_right",
        "hip_left",
        "hip_right",
    )


@pytest.fixture
def synthetic_reference_motion(synthetic_landmark_names: tuple[str, ...]) -> np.ndarray:
    """Synthetic 3D reference motion (T=50 frames, K=10 joints, 3 coords) in meters."""
    t_frames = 50
    k_joints = len(synthetic_landmark_names)
    rng = np.random.default_rng(11277)

    # Base canonical positions (standing upright facing +X, Y is up, Z is right in ADR-0041)
    base_pos = np.zeros((k_joints, 3), dtype=float)
    base_pos[0] = [0.0, 0.9, 0.0]  # pelvis
    base_pos[1] = [0.0, 1.3, 0.0]  # thorax
    base_pos[2] = [0.0, 1.4, -0.2]  # shoulder_left (Z < 0)
    base_pos[3] = [0.0, 1.4, 0.2]  # shoulder_right (Z > 0)
    base_pos[4] = [0.1, 1.1, -0.25]  # elbow_left
    base_pos[5] = [0.1, 1.1, 0.25]  # elbow_right
    base_pos[6] = [0.2, 0.8, -0.1]  # wrist_left
    base_pos[7] = [0.2, 0.8, 0.1]  # wrist_right
    base_pos[8] = [0.0, 0.85, -0.15]  # hip_left
    base_pos[9] = [0.0, 0.85, 0.15]  # hip_right

    trajectory = np.zeros((t_frames, k_joints, 3), dtype=float)
    phases = np.linspace(0.0, 2.0 * np.pi, t_frames)
    for t_idx, phase in enumerate(phases):
        trajectory[t_idx] = base_pos.copy()
        # Add smooth swinging motion
        trajectory[t_idx, :, 0] += 0.2 * np.sin(phase)
        trajectory[t_idx, :, 1] += 0.05 * np.cos(phase)
        trajectory[t_idx, :, 2] += 0.1 * np.sin(phase)

    return trajectory


class TestCOV9_3DComparison:
    """TDD unit tests proving COV-9 contracts per issue #11277."""

    def test_depth_axis_offset_isolated_from_image_plane(
        self,
        synthetic_reference_motion: np.ndarray,
    ) -> None:
        """A pure depth-axis offset of 50 mm is reported in the depth component;

        image-plane component is ~ 0.
        """
        depth_axis = np.array([0.0, 0.0, 1.0])  # Optical axis along +Z
        offset_m = 0.050  # 50 mm along depth axis

        perturbed = synthetic_reference_motion.copy()
        perturbed[:, :, 2] += offset_m  # pure +Z offset

        depth_rmse_mm, image_plane_rmse_mm = compute_depth_and_image_plane_errors(
            perturbed - synthetic_reference_motion,
            depth_axis=depth_axis,
        )

        assert pytest.approx(depth_rmse_mm, abs=1e-3) == 50.0
        assert pytest.approx(image_plane_rmse_mm, abs=1e-3) == 0.0

    def test_uniform_scale_error_invisible_in_pa_mpjpe_visible_in_mpjpe(
        self,
        synthetic_reference_motion: np.ndarray,
    ) -> None:
        """A uniform scale error is invisible in PA-MPJPE but visible in MPJPE (scale fixed).

        Both numbers are asserted.
        """
        scale_factor = 1.15  # 15% scale error
        scaled_predicted = synthetic_reference_motion * scale_factor

        mpjpe_mm, pa_mpjpe_mm, _, _ = compute_mpjpe_and_pa_mpjpe(
            scaled_predicted,
            synthetic_reference_motion,
        )

        # Scale error is visible in fixed-scale MPJPE (> 50 mm)
        assert mpjpe_mm > 50.0
        # Scale error is eliminated by Procrustes optimal scale in PA-MPJPE (~ 0 mm)
        assert pytest.approx(pa_mpjpe_mm, abs=1e-3) == 0.0

    def test_swapped_left_right_joints_detected_and_rejected(
        self,
        synthetic_reference_motion: np.ndarray,
        synthetic_landmark_names: tuple[str, ...],
    ) -> None:
        """Swapped left/right joints -> detected by laterality test and rejected, not averaged."""
        left_idx = [
            synthetic_landmark_names.index("shoulder_left"),
            synthetic_landmark_names.index("hip_left"),
        ]
        right_idx = [
            synthetic_landmark_names.index("shoulder_right"),
            synthetic_landmark_names.index("hip_right"),
        ]

        # Coronal check passes on un-swapped data
        validate_laterality(
            synthetic_reference_motion,
            synthetic_reference_motion,
            left_indices=left_idx,
            right_indices=right_idx,
        )

        # Create swapped joints in predicted data
        swapped = synthetic_reference_motion.copy()
        for l_i, r_i in zip(left_idx, right_idx, strict=True):
            swapped[:, [l_i, r_i]] = swapped[:, [r_i, l_i]]

        with pytest.raises(ValueError, match="[Ll]aterality.*swap"):
            validate_laterality(
                swapped,
                synthetic_reference_motion,
                left_indices=left_idx,
                right_indices=right_idx,
            )

    def test_physical_clock_unknown_emits_no_velocity_metrics(
        self,
        synthetic_reference_motion: np.ndarray,
        synthetic_landmark_names: tuple[str, ...],
    ) -> None:
        """physical_clock == unknown -> no velocity metric emitted (contract test)."""
        result = compute_l3_paired_comparison(
            predicted_trajectories=synthetic_reference_motion,
            reference_trajectories=synthetic_reference_motion,
            joint_names=synthetic_landmark_names,
            physical_clock="unknown",
            video_swing_id="cov-01-s1",
            paired_capture_swing_id="capture-O-s1",
        )

        assert result.physical_clock == "unknown"
        assert result.velocity_metrics is None

        # When physical_clock == "known", velocity metrics are emitted
        result_known = compute_l3_paired_comparison(
            predicted_trajectories=synthetic_reference_motion,
            reference_trajectories=synthetic_reference_motion,
            joint_names=synthetic_landmark_names,
            physical_clock="known",
            fps=120.0,
            video_swing_id="cov-01-s1",
            paired_capture_swing_id="capture-O-s1",
        )
        assert result_known.physical_clock == "known"
        assert result_known.velocity_metrics is not None
        assert "peak_joint_speeds_m_s" in result_known.velocity_metrics

    def test_necromatcher_architecture_lod_contract(self) -> None:
        """Necromatcher comparisons go through native FK landmark output only;

        there is no import of solver internals (architecture LoD test).
        """
        source_path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "motion_capture"
            / "reference"
            / "comparison_3d.py"
        )
        assert source_path.exists(), f"comparison_3d.py must exist at {source_path}"

        source_code = source_path.read_text(encoding="utf-8")
        parsed = ast.parse(source_code)

        disallowed_substrings = (
            "necromatcher_solver",
            "necromatcher_core",
            "optimizer",
            "q_vector",
            "generalized_coordinates",
            "ik_solver",
        )

        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for dis in disallowed_substrings:
                        assert dis not in alias.name.lower(), (
                            f"Illegal import of solver internal {alias.name} violating LoD"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                for dis in disallowed_substrings:
                    assert dis not in node.module.lower(), (
                        f"Illegal from-import of solver internal {node.module} violating LoD"
                    )

    def test_input_hash_binding_and_stale_refusal(
        self,
        synthetic_reference_motion: np.ndarray,
        synthetic_landmark_names: tuple[str, ...],
    ) -> None:
        """Every input hash is bound in the receipt; a stale input raises ValueError."""
        l3_res = compute_l3_paired_comparison(
            predicted_trajectories=synthetic_reference_motion,
            reference_trajectories=synthetic_reference_motion,
            joint_names=synthetic_landmark_names,
            physical_clock="unknown",
            video_swing_id="cov-01-s1",
            paired_capture_swing_id="capture-O-s1",
        )

        ref_hash = hashlib.sha256(synthetic_reference_motion.tobytes()).hexdigest()
        pred_hash = hashlib.sha256(synthetic_reference_motion.tobytes()).hexdigest()
        pairing_hash = hashlib.sha256(b"cov-01-s1:capture-O-s1").hexdigest()

        # Valid receipt creation
        receipt = build_3d_comparison_receipt(
            video_swing_id="cov-01-s1",
            backend="necromatcher_anchored",
            level=Comparison3DLevel.L3,
            reference_hash=ref_hash,
            predicted_hash=pred_hash,
            pairing_hash=pairing_hash,
            l3_result=l3_res,
        )
        assert isinstance(receipt, Comparison3DReceipt)
        assert receipt.reference_hash == ref_hash
        assert receipt.predicted_hash == pred_hash
        assert receipt.pairing_hash == pairing_hash

        # Tampered or invalid hash raises ValueError
        with pytest.raises(
            ValueError, match="reference_hash must be a valid 64-char SHA-256"
        ):
            build_3d_comparison_receipt(
                video_swing_id="cov-01-s1",
                backend="necromatcher_anchored",
                level=Comparison3DLevel.L3,
                reference_hash="stale_or_invalid_hash",
                predicted_hash=pred_hash,
                pairing_hash=pairing_hash,
                l3_result=l3_res,
            )

    def test_anthropometry_ablation_computation(
        self,
        synthetic_reference_motion: np.ndarray,
        synthetic_landmark_names: tuple[str, ...],
    ) -> None:
        """Anthropometry ablation computes anchored-minus-generic error cleanly."""
        # Anchored fit has smaller body-size error (1% scale distortion)
        anchored_motion = synthetic_reference_motion * 1.01
        # Generic fit has larger error due to generic prior body-size mismatch (8% scale distortion)
        generic_motion = synthetic_reference_motion * 1.08

        res_anchored = compute_l3_paired_comparison(
            predicted_trajectories=anchored_motion,
            reference_trajectories=synthetic_reference_motion,
            joint_names=synthetic_landmark_names,
            video_swing_id="cov-01-s1",
            paired_capture_swing_id="capture-O-s1",
            backend="necromatcher_anchored",
        )
        res_generic = compute_l3_paired_comparison(
            predicted_trajectories=generic_motion,
            reference_trajectories=synthetic_reference_motion,
            joint_names=synthetic_landmark_names,
            video_swing_id="cov-01-s1",
            paired_capture_swing_id="capture-O-s1",
            backend="necromatcher_generic",
        )

        ablation = compute_anthropometry_ablation(res_anchored, res_generic)
        assert isinstance(ablation, AnthropometryAblationResult)
        assert ablation.anchored_error_mm < ablation.generic_error_mm
        # delta = anchored - generic (negative because anchored is better)
        assert ablation.delta_error_mm < 0.0
        assert ablation.fraction_due_to_body_size > 0.0

    def test_l1_3d_envelope_fraction_inside(
        self,
        synthetic_reference_motion: np.ndarray,
        synthetic_landmark_names: tuple[str, ...],
    ) -> None:
        """L1-3D evaluates trajectory inside fraction against a 13-swing variation envelope."""
        t_len, k_len, _ = synthetic_reference_motion.shape
        # Create synthetic 3D envelope (p5 and p95 bounds per frame and joint)
        envelope_p5 = synthetic_reference_motion - 0.050
        envelope_p95 = synthetic_reference_motion + 0.050

        # Motion entirely inside
        res_inside = compute_l1_3d_envelope_comparison(
            synthetic_reference_motion,
            envelope_p5=envelope_p5,
            envelope_p95=envelope_p95,
            joint_names=synthetic_landmark_names,
            video_swing_id="cov-02-unpaired",
            backend="hmr2",
        )
        assert isinstance(res_inside, L1_3DComparisonResult)
        assert pytest.approx(res_inside.fraction_inside_envelope, abs=1e-3) == 1.0

        # Motion with 50% frames outside (e.g. offset > 0.050)
        perturbed = synthetic_reference_motion.copy()
        perturbed[: t_len // 2] += 0.100  # first half outside
        res_half = compute_l1_3d_envelope_comparison(
            perturbed,
            envelope_p5=envelope_p5,
            envelope_p95=envelope_p95,
            joint_names=synthetic_landmark_names,
            video_swing_id="cov-02-unpaired",
            backend="hmr2",
        )
        assert pytest.approx(res_half.fraction_inside_envelope, abs=1e-2) == 0.5

    def test_l1_3d_receipt_building(
        self,
        synthetic_reference_motion: np.ndarray,
        synthetic_landmark_names: tuple[str, ...],
    ) -> None:
        """Verify building L1_3D receipt requires l1_3d_result and validates hashes."""
        envelope_p5 = synthetic_reference_motion - 0.050
        envelope_p95 = synthetic_reference_motion + 0.050
        res = compute_l1_3d_envelope_comparison(
            synthetic_reference_motion,
            envelope_p5=envelope_p5,
            envelope_p95=envelope_p95,
            joint_names=synthetic_landmark_names,
            video_swing_id="cov-02-unpaired",
            backend="hmr2",
        )
        h1 = hashlib.sha256(b"ref").hexdigest()
        h2 = hashlib.sha256(b"pred").hexdigest()
        h3 = hashlib.sha256(b"pairing").hexdigest()

        receipt = build_3d_comparison_receipt(
            video_swing_id="cov-02-unpaired",
            backend="hmr2",
            level=Comparison3DLevel.L1_3D,
            reference_hash=h1,
            predicted_hash=h2,
            pairing_hash=h3,
            l1_3d_result=res,
        )
        assert receipt.level == Comparison3DLevel.L1_3D
        assert receipt.l1_3d_result is not None

        # Missing l1_3d_result for L1_3D level raises
        with pytest.raises(ValueError, match="l1_3d_result must be provided"):
            build_3d_comparison_receipt(
                video_swing_id="cov-02-unpaired",
                backend="hmr2",
                level=Comparison3DLevel.L1_3D,
                reference_hash=h1,
                predicted_hash=h2,
                pairing_hash=h3,
            )

    def test_rigid_fixed_scale_alignment_address_subset(
        self,
        synthetic_reference_motion: np.ndarray,
    ) -> None:
        """Address subset alignment transforms predicted points with fixed scale=1.0."""
        # Perturb with pure rigid rotation + translation
        rot = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        t_vec = np.array([0.5, -0.2, 1.0])
        perturbed = synthetic_reference_motion @ rot.T + t_vec

        aligned, transform = align_trajectories_rigid_fixed_scale(
            perturbed,
            synthetic_reference_motion,
            address_indices=[0, 1, 2],
        )
        assert transform.scale == 1.0
        assert transform.body_size_normalized is False
        assert pytest.approx(aligned[0], abs=1e-4) == synthetic_reference_motion[0]

    def test_invalid_shapes_raise_under_dbc(
        self,
        synthetic_reference_motion: np.ndarray,
        synthetic_landmark_names: tuple[str, ...],
    ) -> None:
        """Mismatched trajectory shapes and dimensions fail closed under DbC."""
        with pytest.raises(ValueError, match="match shape"):
            compute_l3_paired_comparison(
                predicted_trajectories=synthetic_reference_motion[:10],
                reference_trajectories=synthetic_reference_motion,
                joint_names=synthetic_landmark_names,
            )

        with pytest.raises(ValueError, match="landmark dimension"):
            compute_l3_paired_comparison(
                predicted_trajectories=synthetic_reference_motion,
                reference_trajectories=synthetic_reference_motion,
                joint_names=synthetic_landmark_names[:3],  # mismatch count
            )

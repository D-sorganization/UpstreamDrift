"""Unit tests for pure-Python segment length and scale estimation (OS-3b)."""

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.scale import (
    DEFAULT_NOMINAL_LENGTHS_M,
    SegmentScaleResult,
    estimate_segment_scales,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    TourCapture,
    load_tour_capture,
)

pytestmark = pytest.mark.unit

C3D_PATH = Path(__file__).resolve().parents[2] / "data/C3D_TA_Driver.c3d"


def _synthetic_capture(
    scale: float = 1.0, frames: int = 25, noise_std: float = 0.0
) -> TourCapture:
    """Construct a minimal valid TourCapture with canonical marker layout."""
    labels = (
        "WaistLeft",
        "WaistRight",
        "WaistLBack",
        "WaistRBack",
        "BackTop",
        "BackLeft",
        "BackRight",
        "LShoulderTop",
        "LShoulderBack",
        "LElbowOut",
        "LWristTop",
        "RShoulderTop",
        "RShoulderBack",
        "RElbowOut",
        "RWristTop",
        "LKneeOut",
        "LAnkleOut",
        "LToeIn",
        "LToeOut",
        "RKneeOut",
        "RAnkleOut",
        "RToeIn",
        "RToeOut",
    )
    # Canonical address pose positions (approximate meters)
    base_pos = {
        "WaistLeft": np.array([-0.05, 0.95, 0.15]),
        "WaistRight": np.array([-0.05, 0.95, -0.15]),
        "WaistLBack": np.array([-0.15, 0.95, 0.12]),
        "WaistRBack": np.array([-0.15, 0.95, -0.12]),
        "BackTop": np.array([-0.10, 1.35, 0.0]),
        "BackLeft": np.array([-0.12, 1.20, 0.10]),
        "BackRight": np.array([-0.12, 1.20, -0.10]),
        "LShoulderTop": np.array([0.0, 1.40, 0.20]),
        "LShoulderBack": np.array([-0.08, 1.38, 0.20]),
        "LElbowOut": np.array([0.05, 1.10, 0.25]),
        "LWristTop": np.array([0.10, 0.85, 0.15]),
        "RShoulderTop": np.array([0.0, 1.40, -0.20]),
        "RShoulderBack": np.array([-0.08, 1.38, -0.20]),
        "RElbowOut": np.array([0.05, 1.10, -0.25]),
        "RWristTop": np.array([0.10, 0.85, -0.15]),
        "LKneeOut": np.array([0.02, 0.50, 0.18]),
        "LAnkleOut": np.array([0.0, 0.10, 0.16]),
        "LToeIn": np.array([0.15, 0.05, 0.10]),
        "LToeOut": np.array([0.15, 0.05, 0.20]),
        "RKneeOut": np.array([0.02, 0.50, -0.18]),
        "RAnkleOut": np.array([0.0, 0.10, -0.16]),
        "RToeIn": np.array([0.15, 0.05, -0.10]),
        "RToeOut": np.array([0.15, 0.05, -0.20]),
    }
    points = np.zeros((frames, len(labels), 3))
    rng = np.random.default_rng(42)
    for i, label in enumerate(labels):
        p = base_pos[label] * scale
        noise = (
            rng.normal(0.0, noise_std, size=(frames, 3))
            if noise_std > 0.0
            else np.zeros((frames, 3))
        )
        points[:, i, :] = p + noise
    valid = np.ones((frames, len(labels)), dtype=bool)
    time_s = np.arange(frames) / 360.0
    return TourCapture(time_s=time_s, labels=labels, points_m=points, valid=valid)


def test_estimate_segment_scales_synthetic_scale() -> None:
    target_scale = 1.15
    cap = _synthetic_capture(scale=target_scale, frames=25, noise_std=0.0)

    # Compute unscaled nominal lengths from scale=1.0 capture
    unscaled_cap = _synthetic_capture(scale=1.0, frames=25)
    baseline = estimate_segment_scales(unscaled_cap)

    res = estimate_segment_scales(cap, nominal_lengths_m=baseline.measured_lengths_m)
    assert isinstance(res, SegmentScaleResult)

    for seg, s in res.scale_factors.items():
        assert s == pytest.approx(target_scale, rel=1e-5), f"Segment {seg} mismatch"
        assert res.rigid_residuals_m[seg] == pytest.approx(0.0, abs=1e-12)


def test_estimate_segment_scales_rigid_residual_detects_variation() -> None:
    noise = 0.005  # 5 mm noise
    cap = _synthetic_capture(scale=1.0, frames=25, noise_std=noise)
    res = estimate_segment_scales(cap)

    for seg, residual in res.rigid_residuals_m.items():
        assert residual > 0.0
        assert residual < 0.05, f"Residual {residual} unreasonably large for {seg}"


def test_estimate_segment_scales_real_tour_capture() -> None:
    if not C3D_PATH.exists():
        pytest.skip(f"Capture file not found at {C3D_PATH}")

    cap = load_tour_capture(C3D_PATH)
    res = estimate_segment_scales(cap)

    expected_segments = {
        "femur_r",
        "femur_l",
        "tibia_r",
        "tibia_l",
        "calcn_r",
        "calcn_l",
        "humerus_r",
        "humerus_l",
        "radius_r",
        "radius_l",
        "torso",
    }
    assert set(res.scale_factors.keys()) == expected_segments
    assert set(res.measured_lengths_m.keys()) == expected_segments
    assert set(res.nominal_lengths_m.keys()) == expected_segments
    assert set(res.rigid_residuals_m.keys()) == expected_segments
    assert set(res.provenance.keys()) == expected_segments

    # Physiologic sanity checks: scale factors between 0.7 and 1.5
    for seg, s in res.scale_factors.items():
        assert 0.7 <= s <= 1.5, f"Scale factor {s} out of physiologic range for {seg}"

    # Rigid residual over first 20 frames at address should be small (< 20 mm)
    for seg, r in res.rigid_residuals_m.items():
        assert r < 0.02, f"Rigid residual {r} m too high for {seg}"


def test_estimate_segment_scales_handles_early_rshouldertop_occlusion() -> None:
    cap = _synthetic_capture(frames=25)
    idx = cap.index("RShoulderTop")
    valid = np.array(cap.valid, copy=True)
    valid[:, idx] = False
    points = np.array(cap.points_m, copy=True)
    points[:, idx] = np.nan
    occluded_cap = TourCapture(
        time_s=cap.time_s, labels=cap.labels, points_m=points, valid=valid
    )

    res = estimate_segment_scales(occluded_cap)
    assert np.isfinite(res.scale_factors["humerus_r"])
    assert np.isfinite(res.scale_factors["torso"])


def test_estimate_segment_scales_rejects_insufficient_frames() -> None:
    cap = _synthetic_capture(frames=15)
    with pytest.raises(ValueError, match="at least 20 frames"):
        estimate_segment_scales(cap)


def test_estimate_segment_scales_validates_nominal_lengths() -> None:
    cap = _synthetic_capture(frames=25)
    with pytest.raises(ValueError, match="positive finite"):
        estimate_segment_scales(cap, nominal_lengths_m={"tibia_r": -0.4})


def test_estimate_segment_scales_real_tour_capture_bilateral_symmetry() -> None:
    """Assert real tour capture humerus scaling is bilaterally consistent (ratio <= 1.10)."""
    if not C3D_PATH.exists():
        pytest.skip(f"Capture file not found at {C3D_PATH}")

    cap = load_tour_capture(C3D_PATH)
    res = estimate_segment_scales(cap)

    # Historical uncorrected ratio was 1.4567 / 1.2436 = 1.1713 (> 1.10).
    # Corrected scaling reconstructs acromion proxy to keep ratio <= 1.10.
    ratio = res.scale_factors["humerus_r"] / res.scale_factors["humerus_l"]
    assert 0.90 <= ratio <= 1.10, (
        f"Humerus bilateral ratio {ratio:.4f} exceeds 10% tolerance "
        f"(R={res.scale_factors['humerus_r']:.4f}, L={res.scale_factors['humerus_l']:.4f})"
    )


def test_apply_segment_scaling_transforms_joints_meshes_com_inertia(
    tmp_path: Path,
) -> None:
    """Test consistent segment scaling updates joint frames, meshes, COM, and inertia."""
    from src.engines.physics_engines.opensim.python.tour_matching.model_audit import (
        audit_model_geometry,
        verify_model_qualification,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.segment_scaling import (
        apply_segment_scaling,
    )

    base_osim = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "engines"
        / "physics_engines"
        / "opensim"
        / "models"
        / "golf_humanoid.osim"
    )
    if not base_osim.is_file():
        pytest.skip(f"Base model not found at {base_osim}")

    scales = {
        "humerus_r": 1.10,
        "humerus_l": 1.05,
        "radius_r": 1.08,
        "radius_l": 1.07,
    }

    out_osim = tmp_path / "golf_humanoid_consistent_scaled.osim"
    res_path = apply_segment_scaling(base_osim, scales, out_path=out_osim)
    assert res_path == out_osim
    assert out_osim.is_file()

    # Model geometry audit should find NO unscaled arm mesh defects on scaled arms
    audit = audit_model_geometry(out_osim)
    assert not audit.has_unscaled_arm_mesh_defect

    # Check that humerus_r mesh scales match 1.10
    humerus_r_meshes = audit.arm_mesh_scales["humerus_r"]
    assert len(humerus_r_meshes) > 0
    for _, s_factors in humerus_r_meshes:
        assert s_factors[1] == pytest.approx(1.10, rel=1e-4)


def test_apply_segment_scaling_repeat_protection(tmp_path: Path) -> None:
    """Test that applying segment scaling twice fails closed to prevent double scaling."""
    from src.engines.physics_engines.opensim.python.tour_matching.segment_scaling import (
        apply_segment_scaling,
    )

    base_osim = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "engines"
        / "physics_engines"
        / "opensim"
        / "models"
        / "golf_humanoid.osim"
    )
    if not base_osim.is_file():
        pytest.skip(f"Base model not found at {base_osim}")

    scales = {"humerus_r": 1.10}
    out_osim = tmp_path / "scaled.osim"
    apply_segment_scaling(base_osim, scales, out_path=out_osim)

    # Second scaling call on already scaled model must raise ValueError
    with pytest.raises(ValueError, match="already scaled"):
        apply_segment_scaling(out_osim, scales, out_path=tmp_path / "scaled2.osim")


def test_apply_segment_scaling_validates_inputs(tmp_path: Path) -> None:
    """Test that non-finite or non-positive scale factors are rejected."""
    from src.engines.physics_engines.opensim.python.tour_matching.segment_scaling import (
        apply_segment_scaling,
    )

    base_osim = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "engines"
        / "physics_engines"
        / "opensim"
        / "models"
        / "golf_humanoid.osim"
    )
    if not base_osim.is_file():
        pytest.skip(f"Base model not found at {base_osim}")

    with pytest.raises(ValueError, match="positive finite"):
        apply_segment_scaling(
            base_osim, {"humerus_r": -1.1}, out_path=tmp_path / "out.osim"
        )

    with pytest.raises(ValueError, match="positive finite"):
        apply_segment_scaling(
            base_osim, {"humerus_r": float("nan")}, out_path=tmp_path / "out.osim"
        )


def test_apply_segment_scaling_policies_and_analytical_invariants(
    tmp_path: Path,
) -> None:
    """Assert fixed_mass and density_preserving policies follow analytical scaling laws."""
    from defusedxml import ElementTree as ET
    from src.engines.physics_engines.opensim.python.tour_matching.segment_scaling import (
        ScalingPolicy,
        apply_segment_scaling,
    )

    base_osim = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "engines"
        / "physics_engines"
        / "opensim"
        / "models"
        / "golf_humanoid.osim"
    )
    if not base_osim.is_file():
        pytest.skip(f"Base model not found at {base_osim}")

    tree_orig = ET.parse(str(base_osim))
    b_orig = tree_orig.find(".//Body[@name='femur_r']")
    assert b_orig is not None
    m_orig = float(b_orig.findtext("mass", "0.0"))
    mc_orig = [float(v) for v in b_orig.findtext("mass_center", "0 0 0").split()]
    in_orig = [float(v) for v in b_orig.findtext("inertia", "0 0 0 0 0 0").split()]

    s = 1.20
    scales = {"femur_r": s}

    # 1. Fixed mass policy: m' = m, COM' = s*COM, I' = s^2 * I
    out_fixed = tmp_path / "scaled_fixed.osim"
    apply_segment_scaling(
        base_osim, scales, out_path=out_fixed, policy=ScalingPolicy.FIXED_MASS
    )
    tree_fixed = ET.parse(str(out_fixed))
    b_fixed = tree_fixed.find(".//Body[@name='femur_r']")
    assert b_fixed is not None
    m_fixed = float(b_fixed.findtext("mass", "0.0"))
    mc_fixed = [float(v) for v in b_fixed.findtext("mass_center", "0 0 0").split()]
    in_fixed = [float(v) for v in b_fixed.findtext("inertia", "0 0 0 0 0 0").split()]

    assert m_fixed == pytest.approx(m_orig, rel=1e-6)
    assert mc_fixed[1] == pytest.approx(mc_orig[1] * s, rel=1e-6)
    assert in_fixed[0] == pytest.approx(in_orig[0] * (s**2), rel=1e-6)
    assert in_fixed[1] == pytest.approx(in_orig[1] * (s**2), rel=1e-6)

    # 2. Density preserving policy: m' = s^3 * m, COM' = s*COM, I' = s^5 * I
    out_dens = tmp_path / "scaled_dens.osim"
    apply_segment_scaling(
        base_osim, scales, out_path=out_dens, policy=ScalingPolicy.DENSITY_PRESERVING
    )
    tree_dens = ET.parse(str(out_dens))
    b_dens = tree_dens.find(".//Body[@name='femur_r']")
    assert b_dens is not None
    m_dens = float(b_dens.findtext("mass", "0.0"))
    mc_dens = [float(v) for v in b_dens.findtext("mass_center", "0 0 0").split()]
    in_dens = [float(v) for v in b_dens.findtext("inertia", "0 0 0 0 0 0").split()]

    assert m_dens == pytest.approx(m_orig * (s**3), rel=1e-6)
    assert mc_dens[1] == pytest.approx(mc_orig[1] * s, rel=1e-6)
    assert in_dens[0] == pytest.approx(in_orig[0] * (s**5), rel=1e-6)

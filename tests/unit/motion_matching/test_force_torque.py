"""Unit tests for spatial wrench, contact reaction, and center of pressure contracts (MV-06, #10482)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.motion_matching.force_torque import (
    ContactReaction,
    SpatialWrench,
    compute_center_of_pressure,
    transform_wrench,
)

pytestmark = pytest.mark.unit


def test_spatial_wrench_creation_and_defaults() -> None:
    """Verify SpatialWrench stores coordinates and enforces SI units and conventions."""
    wrench = SpatialWrench(
        application_frame="ground",
        point_m=(0.1, 0.2, 0.0),
        force_n=(10.0, -5.0, 750.0),
        torque_nm=(1.5, 2.0, -0.5),
        direction_convention="applied_to_body",
        sign_convention="standard_cartesian",
    )
    assert wrench.application_frame == "ground"
    assert wrench.point_m == (0.1, 0.2, 0.0)
    assert wrench.force_n == (10.0, -5.0, 750.0)
    assert wrench.torque_nm == (1.5, 2.0, -0.5)
    assert wrench.units == {"force": "N", "torque": "N*m", "length": "m"}


def test_spatial_wrench_rejects_non_finite() -> None:
    """Verify SpatialWrench rejects NaN and Inf components."""
    with pytest.raises(ValueError, match="must be finite"):
        SpatialWrench(
            application_frame="ground",
            point_m=(float("nan"), 0.0, 0.0),
            force_n=(0.0, 0.0, 100.0),
            torque_nm=(0.0, 0.0, 0.0),
        )

    with pytest.raises(ValueError, match="must be finite"):
        SpatialWrench(
            application_frame="ground",
            point_m=(0.0, 0.0, 0.0),
            force_n=(0.0, 0.0, float("inf")),
            torque_nm=(0.0, 0.0, 0.0),
        )


def test_transform_wrench_pure_translation_includes_moment_arm() -> None:
    """Verify wrench transformation translates point of application and computes moment arm."""
    # A vertical force of 100 N at (1.0, 0.0, 0.0) with zero torque at origin.
    wrench_a = SpatialWrench(
        application_frame="ground",
        point_m=(1.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        torque_nm=(0.0, 0.0, 0.0),
    )

    # Transform to origin (0.0, 0.0, 0.0):
    # r = p_A - p_B = (1.0, 0.0, 0.0) - (0.0, 0.0, 0.0) = (1.0, 0.0, 0.0)
    # tau_B = tau_A + r x F = (0, 0, 0) + (1, 0, 0) x (0, 0, 100)
    # i x k = -j => tau_B = (0.0, -100.0, 0.0)
    wrench_b = transform_wrench(
        wrench_a,
        target_frame="world",
        new_point_m=(0.0, 0.0, 0.0),
        rotation_matrix=np.eye(3),
    )

    assert wrench_b.application_frame == "world"
    assert wrench_b.point_m == (0.0, 0.0, 0.0)
    assert np.allclose(wrench_b.force_n, (0.0, 0.0, 100.0))
    assert np.allclose(wrench_b.torque_nm, (0.0, -100.0, 0.0))


def test_transform_wrench_rotation_and_translation() -> None:
    """Verify wrench transformation with both 90-deg yaw rotation and translation."""
    # Force along x: (50.0, 0.0, 0.0) at point (0.0, 2.0, 0.0) with zero torque
    wrench = SpatialWrench(
        application_frame="body",
        point_m=(0.0, 2.0, 0.0),
        force_n=(50.0, 0.0, 0.0),
        torque_nm=(0.0, 0.0, 10.0),
    )

    # 90 deg rotation around z: x -> y, y -> -x
    rot_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Transform to new origin (0.0, 0.0, 0.0):
    # In target frame:
    # rotated point p_A' = rot_z_90 @ (0, 2, 0) = (-2, 0, 0)
    # rotated force F' = rot_z_90 @ (50, 0, 0) = (0, 50, 0)
    # rotated torque tau' = rot_z_90 @ (0, 0, 10) = (0, 0, 10)
    # moment arm r' = p_A' - (0, 0, 0) = (-2, 0, 0)
    # r' x F' = (-2, 0, 0) x (0, 50, 0) = (0, 0, -100)
    # total torque = (0, 0, 10) + (0, 0, -100) = (0, 0, -90)
    transformed = transform_wrench(
        wrench,
        target_frame="world",
        new_point_m=(0.0, 0.0, 0.0),
        rotation_matrix=rot_z_90,
    )

    assert transformed.application_frame == "world"
    assert np.allclose(transformed.force_n, (0.0, 50.0, 0.0))
    assert np.allclose(transformed.torque_nm, (0.0, 0.0, -90.0))


def test_center_of_pressure_defined_when_fz_above_threshold() -> None:
    """Verify CoP calculation when vertical normal force exceeds threshold."""
    # Let F_z = 500 N, tau_x = 25 N*m, tau_y = -50 N*m, point = (0, 0, 0)
    # x_cop = p_x - tau_y / F_z = 0 - (-50) / 500 = +0.10 m
    # y_cop = p_y + tau_x / F_z = 0 + 25 / 500 = +0.05 m
    wrench = SpatialWrench(
        application_frame="ground",
        point_m=(0.0, 0.0, 0.0),
        force_n=(10.0, 5.0, 500.0),
        torque_nm=(25.0, -50.0, 2.0),
    )
    cop = compute_center_of_pressure(wrench, f_threshold_n=5.0)
    assert cop is not None
    assert math.isclose(cop[0], 0.10, abs_tol=1e-6)
    assert math.isclose(cop[1], 0.05, abs_tol=1e-6)


def test_center_of_pressure_undefined_when_fz_below_threshold() -> None:
    """Verify CoP is strictly None (never (0, 0) or default) when normal force is low or negative."""
    # Case 1: Low normal force
    wrench_low = SpatialWrench(
        application_frame="ground",
        point_m=(0.0, 0.0, 0.0),
        force_n=(1.0, 1.0, 2.5),
        torque_nm=(1.0, 1.0, 0.0),
    )
    assert compute_center_of_pressure(wrench_low, f_threshold_n=5.0) is None

    # Case 2: Zero normal force
    wrench_zero = SpatialWrench(
        application_frame="ground",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 0.0),
        torque_nm=(0.0, 0.0, 0.0),
    )
    assert compute_center_of_pressure(wrench_zero, f_threshold_n=5.0) is None

    # Case 3: Tension / negative normal force
    wrench_neg = SpatialWrench(
        application_frame="ground",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, -50.0),
        torque_nm=(0.0, 0.0, 0.0),
    )
    assert compute_center_of_pressure(wrench_neg, f_threshold_n=5.0) is None


def test_contact_reaction_friction_utilization() -> None:
    """Verify ContactReaction calculates friction cone utilization and preserves missing channels."""
    reaction = ContactReaction(
        time_s=0.25,
        net_grf_n=(150.0, 0.0, 600.0),
        left_foot_grf_n=(100.0, 0.0, 400.0),
        right_foot_grf_n=(50.0, 0.0, 200.0),
        left_foot_cop_m=(-0.1, 0.05),
        right_foot_cop_m=(0.1, 0.05),
        contact_status={"left_foot": True, "right_foot": True},
        friction_utilization={"left_foot": 100.0 / 400.0, "right_foot": 50.0 / 200.0},
    )
    assert reaction.time_s == 0.25
    assert reaction.friction_utilization["left_foot"] == 0.25
    assert reaction.friction_utilization["right_foot"] == 0.25
    assert reaction.grip_wrench is None
    assert reaction.closure_residual_m is None

"""Analytic derivatives of the shared sphere-ground contact law."""

from collections.abc import Callable

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_derivatives import (
    sphere_ground_contact_derivatives,
)
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
    sphere_ground_contact,
)

pytestmark = pytest.mark.unit

RADIUS_M = 0.04


def _parameters() -> ContactParameters:
    return ContactParameters(
        stiffness_n_m=82_000.0,
        dissipation_s_m=0.35,
        static_friction=0.92,
        dynamic_friction=0.57,
        viscous_friction=0.08,
        transition_velocity_m_s=0.12,
    )


def _ground(normal: np.ndarray | tuple[float, float, float]) -> GroundPlane:
    return GroundPlane(normal=tuple(normal), height_m=0.017)


def _force(
    center: np.ndarray,
    velocity: np.ndarray,
    ground: GroundPlane,
    parameters: ContactParameters,
) -> np.ndarray:
    sample = sphere_ground_contact(center, velocity, RADIUS_M, ground, parameters)
    return sample.normal_force_n + sample.friction_force_n


def _centered_directional_difference(
    function: Callable[[np.ndarray], np.ndarray],
    value: np.ndarray,
    direction: np.ndarray,
    step: float,
) -> np.ndarray:
    return (function(value + step * direction) - function(value - step * direction)) / (
        2.0 * step
    )


def _centered_jacobian(
    function: Callable[[np.ndarray], np.ndarray], value: np.ndarray, step: float
) -> np.ndarray:
    basis = np.eye(3)
    return np.column_stack(
        [
            _centered_directional_difference(function, value, direction, step)
            for direction in basis
        ]
    )


@pytest.mark.parametrize(
    ("normal", "normal_speed", "tangential_speed"),
    [
        ((0.3, -0.4, 0.866025403784), 0.0, 0.0),
        ((0.0, 0.0, 1.0), -0.3, 0.0),
        ((0.3, -0.4, 0.866025403784), -0.3, 2e-9),
        ((0.3, -0.4, 0.866025403784), -0.3, 0.12 * 0.999e-4),
        ((0.3, -0.4, 0.866025403784), -0.3, 0.12 * 1.001e-4),
        ((-0.2, 0.7, 0.68556546004), -0.45, 1.7),
        ((0.3, -0.4, 0.866025403784), 0.75, 0.9),
    ],
    ids=(
        "stationary-no-slip",
        "pure-normal-approach",
        "tiny-slip-approach",
        "series-crossover-below",
        "series-crossover-above",
        "high-slip-approach",
        "separating",
    ),
)
def test_active_contact_jacobians_match_directional_force_differences(
    normal: tuple[float, float, float],
    normal_speed: float,
    tangential_speed: float,
) -> None:
    ground = _ground(normal)
    n = np.asarray(ground.normal)
    center = n * (ground.height_m + RADIUS_M - 0.013) + np.array([0.021, -0.009, 0.004])
    # Restore the requested normal penetration after adding an arbitrary world offset.
    center += n * (ground.height_m + RADIUS_M - 0.013 - n @ center)
    tangent = np.array([0.7, 0.2, -0.4])
    tangent -= n * (n @ tangent)
    tangent /= np.linalg.norm(tangent)
    velocity_array = normal_speed * n + tangential_speed * tangent
    center_direction = np.array([0.31, -0.73, 0.42])
    velocity_direction = np.array([-0.28, 0.51, 0.81])

    result = sphere_ground_contact_derivatives(
        center, velocity_array, RADIUS_M, ground, _parameters()
    )
    center_fd = _centered_directional_difference(
        lambda candidate: _force(candidate, velocity_array, ground, _parameters()),
        center,
        center_direction,
        2e-7,
    )
    velocity_fd = _centered_directional_difference(
        lambda candidate: _force(center, candidate, ground, _parameters()),
        velocity_array,
        velocity_direction,
        2e-7,
    )

    assert result.differentiable
    np.testing.assert_allclose(
        result.dforce_dcenter @ center_direction, center_fd, rtol=2e-6, atol=2e-5
    )
    np.testing.assert_allclose(
        result.dforce_dvelocity @ velocity_direction,
        velocity_fd,
        rtol=2e-6,
        atol=2e-5,
    )


def test_random_active_contact_jacobians_match_full_finite_differences() -> None:
    rng = np.random.default_rng(10255)
    parameters = _parameters()
    for _ in range(10):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        ground = _ground(normal)
        n = np.asarray(ground.normal)
        penetration = rng.uniform(0.005, 0.02)
        center = rng.normal(scale=0.05, size=3)
        center += n * (ground.height_m + RADIUS_M - penetration - n @ center)
        tangent = rng.normal(size=3)
        tangent -= n * (n @ tangent)
        tangent /= np.linalg.norm(tangent)
        velocity = rng.uniform(-0.8, 0.8) * n + rng.uniform(0.002, 2.0) * tangent

        result = sphere_ground_contact_derivatives(
            center, velocity, RADIUS_M, ground, parameters
        )
        center_fd = _centered_jacobian(
            lambda candidate, velocity=velocity, ground=ground: _force(
                candidate, velocity, ground, parameters
            ),
            center,
            1e-7,
        )
        velocity_fd = _centered_jacobian(
            lambda candidate, center=center, ground=ground: _force(
                center, candidate, ground, parameters
            ),
            velocity,
            1e-7,
        )

        assert result.differentiable
        np.testing.assert_allclose(
            result.dforce_dcenter, center_fd, rtol=3e-6, atol=3e-5
        )
        np.testing.assert_allclose(
            result.dforce_dvelocity, velocity_fd, rtol=3e-6, atol=3e-5
        )


@pytest.mark.parametrize(
    ("center_offset", "normal_velocity"),
    [(0.02, -0.4), (-0.01, 4.0)],
    ids=("off-ground", "normal-force-clipped"),
)
def test_inactive_contact_has_exact_zero_derivatives(
    center_offset: float, normal_velocity: float
) -> None:
    ground = _ground((0.2, -0.1, 0.97467943448))
    n = np.asarray(ground.normal)
    center = n * (ground.height_m + RADIUS_M + center_offset)
    velocity = np.array([0.7, -0.2, 0.1]) + normal_velocity * n
    velocity -= n * (n @ velocity - normal_velocity)

    result = sphere_ground_contact_derivatives(
        center, velocity, RADIUS_M, ground, _parameters()
    )

    np.testing.assert_array_equal(result.dforce_dcenter, np.zeros((3, 3)))
    np.testing.assert_array_equal(result.dforce_dvelocity, np.zeros((3, 3)))
    assert result.differentiable


def test_penetration_boundary_reports_distinct_one_sided_slopes() -> None:
    ground = _ground((0.25, 0.4, 0.88175960443))
    n = np.asarray(ground.normal)
    center = n * (ground.height_m + RADIUS_M)
    velocity = np.array([0.4, -0.3, -0.2])
    result = sphere_ground_contact_derivatives(
        center, velocity, RADIUS_M, ground, _parameters()
    )
    force_at_boundary = _force(center, velocity, ground, _parameters())
    step = 1e-8
    inward = (
        _force(center - step * n, velocity, ground, _parameters()) - force_at_boundary
    ) / step
    outward = (
        _force(center + step * n, velocity, ground, _parameters()) - force_at_boundary
    ) / step

    assert not result.differentiable
    np.testing.assert_array_equal(result.dforce_dcenter, np.zeros((3, 3)))
    assert np.linalg.norm(inward) > 1_000.0
    np.testing.assert_allclose(outward, 0.0, atol=1e-12)


def test_normal_force_clip_boundary_reports_distinct_one_sided_slopes() -> None:
    parameters = _parameters()
    ground = _ground((-0.35, 0.1, 0.931396))
    n = np.asarray(ground.normal)
    center = n * (ground.height_m + RADIUS_M - 0.011)
    tangent = np.array([0.8, -0.25, 0.0])
    tangent -= n * (n @ tangent)
    velocity = tangent + n / parameters.dissipation_s_m
    result = sphere_ground_contact_derivatives(
        center, velocity, RADIUS_M, ground, parameters
    )
    force_at_boundary = _force(center, velocity, ground, parameters)
    step = 1e-8
    active_side = (
        _force(center, velocity - step * n, ground, parameters) - force_at_boundary
    ) / step
    clipped_side = (
        _force(center, velocity + step * n, ground, parameters) - force_at_boundary
    ) / step

    assert not result.differentiable
    np.testing.assert_array_equal(result.dforce_dvelocity, np.zeros((3, 3)))
    assert np.linalg.norm(active_side) > 100.0
    np.testing.assert_allclose(clipped_side, 0.0, atol=1e-12)


def test_derivatives_are_rotationally_equivariant() -> None:
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    angle = 0.73
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    ground = _ground((0.2, -0.5, 0.84261497732))
    center = np.array([0.03, -0.04, 0.02])
    n = np.asarray(ground.normal)
    center += n * (ground.height_m + RADIUS_M - 0.009 - n @ center)
    velocity = np.array([0.9, -0.35, -0.4])
    rotated_ground = _ground(rotation @ n)

    original = sphere_ground_contact_derivatives(
        center, velocity, RADIUS_M, ground, _parameters()
    )
    rotated = sphere_ground_contact_derivatives(
        rotation @ center,
        rotation @ velocity,
        RADIUS_M,
        rotated_ground,
        _parameters(),
    )

    np.testing.assert_allclose(
        rotated.sample.normal_force_n + rotated.sample.friction_force_n,
        rotation @ (original.sample.normal_force_n + original.sample.friction_force_n),
        rtol=2e-13,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        rotated.dforce_dcenter,
        rotation @ original.dforce_dcenter @ rotation.T,
        rtol=2e-13,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        rotated.dforce_dvelocity,
        rotation @ original.dforce_dvelocity @ rotation.T,
        rtol=2e-13,
        atol=2e-9,
    )


@pytest.mark.parametrize(
    ("center", "velocity", "radius", "message"),
    [
        (np.zeros(2), np.zeros(3), RADIUS_M, "center_m must be a finite 3-vector"),
        (np.zeros(3), np.array([0.0, np.nan, 0.0]), RADIUS_M, "velocity_m_s"),
        (np.zeros(3), np.zeros(3), 0.0, "Sphere radius must be positive"),
    ],
)
def test_input_validation_is_reused_from_contact_law(
    center: np.ndarray, velocity: np.ndarray, radius: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        sphere_ground_contact_derivatives(
            center, velocity, radius, _ground((0.0, 0.0, 1.0)), _parameters()
        )


def test_derivative_matrices_are_owned_and_read_only() -> None:
    ground = _ground((0.0, 0.0, 1.0))
    result = sphere_ground_contact_derivatives(
        np.array([0.0, 0.0, 0.047]),
        np.array([0.3, -0.2, -0.1]),
        RADIUS_M,
        ground,
        _parameters(),
    )

    for matrix in (result.dforce_dcenter, result.dforce_dvelocity):
        assert matrix.flags.owndata
        assert not matrix.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            matrix[0, 0] = 0.0

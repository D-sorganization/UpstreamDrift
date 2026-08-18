"""The rigid intruder: its distance field, its motion and its contact.

The two things that must not be got wrong here are the **sign of the
torque** -- plane strain leaves exactly one component alive and it is
easy to invert -- and **tunnelling**, which at 25 m/s is the difference
between a solver and a random number generator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.solvers.exceptions import SolverInputError
from bunkershot3d.solvers.mpm.body import (
    RigidSection,
    convex_hull_2d,
    plane_torque_about_y,
)

pytestmark = pytest.mark.unit


def unit_square(**kwargs: object) -> RigidSection:
    """A 1 m square centred on the origin, counter-clockwise."""
    vertices = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    return RigidSection(vertices, **kwargs)  # type: ignore[arg-type]


class TestConvexHull:
    """Monotone chain, implemented rather than imported."""

    def test_recovers_a_square_from_a_cloud(self) -> None:
        rng = np.random.default_rng(31)
        interior = rng.uniform(-0.4, 0.4, size=(200, 2))
        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        hull = convex_hull_2d(np.vstack([interior, corners]))
        assert hull.shape == (4, 2)
        np.testing.assert_allclose(np.sort(hull, axis=0), np.sort(corners, axis=0))

    def test_winding_is_counter_clockwise(self) -> None:
        hull = convex_hull_2d(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        following = np.roll(hull, -1, axis=0)
        area = 0.5 * float(
            (hull[:, 0] * following[:, 1] - following[:, 0] * hull[:, 1]).sum()
        )
        assert area > 0.0

    def test_collinear_points_are_refused(self) -> None:
        line = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        with pytest.raises(SolverInputError, match="collinear"):
            convex_hull_2d(line)

    def test_too_few_points_are_refused(self) -> None:
        with pytest.raises(SolverInputError, match="at least 3 distinct"):
            convex_hull_2d(np.array([[0.0, 0.0], [1.0, 0.0]]))


class TestSignedDistance:
    """Negative inside, and the normal is the real gradient direction."""

    def test_centre_is_deepest(self) -> None:
        distance, _ = unit_square().signed_distance(np.array([[0.0, 0.0]]))
        assert distance[0] == pytest.approx(-0.5)

    def test_face_distance_is_exact(self) -> None:
        section = unit_square()
        distance, normal = section.signed_distance(np.array([[0.0, 1.25]]))
        assert distance[0] == pytest.approx(0.75)
        np.testing.assert_allclose(normal[0], [0.0, 1.0], atol=1e-14)

    def test_corner_distance_is_the_true_distance(self) -> None:
        """A max-of-half-planes surrogate would report 0.5 here, not 0.707."""
        section = unit_square()
        distance, normal = section.signed_distance(np.array([[1.0, 1.0]]))
        assert distance[0] == pytest.approx(math.sqrt(0.5), rel=1e-12)
        np.testing.assert_allclose(
            normal[0], [math.sqrt(0.5), math.sqrt(0.5)], atol=1e-12
        )

    def test_normal_is_outward_everywhere_outside(self) -> None:
        rng = np.random.default_rng(37)
        section = unit_square()
        points = rng.uniform(-2.0, 2.0, size=(500, 2))
        distance, normal = section.signed_distance(points)
        outside = distance > 1e-9
        stepped = points[outside] + 1e-6 * normal[outside]
        stepped_distance, _ = section.signed_distance(stepped)
        assert np.all(stepped_distance > distance[outside])

    def test_contains_agrees_with_the_sign(self) -> None:
        rng = np.random.default_rng(41)
        section = unit_square()
        points = rng.uniform(-1.0, 1.0, size=(300, 2))
        distance, _ = section.signed_distance(points)
        np.testing.assert_array_equal(section.contains(points), distance < 0.0)

    def test_clockwise_input_is_rewound(self) -> None:
        clockwise = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]])
        section = RigidSection(clockwise)
        assert section.area_m2 == pytest.approx(1.0)
        distance, _ = section.signed_distance(np.array([[0.0, 0.0]]))
        assert distance[0] < 0.0


class TestMotion:
    """Rigid motion, and the sign of everything that rotates."""

    def test_translation_advances_by_velocity(self) -> None:
        section = unit_square(velocity_m_s=(2.0, -1.0))
        moved = section.advanced(0.5)
        np.testing.assert_allclose(
            moved.vertices_m, section.vertices_m + np.array([1.0, -0.5])
        )

    def test_rotation_matches_the_velocity_field(self) -> None:
        """The advanced pose and ``velocity_at`` must be the same rotation."""
        spin = 12.0
        section = unit_square(angular_velocity_rad_s=spin, reference_point_m=(0.0, 0.0))
        step = 1e-7
        moved = section.advanced(step)
        numeric = (moved.vertices_m - section.vertices_m) / step
        analytic = section.velocity_at(section.vertices_m)
        np.testing.assert_allclose(numeric, analytic, rtol=1e-6, atol=1e-6)

    def test_velocity_from_spin_is_the_y_axis_convention(self) -> None:
        section = unit_square(angular_velocity_rad_s=1.0, reference_point_m=(0.0, 0.0))
        velocity = section.velocity_at(np.array([[1.0, 0.0]]))
        # omega y_hat x (1, 0, 0) = (0, 0, -1): straight down in z.
        np.testing.assert_allclose(velocity[0], [0.0, -1.0], atol=1e-14)

    def test_max_speed_includes_the_spin(self) -> None:
        section = unit_square(
            velocity_m_s=(1.0, 0.0),
            angular_velocity_rad_s=100.0,
            reference_point_m=(0.0, 0.0),
        )
        assert section.max_speed_m_s > section.speed_m_s


class TestTorqueSign:
    """One component survives plane strain and its sign is load-bearing."""

    def test_matches_the_three_dimensional_cross_product(self) -> None:
        lever = np.array([[0.03, -0.01], [0.0, 0.05]])
        force = np.array([[10.0, 4.0], [-2.0, 7.0]])
        lever_3d = np.stack([lever[:, 0], np.zeros(2), lever[:, 1]], axis=1)
        force_3d = np.stack([force[:, 0], np.zeros(2), force[:, 1]], axis=1)
        expected = float(np.cross(lever_3d, force_3d)[:, 1].sum())
        assert plane_torque_about_y(lever, force) == pytest.approx(expected, rel=1e-14)


class TestContactProjection:
    """The momentum ledger is the whole of F1's contact wrench."""

    def _nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """One node just inside the top face, one well clear of the body."""
        positions = np.array([[0.0, 0.45], [0.0, 2.0]])
        mass = np.array([1.0, 1.0])
        return positions, mass

    def test_a_separating_node_is_left_alone(self) -> None:
        """Sand may leave the club freely: this is what opens the divot."""
        section = unit_square(velocity_m_s=(0.0, 0.0), friction=0.0)
        positions, mass = self._nodes()
        velocity = np.array([[0.0, 5.0], [0.0, 0.0]])
        updated, impulse = section.project_grid_velocity(
            positions, velocity, mass, time_step_s=1e-4
        )
        np.testing.assert_allclose(updated[0], velocity[0])
        np.testing.assert_allclose(impulse.impulse_n_s[0], [0.0, 0.0], atol=1e-15)

    def test_an_approaching_node_loses_its_normal_velocity(self) -> None:
        section = unit_square(velocity_m_s=(0.0, 0.0), friction=0.0)
        positions, mass = self._nodes()
        velocity = np.array([[0.0, -5.0], [0.0, 0.0]])
        updated, impulse = section.project_grid_velocity(
            positions, velocity, mass, time_step_s=1e-4
        )
        assert float(updated[0, 1]) == pytest.approx(0.0, abs=1e-14)
        np.testing.assert_allclose(impulse.impulse_n_s[0], [0.0, 5.0], atol=1e-13)

    def test_sticking_removes_the_whole_relative_velocity(self) -> None:
        section = unit_square(velocity_m_s=(0.0, 0.0), friction=10.0)
        positions = np.array([[0.0, 0.45]])
        mass = np.array([2.0])
        velocity = np.array([[3.0, -1.0]])
        updated, impulse = section.project_grid_velocity(
            positions, velocity, mass, time_step_s=1e-4
        )
        np.testing.assert_allclose(updated[0], [0.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(impulse.impulse_n_s[0], [-6.0, 2.0], atol=1e-13)

    def test_a_frictionless_slide_keeps_the_tangential_velocity(self) -> None:
        section = unit_square(velocity_m_s=(0.0, 0.0), friction=0.0)
        positions = np.array([[0.0, 0.45]])
        mass = np.array([2.0])
        velocity = np.array([[3.0, -1.0]])
        updated, _ = section.project_grid_velocity(
            positions, velocity, mass, time_step_s=1e-4
        )
        np.testing.assert_allclose(updated[0], [3.0, 0.0], atol=1e-14)

    def test_the_ledger_reports_the_reaction_on_the_body(self) -> None:
        """Newton's third law, arithmetically."""
        section = unit_square(velocity_m_s=(0.0, 0.0), friction=10.0)
        positions = np.array([[0.0, 0.45]])
        mass = np.array([2.0])
        velocity = np.array([[0.0, -4.0]])
        step = 2e-4
        _, impulse = section.project_grid_velocity(
            positions, velocity, mass, time_step_s=step
        )
        force = impulse.force_on_body_n(step)
        # The sand carried -8 kg m/s downward and was stopped, so the club
        # took 8 kg m/s downward over the step.
        np.testing.assert_allclose(force, [0.0, -8.0 / step], rtol=1e-13)

    def test_a_moving_body_is_matched_not_stopped(self) -> None:
        """Sand under a descending sole is carried down with it."""
        section = unit_square(velocity_m_s=(0.0, -6.0), friction=10.0)
        positions = np.array([[0.0, -0.45]])
        mass = np.array([1.0])
        velocity = np.array([[0.0, 0.0]])
        updated, _ = section.project_grid_velocity(
            positions, velocity, mass, time_step_s=1e-4
        )
        np.testing.assert_allclose(updated[0], [0.0, -6.0], atol=1e-14)

    def test_partial_friction_reduces_but_does_not_stop_the_slide(self) -> None:
        """The Coulomb cone bounds the tangential velocity; it never sets it."""
        section = unit_square(velocity_m_s=(0.0, 0.0), friction=0.25)
        positions = np.array([[0.0, 0.45]])
        mass = np.array([1.0])
        velocity = np.array([[4.0, -2.0]])
        updated, _ = section.project_grid_velocity(
            positions, velocity, mass, time_step_s=1e-4
        )
        # |v_t| = 4, mu |v_n| = 0.5, so 3.5 survives.
        np.testing.assert_allclose(updated[0], [3.5, 0.0], atol=1e-13)

    def test_massless_nodes_are_skipped(self) -> None:
        section = unit_square(friction=1.0)
        positions = np.array([[0.0, 0.0]])
        mass = np.array([0.0])
        velocity = np.array([[0.0, -5.0]])
        _, impulse = section.project_grid_velocity(
            positions, velocity, mass, time_step_s=1e-4
        )
        assert impulse.n_contacts == 0
        np.testing.assert_allclose(impulse.force_on_body_n(1e-4), [0.0, 0.0])

    def test_a_non_positive_step_is_refused(self) -> None:
        section = unit_square()
        with pytest.raises(SolverInputError, match="time_step_s"):
            section.project_grid_velocity(
                np.zeros((1, 2)), np.zeros((1, 2)), np.ones(1), time_step_s=0.0
            )


class TestSweptContact:
    """A node the club is about to reach is stopped before it arrives."""

    def test_a_node_ahead_of_the_body_is_collided(self) -> None:
        section = unit_square(velocity_m_s=(20.0, 0.0), friction=0.0)
        # 1 mm ahead of the leading face, which travels 2 mm this step.
        positions = np.array([[0.501, 0.0]])
        mass = np.array([1.0])
        velocity = np.array([[0.0, 0.0]])
        updated, impulse = section.project_grid_velocity(
            positions, velocity, mass, time_step_s=1.0e-4
        )
        assert impulse.n_contacts == 1
        assert impulse.n_swept == 1
        assert float(updated[0, 0]) == pytest.approx(20.0)

    def test_a_node_the_body_will_not_reach_is_untouched(self) -> None:
        section = unit_square(velocity_m_s=(20.0, 0.0), friction=0.0)
        positions = np.array([[0.60, 0.0]])
        mass = np.array([1.0])
        velocity = np.zeros((1, 2))
        _, impulse = section.project_grid_velocity(
            positions, velocity, mass, time_step_s=1.0e-4
        )
        assert impulse.n_contacts == 0


class TestPushOut:
    """The backstop, and the reason it is reported rather than hidden."""

    def test_an_embedded_particle_is_placed_on_the_surface(self) -> None:
        section = unit_square()
        positions = np.array([[0.0, 0.2], [2.0, 2.0]])
        velocity = np.array([[0.0, -1.0], [0.0, 0.0]])
        moved, updated, count = section.push_out(positions, velocity)
        assert count == 1
        distance, _ = section.signed_distance(moved)
        assert float(distance[0]) == pytest.approx(0.0, abs=1e-14)
        np.testing.assert_allclose(moved[1], positions[1])

    def test_inward_velocity_is_removed_and_sliding_is_not(self) -> None:
        section = unit_square()
        positions = np.array([[0.0, 0.45]])
        velocity = np.array([[3.0, -2.0]])
        _, updated, _ = section.push_out(positions, velocity)
        np.testing.assert_allclose(updated[0], [3.0, 0.0], atol=1e-14)

    def test_nothing_inside_means_nothing_moves(self) -> None:
        section = unit_square()
        positions = np.array([[2.0, 2.0]])
        velocity = np.array([[1.0, 1.0]])
        moved, updated, count = section.push_out(positions, velocity)
        assert count == 0
        assert moved is positions
        assert updated is velocity

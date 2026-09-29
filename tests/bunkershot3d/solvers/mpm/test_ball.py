"""The ball as a rigid circular section in the plane-strain plane (#8733 §1).

ADR-0033 decided the ball becomes a body *inside* F1 rather than a
post-processing step, so the tests here are about two things that are easy
to get wrong and expensive to discover later:

* the polygonal circle is an **equal-area** approximation, so the section
  the sand meets displaces the same area per unit width as the circle it
  stands for;
* the two facts ADR-0033 requires to travel with the ball -- that it is an
  infinite cylinder rather than a sphere, and that the below-equator /
  face-side split is qualitative and in-plane only -- are enforced by the
  API, not merely written in a docstring.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.solvers.exceptions import OutOfEnvelopeError, SolverInputError
from bunkershot3d.solvers.mpm.ball import (
    BALL_DIAMETER_M,
    BALL_RADIUS_M,
    DEFAULT_BALL_FACETS,
    PLANE_STRAIN_BALL_NOTE,
    BallContactSplit,
    BallSection,
    circular_section,
    n_facets_for_cell_size,
)
from bunkershot3d.solvers.mpm.body import ContactImpulse, RigidSection

pytestmark = pytest.mark.unit


class TestCircularSection:
    """The polygon that stands in for a circle."""

    def test_it_is_a_rigid_section(self) -> None:
        section = circular_section((0.0, 0.05), 0.02)
        assert isinstance(section, RigidSection)

    def test_it_matches_the_circle_area(self) -> None:
        radius = 0.021336
        section = circular_section((0.1, 0.05), radius, n_facets=24)
        assert section.area_m2 == pytest.approx(math.pi * radius**2, rel=1e-12)

    def test_area_matches_at_every_facet_count(self) -> None:
        radius = 0.03
        for facets in (8, 12, 24, 48):
            section = circular_section((0.0, 0.0), radius, n_facets=facets)
            assert section.area_m2 == pytest.approx(math.pi * radius**2, rel=1e-12)

    def test_it_is_centred_on_the_requested_point(self) -> None:
        section = circular_section((0.031, -0.017), 0.02, n_facets=32)
        assert section.reference_point_m == pytest.approx([0.031, -0.017])

    def test_the_inscribed_radius_brackets_the_true_radius(self) -> None:
        # Equal-area means the polygon crosses the circle: its inradius is
        # inside and its circumradius outside, so neither the sand it
        # displaces nor the surface it presents is systematically biased.
        radius = 0.02
        section = circular_section((0.0, 0.0), radius, n_facets=DEFAULT_BALL_FACETS)
        distances = np.linalg.norm(section.vertices_m, axis=1)
        assert distances.min() > radius
        inradius = distances.min() * math.cos(math.pi / DEFAULT_BALL_FACETS)
        assert inradius < radius

    def test_it_carries_the_velocity_it_was_given(self) -> None:
        section = circular_section((0.0, 0.0), 0.02, velocity_m_s=(1.5, -2.5))
        assert section.velocity_m_s == pytest.approx([1.5, -2.5])

    def test_too_few_facets_is_refused(self) -> None:
        with pytest.raises(SolverInputError, match="n_facets"):
            circular_section((0.0, 0.0), 0.02, n_facets=2)

    def test_a_non_positive_radius_is_refused(self) -> None:
        with pytest.raises(SolverInputError, match="radius_m"):
            circular_section((0.0, 0.0), 0.0)


class TestFacetCount:
    """Choosing enough facets that the grid cannot see the flat spots."""

    def test_the_chord_fits_inside_a_cell(self) -> None:
        radius = BALL_RADIUS_M
        cell = 0.004
        facets = n_facets_for_cell_size(radius_m=radius, cell_size_m=cell)
        chord = 2.0 * radius * math.sin(math.pi / facets)
        assert chord <= cell

    def test_a_coarse_grid_still_gets_a_usable_polygon(self) -> None:
        facets = n_facets_for_cell_size(radius_m=BALL_RADIUS_M, cell_size_m=1.0)
        assert facets >= 8

    def test_a_finer_grid_asks_for_more_facets(self) -> None:
        coarse = n_facets_for_cell_size(radius_m=0.02, cell_size_m=0.004)
        fine = n_facets_for_cell_size(radius_m=0.02, cell_size_m=0.001)
        assert fine > coarse

    def test_a_non_positive_cell_is_refused(self) -> None:
        with pytest.raises(SolverInputError, match="cell_size_m"):
            n_facets_for_cell_size(radius_m=0.02, cell_size_m=0.0)


class TestBallSection:
    """The ball body itself."""

    def test_the_default_is_a_regulation_ball(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        assert ball.radius_m == pytest.approx(BALL_RADIUS_M)
        assert pytest.approx(0.042672) == BALL_DIAMETER_M

    def test_resting_on_places_the_ball_tangent_to_the_surface(self) -> None:
        ball = BallSection.resting_on(x_m=0.03, free_surface_height_m=0.01)
        assert ball.centre_m[1] == pytest.approx(0.01 + BALL_RADIUS_M)
        assert ball.centre_m[0] == pytest.approx(0.03)

    def test_the_section_is_a_body_the_solver_can_take(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        assert isinstance(ball.section, RigidSection)

    def test_the_centre_is_the_sections_reference_point(self) -> None:
        ball = BallSection.at((0.02, 0.05))
        assert ball.section.reference_point_m == pytest.approx(ball.centre_m)

    def test_advancing_moves_the_centre_with_the_velocity(self) -> None:
        ball = BallSection.at((0.0, 0.05), velocity_m_s=(3.0, 1.0))
        moved = ball.advanced(0.01)
        assert moved.centre_m == pytest.approx([0.03, 0.06])
        assert moved.radius_m == pytest.approx(ball.radius_m)

    def test_advancing_returns_a_ball_not_a_bare_section(self) -> None:
        moved = BallSection.at((0.0, 0.05)).advanced(1e-3)
        assert isinstance(moved, BallSection)

    def test_with_section_keeps_the_radius_and_facets(self) -> None:
        ball = BallSection.at((0.0, 0.05), n_facets=16)
        rebuilt = ball.with_section(ball.section.translated((0.01, 0.0)))
        assert rebuilt.n_facets == 16
        assert rebuilt.centre_m[0] == pytest.approx(0.01)

    def test_a_negative_radius_is_refused(self) -> None:
        with pytest.raises(SolverInputError, match="radius_m"):
            BallSection.at((0.0, 0.0), radius_m=-0.01)


class TestInfiniteCylinderNotSphere:
    """The first fact ADR-0033 requires to travel with the ball."""

    def test_the_note_says_infinite_cylinder(self) -> None:
        assert "cylinder" in PLANE_STRAIN_BALL_NOTE
        assert "sphere" in PLANE_STRAIN_BALL_NOTE

    def test_the_geometry_note_is_on_the_body(self) -> None:
        assert BallSection.at((0.0, 0.05)).geometry_note() == PLANE_STRAIN_BALL_NOTE

    def test_the_line_mass_is_per_unit_width(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        line_mass = ball.line_mass_kg_per_m(1130.0)
        assert line_mass == pytest.approx(1130.0 * math.pi * BALL_RADIUS_M**2)

    def test_the_line_mass_is_not_a_ball_mass(self) -> None:
        # A regulation ball is 45.93 g. The plane-strain body's line mass
        # times any plausible width is not that number, and the point of
        # measuring it here is that nobody can later mistake one for the
        # other.
        ball = BallSection.at((0.0, 0.05))
        for width_m in (0.02, 0.0427, 0.05):
            assert ball.line_mass_kg_per_m(1130.0) * width_m != pytest.approx(
                0.04593, rel=0.05
            )

    def test_asking_for_a_sphere_mass_raises(self) -> None:
        with pytest.raises(OutOfEnvelopeError, match="cylinder"):
            BallSection.at((0.0, 0.05)).sphere_mass_kg(1130.0)

    def test_a_negative_density_is_refused(self) -> None:
        with pytest.raises(SolverInputError, match="density"):
            BallSection.at((0.0, 0.05)).line_mass_kg_per_m(-1.0)


class TestRefusals:
    """Ball launch and out-of-plane distribution stay refused."""

    def test_launch_velocity_raises(self) -> None:
        with pytest.raises(OutOfEnvelopeError, match="ball_launch"):
            BallSection.at((0.0, 0.05)).launch_velocity_m_s()

    def test_launch_refusal_names_the_f0_path(self) -> None:
        with pytest.raises(OutOfEnvelopeError, match="8657"):
            BallSection.at((0.0, 0.05)).launch_velocity_m_s()

    def test_heel_toe_split_raises(self) -> None:
        with pytest.raises(OutOfEnvelopeError, match="out_of_plane"):
            BallSection.at((0.0, 0.05)).heel_toe_split()

    def test_lateral_distribution_raises(self) -> None:
        with pytest.raises(OutOfEnvelopeError, match="out_of_plane"):
            BallSection.at((0.0, 0.05)).lateral_distribution()


def _impulse(
    positions: list[list[float]], vectors: list[list[float]]
) -> ContactImpulse:
    """A hand-built ledger, so the split is tested on known geometry."""
    return ContactImpulse(
        node_index=np.arange(len(positions), dtype=np.int64),
        impulse_n_s=np.array(vectors, dtype=np.float64).reshape(-1, 2),
        position_m=np.array(positions, dtype=np.float64).reshape(-1, 2),
        stress_force_n=np.zeros(2, dtype=np.float64),
        n_swept=0,
    )


class TestQualitativeSplit:
    """The below-equator / face-side split #8712 asks for."""

    def test_below_the_equator_is_below_the_centre(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        ledger = _impulse(
            [[0.0, 0.04], [0.0, 0.06]],
            [[1.0, 0.0], [3.0, 0.0]],
        )
        split = ball.split_contact(ledger, approach_direction=(1.0, 0.0))
        assert split.below_equator_n_s == pytest.approx([1.0, 0.0])
        assert split.above_equator_n_s == pytest.approx([3.0, 0.0])

    def test_the_face_side_is_the_side_the_club_comes_from(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        ledger = _impulse(
            [[-0.01, 0.05], [0.01, 0.05]],
            [[2.0, 0.0], [5.0, 0.0]],
        )
        split = ball.split_contact(ledger, approach_direction=(1.0, 0.0))
        assert split.face_side_n_s == pytest.approx([2.0, 0.0])
        assert split.far_side_n_s == pytest.approx([5.0, 0.0])

    def test_reversing_the_approach_reverses_the_sides(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        ledger = _impulse(
            [[-0.01, 0.05], [0.01, 0.05]],
            [[2.0, 0.0], [5.0, 0.0]],
        )
        split = ball.split_contact(ledger, approach_direction=(-1.0, 0.0))
        assert split.face_side_n_s == pytest.approx([5.0, 0.0])

    def test_the_halves_sum_to_the_total_both_ways(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        ledger = _impulse(
            [[-0.01, 0.043], [0.008, 0.058], [0.0, 0.049]],
            [[2.0, -1.0], [5.0, 0.5], [-1.0, 2.0]],
        )
        split = ball.split_contact(ledger, approach_direction=(0.8, -0.6))
        assert split.total_n_s == pytest.approx(
            np.array([[2.0, -1.0], [5.0, 0.5], [-1.0, 2.0]]).sum(axis=0)
        )
        assert split.below_equator_n_s + split.above_equator_n_s == pytest.approx(
            split.total_n_s
        )
        assert split.face_side_n_s + split.far_side_n_s == pytest.approx(
            split.total_n_s
        )

    def test_fractions_are_dimensionless_and_bounded(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        ledger = _impulse(
            [[-0.01, 0.043], [0.008, 0.058]],
            [[2.0, -1.0], [5.0, 0.5]],
        )
        split = ball.split_contact(ledger, approach_direction=(1.0, 0.0))
        assert 0.0 <= split.below_equator_fraction <= 1.0
        assert 0.0 <= split.face_side_fraction <= 1.0

    def test_an_empty_ledger_splits_to_zero_rather_than_dividing(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        split = ball.split_contact(_impulse([], []), approach_direction=(1.0, 0.0))
        assert split.n_contacts == 0
        assert split.below_equator_fraction == 0.0
        assert split.face_side_fraction == 0.0

    def test_the_split_says_it_is_qualitative(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        split = ball.split_contact(_impulse([], []), approach_direction=(1.0, 0.0))
        assert split.is_qualitative is True
        assert "qualitative" in split.summary().lower()

    def test_the_summary_carries_the_cylinder_note(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        split = ball.split_contact(_impulse([], []), approach_direction=(1.0, 0.0))
        assert "cylinder" in split.summary()

    def test_a_zero_approach_direction_is_refused(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        with pytest.raises(SolverInputError, match="approach_direction"):
            ball.split_contact(_impulse([], []), approach_direction=(0.0, 0.0))

    def test_the_split_refuses_a_lateral_question(self) -> None:
        ball = BallSection.at((0.0, 0.05))
        split = ball.split_contact(_impulse([], []), approach_direction=(1.0, 0.0))
        assert isinstance(split, BallContactSplit)
        with pytest.raises(OutOfEnvelopeError, match="out_of_plane"):
            split.heel_toe_fraction()

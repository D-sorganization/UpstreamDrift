"""Playability window area -- the primary scalar objective (issue #8614, W7).

The grids here are chosen so the trapezoidal node weights are trivial to write
down. Axis A has stations ``[0, 1, 2, 3, 4]``, so its weights are
``[0.5, 1, 1, 1, 0.5]`` and sum to the span 4. Axis B has ``[0, 0.5, 1.0]``, so
its weights are ``[0.25, 0.5, 0.25]`` and sum to 1. The domain area is therefore
``4 * 1 = 4`` in A-units x B-units.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.metrics import (
    PlayabilityAxis,
    playability_objective,
    playability_window,
)

pytestmark = pytest.mark.unit

TARGET_CARRY_M = 30.0

#: Inside the +/-10 % band (27.0 to 33.0 m).
GOOD_CARRY_M = 30.0

#: Outside it.
BAD_CARRY_M = 20.0


@pytest.fixture
def axis_a() -> PlayabilityAxis:
    """Entry distance, five stations, weights [0.5, 1, 1, 1, 0.5]."""
    return PlayabilityAxis(
        name="entry_distance", unit="m", values=np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    )


@pytest.fixture
def axis_b() -> PlayabilityAxis:
    """Attack angle, three stations, weights [0.25, 0.5, 0.25]."""
    return PlayabilityAxis(
        name="attack_angle", unit="rad", values=np.array([0.0, 0.5, 1.0])
    )


class TestPlayabilityAxis:
    """The axis, and the trapezoidal weights the area is built from."""

    def test_weights_sum_to_the_span(self, axis_a) -> None:
        """[0.5, 1, 1, 1, 0.5] sums to 4, which is the span 4 - 0."""
        np.testing.assert_allclose(axis_a.node_weights, [0.5, 1.0, 1.0, 1.0, 0.5])
        assert axis_a.node_weights.sum() == pytest.approx(axis_a.span)

    def test_non_uniform_weights_are_the_half_neighbour_spacings(self) -> None:
        """For [0, 1, 3] the weights are [0.5, 1.5, 1.0] and sum to 3."""
        axis = PlayabilityAxis(name="x", unit="m", values=np.array([0.0, 1.0, 3.0]))

        np.testing.assert_allclose(axis.node_weights, [0.5, 1.5, 1.0])
        assert axis.node_weights.sum() == pytest.approx(3.0)

    def test_a_decreasing_axis_is_refused(self) -> None:
        """Negative weights would make an area that shrinks as the window grows."""
        with pytest.raises(ValueError, match="strictly increasing"):
            PlayabilityAxis(name="x", unit="m", values=np.array([1.0, 0.0]))


class TestPlayabilityWindow:
    """Area, fraction, connectivity, and the refusal bookkeeping."""

    def test_a_fully_acceptable_grid_returns_the_whole_domain(
        self, axis_a, axis_b
    ) -> None:
        """Every node in the band: area = span_a * span_b = 4 * 1 = 4."""
        carry = np.full((5, 3), GOOD_CARRY_M)

        window = playability_window(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M
        )

        assert window.area == pytest.approx(4.0, rel=1e-12)
        assert window.domain_area == pytest.approx(4.0, rel=1e-12)
        assert window.fraction == pytest.approx(1.0, rel=1e-12)
        assert window.area_unit == "m.rad"

    def test_one_acceptable_column_carries_that_column_weight(
        self, axis_a, axis_b
    ) -> None:
        """Only b = 0.5 acceptable: area = (sum of A weights) * 0.5 = 4 * 0.5 = 2."""
        carry = np.full((5, 3), BAD_CARRY_M)
        carry[:, 1] = GOOD_CARRY_M

        window = playability_window(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M
        )

        assert window.area == pytest.approx(2.0, rel=1e-12)
        assert window.fraction == pytest.approx(0.5, rel=1e-12)

    def test_a_single_corner_node_carries_only_its_own_weight(
        self, axis_a, axis_b
    ) -> None:
        """The (0, 0) corner weighs 0.5 * 0.25 = 0.125."""
        carry = np.full((5, 3), BAD_CARRY_M)
        carry[0, 0] = GOOD_CARRY_M

        window = playability_window(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M
        )

        assert window.area == pytest.approx(0.125, rel=1e-12)

    def test_the_band_edge_is_inclusive_at_exactly_ten_percent(
        self, axis_a, axis_b
    ) -> None:
        """33.0 m is exactly +10 % of 30 m and counts; 33.1 m does not."""
        carry = np.full((5, 3), 33.0)
        carry[0, 0] = 33.1

        window = playability_window(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M
        )

        assert window.carry_band_m == pytest.approx((27.0, 33.0))
        assert not window.in_window[0, 0]
        assert window.area == pytest.approx(4.0 - 0.125, rel=1e-12)

    def test_a_refused_solve_is_outside_the_window_and_counted(
        self, axis_a, axis_b
    ) -> None:
        """NaN carry means the solver refused; it never counts as playable."""
        carry = np.full((5, 3), GOOD_CARRY_M)
        carry[0, 0] = np.nan

        window = playability_window(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M
        )

        assert not window.in_window[0, 0]
        assert window.area == pytest.approx(4.0 - 0.125, rel=1e-12)
        assert window.refused_fraction == pytest.approx(0.125 / 4.0, rel=1e-12)

    def test_scattered_islands_are_separated_from_the_total(
        self, axis_a, axis_b
    ) -> None:
        """Two isolated nodes: total area 0.25, largest connected region 0.125."""
        carry = np.full((5, 3), BAD_CARRY_M)
        carry[0, 0] = GOOD_CARRY_M
        carry[4, 2] = GOOD_CARRY_M

        window = playability_window(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M
        )

        assert window.area == pytest.approx(0.25, rel=1e-12)
        assert window.largest_connected_area == pytest.approx(0.125, rel=1e-12)

    def test_a_connected_block_is_reported_whole(self, axis_a, axis_b) -> None:
        """Two adjacent nodes in a column: 0.5 * 0.25 + 1.0 * 0.25 = 0.375."""
        carry = np.full((5, 3), BAD_CARRY_M)
        carry[0:2, 0] = GOOD_CARRY_M

        window = playability_window(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M
        )

        assert window.largest_connected_area == pytest.approx(0.375, rel=1e-12)
        assert window.largest_connected_area == pytest.approx(window.area)

    def test_the_nominal_delivery_is_checked_at_the_nearest_node(
        self, axis_a, axis_b
    ) -> None:
        """A window that excludes the nominal delivery is not a usable window."""
        carry = np.full((5, 3), BAD_CARRY_M)
        carry[2, 1] = GOOD_CARRY_M

        inside = playability_window(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M, nominal=(2.1, 0.45)
        )
        outside = playability_window(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M, nominal=(0.0, 0.0)
        )

        assert inside.contains_nominal is True
        assert outside.contains_nominal is False

    def test_an_empty_window_scores_zero_rather_than_failing(
        self, axis_a, axis_b
    ) -> None:
        """A design that never lands the shot gets a zero objective, not an error."""
        carry = np.full((5, 3), BAD_CARRY_M)

        window = playability_window(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M
        )

        assert window.area == 0.0
        assert window.largest_connected_area == 0.0

    def test_shape_target_and_tolerance_are_validated(self, axis_a, axis_b) -> None:
        """Bad inputs raise rather than silently reshaping or clamping."""
        with pytest.raises(ValueError, match="carry_m must have shape"):
            playability_window(
                axis_a, axis_b, np.zeros((3, 5)), target_carry_m=TARGET_CARRY_M
            )
        with pytest.raises(ValueError, match="target_carry_m must be positive"):
            playability_window(axis_a, axis_b, np.zeros((5, 3)), target_carry_m=0.0)
        with pytest.raises(ValueError, match="tolerance_fraction must be in"):
            playability_window(
                axis_a,
                axis_b,
                np.zeros((5, 3)),
                target_carry_m=TARGET_CARRY_M,
                tolerance_fraction=1.5,
            )


class TestPlayabilityObjective:
    """The scalar handed to the optimiser."""

    def test_the_objective_is_the_window_area(self, axis_a, axis_b) -> None:
        """One acceptable column of the 4 x 1 domain scores 2.0."""
        carry = np.full((5, 3), BAD_CARRY_M)
        carry[:, 1] = GOOD_CARRY_M

        assert playability_objective(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M
        ) == pytest.approx(2.0, rel=1e-12)

    def test_connected_only_refuses_to_reward_islands(self, axis_a, axis_b) -> None:
        """Same total area, but split in two: the connected score is halved."""
        carry = np.full((5, 3), BAD_CARRY_M)
        carry[0, 0] = GOOD_CARRY_M
        carry[4, 2] = GOOD_CARRY_M

        total = playability_objective(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M
        )
        connected = playability_objective(
            axis_a, axis_b, carry, target_carry_m=TARGET_CARRY_M, connected_only=True
        )

        assert total == pytest.approx(0.25, rel=1e-12)
        assert connected == pytest.approx(0.125, rel=1e-12)

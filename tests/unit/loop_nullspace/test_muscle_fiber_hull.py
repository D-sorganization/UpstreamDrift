"""UD-3 (#13463): the activation fibre's TRUE affine hull.

``dim = m - rank G`` holds only when a strictly box-feasible point exists.  The
counterexample ``G = [1 1 0]`` (here ``[40 40 0]`` N m, a third muscle with a zero
moment arm), ``a in [0,1]^3`` and the extreme torque ``b = 80`` N m forces
``a1 = a2 = 1``: the fibre is the segment ``{(1, 1, t)}`` of hull dimension one
although ``dim ker G = 2``.  The old sampler read the zero Chebyshev radius of the
kernel parametrisation as a singleton and repeated one point.

Tolerances.  ``HULL_TOL = 1e-8`` (activation units) is ten times the 1e-9
accuracy with which HiGHS, tightened to ``1e-10``, reports an LP endpoint; the
thin-fibre tests sit a factor of ten either side of it (``1e-6`` and ``1e-9``).
Feasibility of samples is checked to ``1e-9``, the module's own ``contains``
default.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.muscle_fiber import (
    HULL_TOL,
    CocontractionIndex,
    FiberInfeasibleError,
    activation_fiber,
    total_activation,
)
from src.shared.python.loop_nullspace.muscle_frame import MuscleFrame, TendonKind
from tests.unit.loop_nullspace._muscle_fixtures import (
    INPUT,
    MODEL,
    elbow_frame,
    pair_frame,
)

pytestmark = pytest.mark.unit

FEAS_TOL = 1e-9
E3 = CocontractionIndex("e3", np.array([0.0, 0.0, 1.0]))


def _segment_frame() -> MuscleFrame:
    """``G = [40 40 0]``: muscle three produces no torque about the joint."""
    return MuscleFrame(
        names=("flexor", "brachialis", "bystander"),
        coordinates=("joint",),
        moment_arms=np.array([[0.04], [0.04], [0.0]]),
        activation_grid=np.array([0.0, 1.0]),
        force_grid=np.array([np.zeros(3), np.full(3, 1000.0)]),
        max_isometric_force=np.full(3, 1000.0),
        tendon=TendonKind.RIGID,
        engine="fixture",
        model_provenance=MODEL,
        input_provenance=INPUT,
    )


def test_segment_counterexample_has_hull_dimension_one_not_two() -> None:
    fib = activation_fiber(_segment_frame(), np.array([80.0]))
    assert (fib.rank, fib.nullity) == (1, 2)  # dim ker G is still two
    assert fib.hull_dimension == 1  # ... but the fibre is a segment
    assert not fib.is_degenerate


def test_segment_samples_vary_along_e3_and_stay_feasible() -> None:
    fib = activation_fiber(_segment_frame(), np.array([80.0]))
    result = fib.sample_result(300, rng=np.random.default_rng(5))
    s = result.samples
    assert s.shape == (300, 3)
    assert not result.is_singleton and result.hull_dimension == 1
    assert np.ptp(s[:, 2]) > 0.8  # explored, not one repeated point
    np.testing.assert_allclose(s[:, :2], 1.0, atol=FEAS_TOL)
    assert all(fib.contains(a, tol=FEAS_TOL) for a in s)
    assert fib.sample(4, rng=np.random.default_rng(5)).shape == (4, 3)


def test_sampling_is_reproducible_for_a_seeded_generator() -> None:
    fib = activation_fiber(_segment_frame(), np.array([80.0]))
    a = fib.sample(20, rng=np.random.default_rng(1))
    b = fib.sample(20, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


def test_singleton_is_explicit_only_for_hull_dimension_zero() -> None:
    fib = activation_fiber(elbow_frame(), np.array([64.0]))
    assert fib.hull_dimension == 0 and fib.is_degenerate
    result = fib.sample_result(5, rng=np.random.default_rng(0))
    assert result.is_singleton and result.hull_dimension == 0
    assert np.ptp(result.samples, axis=0).max() == 0.0
    assert fib.contains(result.samples[0], tol=FEAS_TOL)


def test_full_rank_square_system_is_a_singleton_with_zero_nullity() -> None:
    fib = activation_fiber(pair_frame(), np.array([40.0]))  # a = (1, 0)
    assert fib.hull_dimension == 0
    np.testing.assert_allclose(
        fib.sample(3, rng=np.random.default_rng(0)), [[1.0, 0.0]] * 3, atol=FEAS_TOL
    )


def test_full_dimensional_fibre_keeps_dimension_m_minus_rank() -> None:
    fib = activation_fiber(elbow_frame(), np.array([20.0]))
    assert fib.hull_dimension == fib.nullity == 2
    result = fib.sample_result(50, rng=np.random.default_rng(2))
    assert not result.is_singleton and result.hull_dimension == 2


def test_hull_tolerance_separates_thin_from_collapsed_fibres() -> None:
    # a1 + a2 = 2 - delta with delta = 1e-6: both coordinates range over 1e-6
    # (> HULL_TOL), so the hull keeps (a1, a2, a3) minus one equality: dimension 2.
    wide = activation_fiber(_segment_frame(), np.array([80.0 - 40.0 * 1e-6]))
    assert wide.hull_dimension == 2
    # delta = 1e-9 < HULL_TOL: a1, a2 are fixed, the hull is the segment again.
    thin = activation_fiber(_segment_frame(), np.array([80.0 - 40.0 * 1e-9]))
    assert thin.hull_dimension == 1
    assert 1e-9 < HULL_TOL < 1e-6


def test_level_subfibre_of_e3_is_a_singleton() -> None:
    fib = activation_fiber(_segment_frame(), np.array([80.0]))
    assert fib.level_subfibre_dimension(E3, 0.5) == 0  # not m - rank - 1 = 1
    np.testing.assert_allclose(
        fib.point_at_level(E3, 0.5), [1.0, 1.0, 0.5], atol=FEAS_TOL
    )


def test_level_subfibre_dimension_is_m_minus_rank_minus_one_when_generic() -> None:
    fib = activation_fiber(elbow_frame(), np.array([20.0]))
    index = total_activation(3)
    lo, hi = fib.index_range(index)
    # w = 1 is not in range(G^T) and the mid level is strictly inside (lo, hi).
    assert fib.level_subfibre_dimension(index, 0.5 * (lo + hi)) == 1


def test_level_subfibre_rejects_levels_outside_the_range() -> None:
    fib = activation_fiber(elbow_frame(), np.array([20.0]))
    lo, _ = fib.index_range(total_activation(3))
    with pytest.raises(FiberInfeasibleError, match="level"):
        fib.level_subfibre_dimension(total_activation(3), lo - 0.5)
    with pytest.raises(ValueError, match="finite"):
        fib.level_subfibre_dimension(total_activation(3), float("nan"))


def test_sample_preconditions() -> None:
    fib = activation_fiber(elbow_frame(), np.array([20.0]))
    with pytest.raises(ValueError, match="burn_in"):
        fib.sample_result(0, rng=np.random.default_rng(0))
    with pytest.raises(TypeError, match="Generator"):
        fib.sample_result(3, rng=0)  # type: ignore[arg-type]

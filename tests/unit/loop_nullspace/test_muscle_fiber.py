"""NSX-94 (#12723): the activation fibre at fixed net torque.

Tolerances (derived):

* Torque invariance.  ``tau = G a`` is a sum of ``m`` products each of size at
  most ``|G_k|``; every operation carries a relative error of at most ``eps``,
  so ``|G a - b| <= (m + 2) eps sum_k |G_k|`` for fibre points built from the
  null basis.  For the elbow ``sum |G_k| = 106`` N m and ``eps = 2.2e-16``, i.e.
  ``1.2e-13``; the issue's ``1e-12`` carries a factor eight of headroom for
  the SVD null basis error.
* Vertices are enumerated exactly (rational pairs of active bounds), so the
  level range is compared to ``1e-9``, which is HiGHS' tightened feasibility
  tolerance (``1e-10``) with a safety factor of ten.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import minimize

from src.shared.python.loop_nullspace.muscle_fiber import (
    CocontractionIndex,
    FiberInfeasibleError,
    activation_fiber,
    total_activation,
)
from src.shared.python.loop_nullspace.muscle_frame import (
    MuscleFrame,
    NotQualifiedError,
    TendonKind,
)
from tests.unit.loop_nullspace._muscle_fixtures import (
    INPUT,
    MODEL,
    PAIR_ARM,
    PAIR_FORCE,
    elbow_frame,
    pair_frame,
)

pytestmark = pytest.mark.unit

TORQUE_TOL = 1e-12


def _elbow_vertices(target: float) -> np.ndarray:
    """Exact vertices of ``{a in [0,1]^3 : [40 24 -42] a = target}``."""
    g = np.array([40.0, 24.0, -42.0])
    verts = []
    for free in range(3):
        fixed = [i for i in range(3) if i != free]
        for bounds in itertools.product((0.0, 1.0), repeat=2):
            a = np.zeros(3)
            a[fixed] = bounds
            a[free] = (target - g[fixed] @ a[fixed]) / g[free]
            if -1e-12 <= a[free] <= 1 + 1e-12:
                verts.append(a)
    return np.array(verts)


def test_elbow_fibre_dimension_and_rank() -> None:
    fib = activation_fiber(elbow_frame(), np.array([20.0]))
    assert (fib.rank, fib.nullity) == (1, 2)
    assert fib.reserve_fraction == pytest.approx(0.0, abs=1e-12)


def test_torque_is_invariant_across_the_fibre_to_1e12() -> None:
    frame = elbow_frame()
    fib = activation_fiber(frame, np.array([20.0]))
    samples = fib.sample(200, rng=np.random.default_rng(7))
    assert samples.shape == (200, 3)
    assert samples.min() >= -1e-12 and samples.max() <= 1 + 1e-12
    torques = np.array([frame.torque(a)[0] for a in samples])
    assert np.max(np.abs(torques - 20.0)) < TORQUE_TOL
    # The samples genuinely differ: the fibre is explored, not a single point.
    assert np.ptp(samples @ np.ones(3)) > 0.3


def test_sampler_covers_the_exact_vertex_hull() -> None:
    fib = activation_fiber(elbow_frame(), np.array([20.0]))
    verts = _elbow_vertices(20.0)
    levels = verts.sum(axis=1)
    samples = fib.sample(4000, rng=np.random.default_rng(3), thin=3)
    got = samples.sum(axis=1)
    assert got.min() >= levels.min() - 1e-9 and got.max() <= levels.max() + 1e-9
    # Sampled coverage of the exact range (not a certified bound).
    assert got.min() < levels.min() + 0.1 * np.ptp(levels)
    assert got.max() > levels.max() - 0.1 * np.ptp(levels)


@pytest.mark.parametrize("target", [5.0, 20.0, 40.0])
def test_level_range_matches_exact_vertex_enumeration(target: float) -> None:
    fib = activation_fiber(elbow_frame(), np.array([target]))
    lo, hi = fib.index_range(total_activation(3))
    levels = _elbow_vertices(target).sum(axis=1)
    assert lo == pytest.approx(levels.min(), abs=1e-9)
    assert hi == pytest.approx(levels.max(), abs=1e-9)


def test_point_at_level_matches_pair_closed_form() -> None:
    frame = pair_frame()
    tau = 8.0
    d = tau / (PAIR_ARM * PAIR_FORCE)  # a1 - a2 = 0.2
    fib = activation_fiber(frame, np.array([tau]))
    index = total_activation(2)
    lo, hi = fib.index_range(index)
    assert lo == pytest.approx(d, abs=1e-9) and hi == pytest.approx(2 - d, abs=1e-9)
    for level in (0.3, 1.0, 1.7):
        a = fib.point_at_level(index, level)
        np.testing.assert_allclose(a, [(level + d) / 2, (level - d) / 2], atol=1e-12)


def test_point_at_level_is_the_minimum_norm_point_of_the_sub_fibre() -> None:
    frame = elbow_frame()
    fib = activation_fiber(frame, np.array([20.0]))
    index = total_activation(3)
    lo, hi = fib.index_range(index)
    level = 0.5 * (lo + hi)
    a = fib.point_at_level(index, level)
    assert abs(frame.torque(a)[0] - 20.0) < TORQUE_TOL
    assert abs(a.sum() - level) < TORQUE_TOL
    assert a.min() >= -1e-12 and a.max() <= 1 + 1e-12
    cons = [
        {"type": "eq", "fun": lambda x: frame.torque(x)[0] - 20.0},
        {"type": "eq", "fun": lambda x: x.sum() - level},
    ]
    ref = minimize(
        lambda x: float(x @ x),
        np.full(3, level / 3),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints=cons,
        options={"ftol": 1e-14, "maxiter": 500},
    )
    assert ref.success
    np.testing.assert_allclose(a, ref.x, atol=1e-6)


def test_fibre_collapses_at_the_boundary_of_the_reachable_torque() -> None:
    fib = activation_fiber(elbow_frame(), np.array([64.0]))
    lo, hi = fib.index_range(total_activation(3))
    assert hi - lo < 1e-7
    assert fib.is_degenerate


def test_level_outside_the_range_is_infeasible() -> None:
    fib = activation_fiber(pair_frame(), np.array([8.0]))
    with pytest.raises(FiberInfeasibleError, match="level"):
        fib.point_at_level(total_activation(2), 0.05)


def test_unqualified_frame_fails_closed() -> None:
    with pytest.raises(NotQualifiedError, match="reserve"):
        activation_fiber(elbow_frame(), np.array([80.0]))


def test_index_validation() -> None:
    with pytest.raises(ValueError, match="weights"):
        CocontractionIndex("bad", np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="weights"):
        CocontractionIndex("zero", np.zeros(3))
    fib = activation_fiber(elbow_frame(), np.array([20.0]))
    with pytest.raises(ValueError, match="weights"):
        fib.index_range(total_activation(2))


def _slsqp_min_norm(
    g: np.ndarray, tau: np.ndarray, level: float, start: np.ndarray
) -> tuple[bool, np.ndarray]:
    res = minimize(
        lambda x: float(x @ x),
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * start.size,
        constraints=[
            {"type": "eq", "fun": lambda x: g @ x - tau},
            {"type": "eq", "fun": lambda x: x.sum() - level},
        ],
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    return bool(res.success), np.asarray(res.x)


def _random_frame(rng: np.random.Generator) -> MuscleFrame:
    m, n = int(rng.integers(4, 8)), int(rng.integers(1, 3))
    force = rng.uniform(200.0, 1200.0, m)
    return MuscleFrame(
        names=tuple(f"m{i}" for i in range(m)),
        coordinates=tuple(f"c{i}" for i in range(n)),
        moment_arms=rng.normal(size=(m, n)) * 0.04,
        activation_grid=np.array([0.0, 1.0]),
        force_grid=np.vstack([np.zeros(m), force]),
        max_isometric_force=force,
        tendon=TendonKind.RIGID,
        engine="fixture",
        model_provenance=MODEL,
        input_provenance=INPUT,
    )


def test_dual_projection_matches_slsqp_on_random_polytopes() -> None:
    rng = np.random.default_rng(11)
    compared = 0
    for _ in range(25):
        frame = _random_frame(rng)
        m = frame.n_muscles
        a_true = rng.uniform(0.15, 0.85, m)
        tau = frame.torque(a_true)
        fib = activation_fiber(frame, tau)
        index = total_activation(m)
        lo, hi = fib.index_range(index)
        level = 0.5 * (lo + hi)
        a = fib.point_at_level(index, level)
        assert np.max(np.abs(frame.torque(a) - tau)) < 1e-9
        assert abs(a.sum() - level) < 1e-9
        assert a.min() >= -1e-12 and a.max() <= 1 + 1e-12
        ok, ref = _slsqp_min_norm(frame.gain_matrix, tau, level, a_true)
        if ok:
            compared += 1
            # The dual solution is optimal: never worse than SLSQP's.
            assert float(a @ a) <= float(ref @ ref) + 1e-8
    assert compared >= 15

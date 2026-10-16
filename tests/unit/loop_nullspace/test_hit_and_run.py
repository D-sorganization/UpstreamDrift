"""Sampler invariance tests on oracles (NSX-70, #12588).

Every oracle is a set with a closed-form uniform distribution:

* the unit box ``[0, 1]^3`` (mean 1/2, variance 1/12, zero covariance);
* the standard simplex ``z >= 0, sum z <= 1`` in 3-D (mean 1/4, variance
  ``d / ((d + 1)^2 (d + 2))``, covariance ``-1 / ((d + 1)^2 (d + 2))``);
* the unit disk as a second-order-cone block (mean 0, covariance ``I / 4``);
* two frames sharing one constant-in-time coefficient, so the coefficient is
  uniform on the intersection ``[-1/2, 1]`` of the per-frame boxes.

Monte-Carlo error is measured from independent chains (the standard error of
the chain means), not from the module's own effective-sample-size code.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("highspy")

from src.shared.python.loop_nullspace.feasible_rows import (  # noqa: E402
    ConeBlock,
    FeasibleRow,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet  # noqa: E402
from src.shared.python.loop_nullspace.hit_and_run import (  # noqa: E402
    TrajectoryDomain,
    bspline_temporal_basis,
    hit_and_run,
    interior_point,
    run_chains,
)

pytestmark = pytest.mark.unit

N_CHAINS = 16
N_DRAWS = 6000
NSIGMA = 5.0


def _rows(g: np.ndarray, h: np.ndarray) -> tuple[FeasibleRow, ...]:
    return tuple(
        FeasibleRow(g[i], float(h[i]), f"R{i}", "H-TORQUE-CAPACITY", "torque")
        for i in range(len(h))
    )


def _box(d: int, lo: float = 0.0, hi: float = 1.0) -> FeasibleSet:
    g = np.vstack([np.eye(d), -np.eye(d)])
    h = np.concatenate([np.full(d, hi), np.full(d, -lo)])
    return FeasibleSet.from_rows(d, _rows(g, h))


def _simplex(d: int) -> FeasibleSet:
    g = np.vstack([-np.eye(d), np.ones((1, d))])
    h = np.concatenate([np.zeros(d), [1.0]])
    return FeasibleSet.from_rows(d, _rows(g, h))


def _disk() -> FeasibleSet:
    cone = ConeBlock(
        A=np.vstack([np.zeros((1, 2)), -np.eye(2)]),
        b=np.array([1.0, 0.0, 0.0]),
        label="C0",
        hypothesis_id="H-FRICTION-CONE",
    )
    return FeasibleSet.from_rows(2, (), (cone,))


def _domain(fs: FeasibleSet, frames: int = 1, basis=None) -> TrajectoryDomain:
    b = np.ones((frames, 1)) if basis is None else basis
    return TrajectoryDomain([fs] * frames, b)


def _pooled(domain: TrajectoryDomain, seed: int):
    chains = run_chains(
        domain, n_chains=N_CHAINS, n_draws=N_DRAWS, burn_in=500, seed=seed
    )
    arr = np.stack([c.coefficients for c in chains])  # (m, n, D)
    return arr


def _assert_moments(arr: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> None:
    m = arr.shape[0]
    chain_means = arr.mean(axis=1)
    se = chain_means.std(axis=0, ddof=1) / np.sqrt(m)
    got = chain_means.mean(axis=0)
    assert np.all(np.abs(got - mean) <= NSIGMA * se), (got, mean, se)
    pooled = arr.reshape(-1, arr.shape[2])
    chain_covs = np.stack([np.cov(c, rowvar=False) for c in arr])
    se_cov = chain_covs.std(axis=0, ddof=1) / np.sqrt(m)
    got_cov = np.cov(pooled, rowvar=False)
    assert np.all(np.abs(got_cov - cov) <= NSIGMA * se_cov + 1e-12), (got_cov, cov)


def test_uniform_box_mean_and_covariance_match_closed_form() -> None:
    arr = _pooled(_domain(_box(3)), seed=11)
    _assert_moments(arr, np.full(3, 0.5), np.eye(3) / 12.0)


def test_uniform_simplex_mean_and_covariance_match_closed_form() -> None:
    d = 3
    arr = _pooled(_domain(_simplex(d)), seed=12)
    var = d / ((d + 1) ** 2 * (d + 2))
    cov_off = -1.0 / ((d + 1) ** 2 * (d + 2))
    cov = np.full((d, d), cov_off) + np.eye(d) * (var - cov_off)
    _assert_moments(arr, np.full(d, 1.0 / (d + 1)), cov)


def test_uniform_disk_through_the_exact_cone_chord() -> None:
    arr = _pooled(_domain(_disk()), seed=13)
    _assert_moments(arr, np.zeros(2), np.eye(2) / 4.0)
    radius = np.linalg.norm(arr.reshape(-1, 2), axis=1)
    assert radius.max() <= 1.0 + 1e-12
    # P(r <= 1/2) = 1/4 for the uniform disk
    frac = float(np.mean(radius <= 0.5))
    per_chain = np.mean(radius.reshape(N_CHAINS, -1) <= 0.5, axis=1)
    se = per_chain.std(ddof=1) / np.sqrt(N_CHAINS)
    assert abs(frac - 0.25) <= NSIGMA * se


def test_temporal_coupling_restricts_the_coefficient_to_the_intersection() -> None:
    frames = [_box(1, -1.0, 1.0), _box(1, -0.5, 2.0)]
    domain = TrajectoryDomain(frames, np.ones((2, 1)))
    arr = _pooled(domain, seed=14)
    assert arr.min() >= -0.5 - 1e-12 and arr.max() <= 1.0 + 1e-12
    _assert_moments(arr, np.array([0.25]), np.array([[1.5**2 / 12.0]]))


def test_every_sample_satisfies_every_constraint() -> None:
    fs = _simplex(3)
    domain = _domain(fs, frames=3, basis=np.array([[1.0], [1.0], [1.0]]))
    chains = run_chains(domain, n_chains=4, n_draws=2000, burn_in=0, seed=21)
    for c in chains:
        for coeffs in c.coefficients:
            assert domain.contains(coeffs, tol=1e-9)
    z = domain.trajectory(chains[0].coefficients[-1])
    assert z.shape == (3, 3)


def test_chain_is_reproducible_with_its_seed_and_differs_across_seeds() -> None:
    domain = _domain(_box(3))
    start = interior_point(domain)
    a = hit_and_run(domain, start, n_draws=300, seed=7)
    b = hit_and_run(domain, start, n_draws=300, seed=7)
    c = hit_and_run(domain, start, n_draws=300, seed=8)
    assert np.array_equal(a.coefficients, b.coefficients)
    assert not np.array_equal(a.coefficients, c.coefficients)
    first = run_chains(domain, n_chains=3, n_draws=50, seed=5)
    second = run_chains(domain, n_chains=3, n_draws=50, seed=5)
    for u, v in zip(first, second, strict=True):
        assert np.array_equal(u.coefficients, v.coefficients)
    assert not np.array_equal(first[0].coefficients, first[1].coefficients)


def test_thinning_keeps_every_kth_state() -> None:
    domain = _domain(_box(2))
    start = interior_point(domain)
    full = hit_and_run(domain, start, n_draws=40, seed=3)
    thin = hit_and_run(domain, start, n_draws=10, seed=3, thin=4)
    assert np.array_equal(thin.coefficients, full.coefficients[3::4])


def test_preconditioned_directions_still_sample_the_uniform_box() -> None:
    d = 3
    domain = _domain(_box(d))
    scale = np.diag([1.0, 0.2, 3.0])
    chains = run_chains(
        domain,
        n_chains=N_CHAINS,
        n_draws=N_DRAWS,
        burn_in=500,
        seed=15,
        direction_scale=scale,
    )
    arr = np.stack([c.coefficients for c in chains])
    _assert_moments(arr, np.full(d, 0.5), np.eye(d) / 12.0)


def test_interior_point_is_strictly_inside() -> None:
    domain = _domain(_simplex(3))
    point = interior_point(domain)
    assert domain.min_slack(point) > 1e-3


def test_low_rank_basis_dimension_and_trajectory() -> None:
    times = np.linspace(0.0, 1.0, 21)
    basis = bspline_temporal_basis(times, 4, degree=3)
    assert basis.shape == (21, 4)
    assert np.allclose(basis.sum(axis=1), 1.0)
    domain = TrajectoryDomain([_box(2, -1.0, 1.0)] * 21, basis)
    assert domain.dimension == 8
    chain = hit_and_run(domain, interior_point(domain), n_draws=20, seed=1)
    assert chain.coefficients.shape == (20, 8)
    assert domain.trajectory(chain.coefficients[0]).shape == (21, 2)


def test_preconditions_raise() -> None:
    fs = _box(2)
    with pytest.raises(ValueError, match="at least one"):
        TrajectoryDomain([], np.ones((0, 1)))
    with pytest.raises(ValueError, match="one row per frame"):
        TrajectoryDomain([fs, fs], np.ones((3, 1)))
    with pytest.raises(TypeError, match="FeasibleSet"):
        TrajectoryDomain([object()], np.ones((1, 1)))  # type: ignore[list-item]
    with pytest.raises(ValueError, match="n_z"):
        TrajectoryDomain([fs, _box(3)], np.ones((2, 1)))
    domain = _domain(fs)
    start = interior_point(domain)
    with pytest.raises(ValueError, match="n_draws"):
        hit_and_run(domain, start, n_draws=0, seed=1)
    with pytest.raises(ValueError, match="thin"):
        hit_and_run(domain, start, n_draws=5, seed=1, thin=0)
    with pytest.raises(ValueError, match="strictly inside"):
        hit_and_run(domain, np.array([5.0, 5.0]), n_draws=5, seed=1)
    with pytest.raises(ValueError, match="start"):
        hit_and_run(domain, np.zeros(3), n_draws=5, seed=1)


def test_unbounded_and_empty_domains_are_rejected() -> None:
    half = FeasibleSet.from_rows(1, _rows(np.array([[1.0]]), np.array([1.0])))
    with pytest.raises(ValueError, match="unbounded"):
        interior_point(_domain(half))
    empty = FeasibleSet.from_rows(
        1, _rows(np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0]))
    )
    with pytest.raises(ValueError, match="no interior"):
        interior_point(_domain(empty))


# ------------------------------------------------------- envelope comparison
def test_marginal_reach_on_the_box_oracle() -> None:
    from src.shared.python.loop_nullspace.envelopes import (  # noqa: PLC0415
        EnvelopeQuantity,
        static_envelopes,
    )
    from src.shared.python.loop_nullspace.sampled_marginals import (  # noqa: PLC0415
        marginal_reach,
    )

    fs = _box(2)
    quantity = EnvelopeQuantity("sum", np.ones(2), 0.0, "N", 1.0)
    series = static_envelopes([fs], [quantity])
    assert series.lower[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert series.upper[0, 0] == pytest.approx(2.0, abs=1e-9)
    domain = _domain(fs)
    chain = run_chains(domain, n_chains=1, n_draws=20000, burn_in=200, seed=31)[0]
    z = domain.trajectories(chain.coefficients)
    out = marginal_reach(z, [[quantity]], series.lower, series.upper)
    assert out.containment_violation == 0.0
    assert out.names == ("sum",)
    # z1 + z2 is triangular on [0, 2]: CDF s^2 / 2 for s <= 1, so the 5th
    # percentile is sqrt(0.1) and the 95th is 2 - sqrt(0.1)
    assert out.sample_p05[0, 0] == pytest.approx(np.sqrt(2 * 0.05), abs=0.03)
    assert out.sample_p95[0, 0] == pytest.approx(2 - np.sqrt(2 * 0.05), abs=0.03)
    assert 0.97 <= out.reach[0, 0] <= 1.0
    narrow = marginal_reach(z, [[quantity]], series.lower + 0.5, series.upper - 0.5)
    assert narrow.containment_violation > 0.4


def test_marginal_reach_preconditions() -> None:
    from src.shared.python.loop_nullspace.envelopes import (  # noqa: PLC0415
        EnvelopeQuantity,
    )
    from src.shared.python.loop_nullspace.sampled_marginals import (  # noqa: PLC0415
        marginal_reach,
    )

    q = EnvelopeQuantity("a", np.ones(2), 0.0, "N", 1.0)
    z = np.zeros((3, 1, 2))
    with pytest.raises(ValueError, match="one quantity list"):
        marginal_reach(z, [[q], [q]], np.zeros((1, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError, match="shape"):
        marginal_reach(z, [[q]], np.zeros((2, 1)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="non-empty"):
        marginal_reach(np.zeros((0, 1, 2)), [[q]], np.zeros((1, 1)), np.ones((1, 1)))

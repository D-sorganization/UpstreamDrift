"""Synthetic tests of the centroidal feasibility filter numerics (#11669)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from src.shared.python.motion_matching.pipeline.centroidal_filter import (
    CentroidalFilterConfig,
    FrameLinearisation,
    ccw_hull,
    constraint_rows,
    constraint_system,
    objective_system,
    second_difference,
    solve_bounded_penalty,
    support_edges,
    time_basis,
    violation_merit,
    zmp_sensitivity,
)

pytestmark = pytest.mark.unit

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
MASS = 70.0
HEIGHT = 1.0


def _cart_zmp(com, force, moment):
    """Cart-table ZMP with a moment term, for a flat ground at z = 0."""
    com, force, moment = (np.asarray(v, dtype=float) for v in (com, force, moment))
    fz = force[2]
    return np.array(
        [
            com[0] - (com[2] * force[0] + moment[1]) / fz,
            com[1] - (com[2] * force[1] - moment[0]) / fz,
        ]
    )


def _lin(nv: int = 3) -> FrameLinearisation:
    rng = np.random.default_rng(3)
    return FrameLinearisation(
        com_jac=rng.normal(size=(3, nv)),
        momentum_jac=rng.normal(size=(3, nv)),
        marker_factor=np.eye(nv),
        closure_residual=np.zeros(0),
        closure_jac=np.zeros((0, nv)),
        foot_jac=np.zeros((0, nv)),
    )


def test_config_rejects_inconsistent_values() -> None:
    CentroidalFilterConfig()
    for bad in (
        {"t_on_s": 2.0, "t_con_s": 1.0},
        {"friction_mu": 0.0},
        {"min_load_bw": 1.0},
        {"bound_rad": 0.0},
        {"iterations": 0},
        {"margin_m": -0.1},
    ):
        with pytest.raises(ValueError):
            CentroidalFilterConfig(**bad)


def test_support_edges_shrink_the_polygon() -> None:
    normals, offsets = support_edges(SQUARE, 0.1)
    inside, outside = np.array([0.5, 0.5]), np.array([0.95, 0.5])
    assert np.all(normals @ inside <= offsets)
    assert np.any(normals @ outside > offsets)  # inside the hull, outside the margin
    with pytest.raises(ValueError, match="nonnegative"):
        support_edges(SQUARE, -0.1)
    with pytest.raises(ValueError, match="polygon"):
        support_edges(SQUARE[:2], 0.0)


def test_ccw_hull_orders_counter_clockwise() -> None:
    shuffled = SQUARE[[2, 0, 3, 1]]
    hull = ccw_hull(np.vstack([shuffled, [[0.5, 0.5]]]))
    area = 0.5 * np.sum(
        hull[:, 0] * np.roll(hull[:, 1], -1) - np.roll(hull[:, 0], -1) * hull[:, 1]
    )
    assert len(hull) == 4 and area > 0
    with pytest.raises(ValueError):
        ccw_hull(SQUARE[:2])


def test_time_basis_joins_the_reference_smoothly() -> None:
    times = np.linspace(0.9, 1.8, 91)
    basis = time_basis(times, 0.05)
    assert basis.shape[0] == len(times)
    assert np.allclose(basis[0], 0.0)  # zero correction at the first sample
    assert np.abs(basis[1]).max() < 0.1  # cubic onset: slope is zero too
    with pytest.raises(ValueError):
        time_basis(times[:4], 0.05)


def test_solver_satisfies_a_halfspace_and_respects_the_box() -> None:
    objective = sp.csr_matrix(np.eye(2))
    rows = sp.csr_matrix(np.array([[-1.0, 0.0]]))  # x0 >= 0.5
    limits = np.array([-0.5])
    wide = solve_bounded_penalty(
        objective, np.zeros(2), rows, limits, CentroidalFilterConfig(bound_rad=1.0)
    )
    assert wide[0] == pytest.approx(0.5, abs=0.02) and abs(wide[1]) < 1e-9
    tight = solve_bounded_penalty(
        objective, np.zeros(2), rows, limits, CentroidalFilterConfig(bound_rad=0.2)
    )
    assert tight[0] == pytest.approx(0.2, abs=1e-6)
    with pytest.raises(ValueError):
        solve_bounded_penalty(
            objective, np.zeros(3), rows, limits, CentroidalFilterConfig()
        )


def test_violation_merit_counts_each_violation_kind() -> None:
    times = np.array([0.0, 1.0, 1.1, 1.2])
    clean = {
        "unloaded": np.zeros(4, bool),
        "outside_m": np.zeros(4),
        "grf_over_weight": np.tile([0.0, 0.0, 1.0], (4, 1)),
    }
    cfg = CentroidalFilterConfig()
    assert violation_merit(clean, times, cfg) == 0.0
    bad = {**clean, "outside_m": np.array([5.0, 0.0, 0.2, 0.0])}
    assert violation_merit(bad, times, cfg) == pytest.approx(0.04)  # t=0 ignored
    light = {**clean, "grf_over_weight": np.tile([0.0, 0.0, 0.2], (4, 1))}
    assert violation_merit(light, times, cfg) == pytest.approx(3 * 0.3**2)
    slide = {**clean, "grf_over_weight": np.tile([1.0, 0.0, 1.0], (4, 1))}
    assert violation_merit(slide, times, cfg) == pytest.approx(3 * (1.0 - 0.6) ** 2)
    unloaded = {**clean, "unloaded": np.array([False, False, True, False])}
    assert violation_merit(unloaded, times, cfg) == pytest.approx(1.0)


def test_zmp_sensitivity_matches_the_cart_table_derivative() -> None:
    wrench = np.array([0.1, 0.2, HEIGHT, 30.0, -10.0, 700.0, 5.0, 8.0, 0.0])
    sens = zmp_sensitivity(wrench, _cart_zmp)
    assert sens[0, 0] == pytest.approx(1.0, abs=1e-6)  # d zmp_x / d com_x
    assert sens[0, 3] == pytest.approx(-HEIGHT / 700.0, rel=1e-4)  # d zmp_x / d Fx
    with pytest.raises(ValueError):
        zmp_sensitivity(np.zeros(8), _cart_zmp)


def test_linearised_rows_predict_the_nonlinear_change() -> None:
    cfg = CentroidalFilterConfig()
    lin = _lin()
    wrench = np.array([0.5, 0.5, HEIGHT, 20.0, 10.0, 686.0, 3.0, -2.0, 0.0])
    base = _cart_zmp(*np.split(wrench, 3))
    sens = zmp_sensitivity(wrench, _cart_zmp)
    rows = constraint_rows(lin, wrench, base, SQUARE, sens, MASS, cfg)
    assert [r[3] for r in rows].count("zmp") == 4
    assert [r[3] for r in rows].count("cone") == 8 and rows[-9][3] == "fz"
    rng = np.random.default_rng(5)
    dq, ddq = 1e-4 * rng.normal(size=3), 1e-3 * rng.normal(size=3)
    moved = (
        wrench
        + np.r_[lin.com_jac @ dq, MASS * lin.com_jac @ ddq, lin.momentum_jac @ ddq]
    )
    normals, _ = support_edges(SQUARE, cfg.margin_m)
    delta = _cart_zmp(*np.split(moved, 3)) - base
    for (pos, acc, _, kind), normal in zip(rows[:4], normals, strict=True):
        assert kind == "zmp"
        assert pos @ dq + acc @ ddq == pytest.approx(normal @ delta, rel=2e-2, abs=1e-6)


def test_step_pulls_an_outside_zmp_back_inside_and_stays_near_zero() -> None:
    nv, frames = 3, 24
    times = np.linspace(1.0, 1.4, frames)
    cfg = CentroidalFilterConfig(
        t_on_s=1.0, t_con_s=1.0, bound_rad=0.5, penalty=1e4, ridge=1e-3
    )
    lin = _lin(nv)
    lin = FrameLinearisation(
        com_jac=np.vstack([np.eye(nv)[:2], np.zeros((1, nv))]),
        momentum_jac=np.zeros((3, nv)),
        marker_factor=np.eye(nv),
        closure_residual=np.zeros(0),
        closure_jac=np.zeros((0, nv)),
        foot_jac=np.zeros((0, nv)),
    )
    wrench = np.array([1.2, 0.5, HEIGHT, 0.0, 0.0, 686.0, 0.0, 0.0, 0.0])  # zmp x = 1.2
    base = _cart_zmp(*np.split(wrench, 3))
    sens = zmp_sensitivity(wrench, _cart_zmp)
    rows = [constraint_rows(lin, wrench, base, SQUARE, sens, MASS, cfg)] * frames
    accel = second_difference(times)
    mat, limits, kinds = constraint_system(rows, accel, nv)
    assert len(kinds) == len(limits) == mat.shape[0] == frames * 13
    objective, target = objective_system([lin] * frames, accel, cfg)
    basis = time_basis(times, 0.05)
    lift = sp.kron(sp.csr_matrix(basis), sp.identity(nv), format="csr")
    before = (mat @ np.zeros(frames * nv) - limits).max()
    coeffs = solve_bounded_penalty(objective @ lift, target, mat @ lift, limits, cfg)
    after = (mat @ (lift @ coeffs) - limits).max()
    assert before > 0.1 and after < 0.5 * before
    assert np.abs(coeffs).max() <= 0.5 + 1e-9


def test_stable_cholesky_survives_a_semidefinite_matrix() -> None:
    from src.shared.python.motion_matching.pipeline.centroidal_filter import (
        _stable_cholesky,
    )

    vec = np.array([[1.0, 2.0, 3.0]])
    semidefinite = 1e8 * (vec.T @ vec)  # rank one: plain Cholesky fails
    factor = _stable_cholesky(semidefinite)
    assert np.allclose(factor @ factor.T, semidefinite, rtol=1e-3, atol=1e-2)
    with pytest.raises(ValueError, match="positive definite"):
        _stable_cholesky(-np.eye(3))

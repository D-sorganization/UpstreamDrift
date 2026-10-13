"""Unit tests for vectorised multi-trial batching, time-parallel Riccati, and runtime receipts."""

from __future__ import annotations

import json
import numpy as np
import pytest

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.estimation.mosaic.inertial import PlanarParameterization
from src.shared.python.estimation.mosaic.kinematic_basis import BSplineBasis
from src.shared.python.estimation.mosaic.local_policy import (
    LinearizedDynamics,
    linearize_along_trajectory,
    tvlqr_gains,
)
from src.shared.python.estimation.mosaic.multi_trial_batching import (
    BlockArrowSolution,
    RuntimeReceipt,
    assemble_block_arrow,
    extract_block_arrow,
    generate_runtime_receipt,
    sequential_block_arrow_schur,
    solve_block_arrow_schur,
    time_parallel_riccati,
)
from src.shared.python.estimation.mosaic.outer_solve import (
    OuterOptions,
    OuterPriors,
    TrialData,
    _Context,
    _evaluate,
    _reduced_system,
)
from src.shared.python.estimation.mosaic.planar_chain import (
    PlanarChain,
    PlanarMarkerSet,
)

pytestmark = pytest.mark.unit

ACTUATED = np.array([False, True, True])
TRUE_LENGTHS = np.array([0.9, 0.6, 0.4])
TRUE_PI = np.array(
    [1.3, 0.585, 0.0, 0.343, 0.7, 0.175, 0.0, 0.064, 0.3, 0.09, 0.0, 0.03]
)
MARKERS = PlanarMarkerSet(
    link_index=np.array([0, 0, 1, 1, 2, 2]),
    offsets=np.array(
        [[0.3, 0.02], [0.8, -0.03], [0.2, 0.0], [0.55, 0.04], [0.1, 0.0], [0.35, -0.02]]
    ),
)


def _make_chain(lengths: np.ndarray) -> PlanarChain:
    return PlanarChain(link_lengths=lengths, actuated=ACTUATED)


def _synthetic_trials(
    n_trials: int, n_nodes: int = 40, seed: int = 8
) -> list[TrialData]:
    rng = np.random.default_rng(seed)
    chain = _make_chain(TRUE_LENGTHS)
    dt = 1.0 / 200.0
    times = (np.arange(n_nodes) * dt).astype(np.float64)
    u = np.zeros((n_trials, n_nodes, 2))
    hanging = np.array([-np.pi / 2 + 0.4, 0.3, -0.2, 0.0, 0.0, 0.0])
    x0 = np.tile(hanging, (n_trials, 1))
    q, _ = chain.rollout(x0, u[:, :-1], TRUE_PI, dt)
    trials = []
    for k in range(n_trials):
        clean = chain.marker_positions(q[k], MARKERS)
        basis = BSplineBasis.uniform(times, n_coefficients=10, degree=4)
        trials.append(TrialData(times, clean, MARKERS, 1e-3, basis, np.arange(n_nodes)))
    return trials


@pytest.mark.parametrize("n_trials", [1, 2, 8, 32])
def test_batched_block_arrow_schur_matches_sequential_planar_fixture(
    n_trials: int,
) -> None:
    """Batched block-arrow Schur solve matches sequential monolithic solve to machine tolerance."""
    trials = _synthetic_trials(n_trials, n_nodes=40)
    shapes = tuple((t.basis.n_coefficients, 3) for t in trials)
    param = PlanarParameterization()
    theta0 = param.to_unconstrained(TRUE_PI)
    priors = OuterPriors(
        geometry_mean=TRUE_LENGTHS,
        geometry_weight=np.ones(3),
        parameter_prior_mean=TRUE_PI,
        parameter_prior_weight=np.ones(12),
        dynamics_sigma=np.full(3, 0.2),
        input_effort_weight=0.1,
        input_smoothness_weight=1.0,
        template_weight=0.0,
    )
    options = OuterOptions()
    ctx = _Context(
        factory=_make_chain,
        trials=tuple(trials),
        priors=priors,
        options=options,
        parameterization=param,
        shapes=shapes,
        n_geometry=3,
        n_theta=theta0.size,
    )
    c_init = [np.zeros(s) for s in shapes]
    xi0 = ctx.pack(c_init, TRUE_LENGTHS, theta0)
    ev = _evaluate(xi0, ctx, factor=0.01)
    hessian, gradient = _reduced_system(ev, ctx, factor=0.01)
    # Outer solver Levenberg-Marquardt system
    scale = np.maximum(np.diag(hessian), 1e-12)
    damping = 0.01
    h_system = hessian + damping * np.diag(scale)

    # Monolithic sequential solve
    delta_mono = np.linalg.solve(h_system, -gradient)

    # Batched Schur solve
    trial_dims = [s[0] * s[1] for s in shapes]
    shared_dim = 3 + theta0.size
    system = extract_block_arrow(h_system, -gradient, trial_dims, shared_dim)
    sol_batched: BlockArrowSolution = solve_block_arrow_schur(
        system.h_kk, system.h_kpi, system.h_pipi, system.g_k, system.g_pi
    )
    sol_seq: BlockArrowSolution = sequential_block_arrow_schur(
        system.h_kk, system.h_kpi, system.h_pipi, system.g_k, system.g_pi
    )

    # Batched matches sequential multi-trial solution to machine tolerance
    np.testing.assert_allclose(sol_batched.full, sol_seq.full, atol=1e-10, rtol=1e-10)
    delta_c, delta_pi = sol_batched
    delta_c_seq, delta_pi_seq = sol_seq
    np.testing.assert_allclose(delta_pi, delta_pi_seq, atol=1e-10, rtol=1e-10)

    # Matches monolithic solve
    np.testing.assert_allclose(sol_batched.full, delta_mono, atol=1e-7, rtol=1e-7)

    # Linear system residual comparison
    res_schur = float(np.linalg.norm(h_system @ sol_batched.full - (-gradient)))
    res_mono = float(np.linalg.norm(h_system @ delta_mono - (-gradient)))
    assert res_schur < 1e-5
    assert res_schur <= res_mono + 1e-7


def test_block_arrow_schur_preconditions() -> None:
    """Preconditions enforced on invalid shapes and dimensions."""
    h_kk = np.eye(5)[None, ...]  # (1, 5, 5)
    h_kpi = np.ones((1, 5, 2))
    h_pipi = np.eye(2)
    g_k = np.ones((1, 5))
    g_pi = np.ones(2)

    # Valid run
    sol = solve_block_arrow_schur(h_kk, h_kpi, h_pipi, g_k, g_pi)
    assert sol.full.shape == (7,)

    # Mismatched H_kk vs H_kpi
    with pytest.raises(PreconditionError):
        solve_block_arrow_schur(np.eye(4)[None, ...], h_kpi, h_pipi, g_k, g_pi)

    # Non-square H_pipi
    with pytest.raises(PreconditionError):
        solve_block_arrow_schur(h_kk, h_kpi, np.ones((2, 3)), g_k, g_pi)

    # Non-finite values
    h_nan = h_kk.copy()
    h_nan[0, 0, 0] = np.nan
    with pytest.raises(PreconditionError):
        solve_block_arrow_schur(h_nan, h_kpi, h_pipi, g_k, g_pi)


def test_time_parallel_riccati_matches_sequential_recursion() -> None:
    """Time-parallel Riccati recursion matches sequential recursion over 0.75 s horizon."""
    chain = _make_chain(TRUE_LENGTHS)
    dt = 1.0 / 200.0
    n_nodes = 150  # 0.75 s horizon at 200 Hz
    times = np.arange(n_nodes) * dt
    u = np.zeros((1, n_nodes, 2))
    hanging = np.array([-np.pi / 2 + 0.4, 0.3, -0.2, 0.0, 0.0, 0.0])
    x0 = np.tile(hanging, (1, 1))
    q, v = chain.rollout(x0, u[:, :-1], TRUE_PI, dt)
    reference = np.concatenate([q[0], v[0]], axis=-1)

    def step(x: np.ndarray, u_in: np.ndarray) -> np.ndarray:
        q_next, v_next = chain.rollout(x, u_in[:, None, :], TRUE_PI, dt)
        return np.concatenate([q_next[:, 0], v_next[:, 0]], axis=-1)

    lin = linearize_along_trajectory(step, reference[:-1], u[0, :-1], 1e-6)
    q_weight = np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])
    r_weight = np.eye(2) * 0.01

    # Sequential Riccati
    k_seq = tvlqr_gains(lin, q_weight, r_weight)

    # Time-parallel Riccati
    k_par = time_parallel_riccati(lin, q_weight, r_weight)

    assert k_par.shape == k_seq.shape
    np.testing.assert_allclose(k_par, k_seq, atol=1e-8, rtol=1e-8)


def test_time_parallel_riccati_preconditions() -> None:
    """Preconditions on time-parallel Riccati inputs."""
    a = np.tile(np.eye(4), (10, 1, 1))
    b = np.tile(np.ones((4, 2)), (10, 1, 1))
    lin = LinearizedDynamics(a, b)
    q_weight = np.eye(4)
    r_weight = np.eye(2)

    # Valid
    gains = time_parallel_riccati(lin, q_weight, r_weight)
    assert gains.shape == (10, 2, 4)

    # Invalid Q shape
    with pytest.raises(PreconditionError):
        time_parallel_riccati(lin, np.eye(3), r_weight)

    # Invalid R shape
    with pytest.raises(PreconditionError):
        time_parallel_riccati(lin, q_weight, np.eye(3))


def test_generate_runtime_receipt() -> None:
    """Runtime receipt contains hardware, threads, library versions, timings, and scaling."""
    trials_map = {1: 0.05, 2: 0.09, 8: 0.32, 32: 1.25}
    receipt = generate_runtime_receipt(
        wall_time_per_iteration_s=0.045,
        trial_scaling=trials_map,
    )

    assert isinstance(receipt, RuntimeReceipt)
    assert receipt.wall_time_per_iteration_s == 0.045
    assert receipt.trial_scaling == trials_map
    assert "numpy" in receipt.library_versions
    assert "os" in receipt.hardware
    assert receipt.threads

    # Verify JSON serializability
    receipt_dict = receipt.to_dict()
    serialized = json.dumps(receipt_dict)
    assert len(serialized) > 0
    deserialized = json.loads(serialized)
    assert deserialized["wall_time_per_iteration_s"] == 0.045

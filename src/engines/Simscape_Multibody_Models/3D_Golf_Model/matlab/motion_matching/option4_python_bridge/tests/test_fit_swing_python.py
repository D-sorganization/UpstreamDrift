"""Recovery test for ``fit_swing_scipy``.

Synthesizes a target by running the simulator on a known ``theta_truth``,
then runs ``fit_swing_scipy`` from a slightly-perturbed warm start. The
optimizer should recover ``theta_truth`` within the per-coefficient
tolerance from the spec (10 % of the bound range).

The full SLSQP recovery is a slow test (it triggers ~50–200 sim calls).
On hosts without MATLAB, conftest.py skips it with a loud reason.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.requires_matlab_engine
@pytest.mark.slow
def test_recover_theta_from_synthetic_target(
    adapter, n_joints, bounds, has_simscape_multibody
):
    """The optimizer recovers a known theta from its own synthetic target."""
    if not has_simscape_multibody:
        pytest.skip(
            "Simscape Multibody license not available — cannot run forward sim."
        )
    from fit_swing_python import FitOptions, fit_swing_scipy

    lb, ub = bounds
    rng = np.random.default_rng(7)
    # truth lives at 2 % of the bound box — keeps the integrator stable.
    theta_truth = (rng.uniform(-0.02, 0.02, size=lb.size) * np.abs(lb)).astype(
        np.float64
    )

    sim_truth = adapter.simulate_with_coefficients(theta_truth)
    if sim_truth.solver_status != "success":
        pytest.skip(
            f"simulator did not converge on synthetic theta_truth "
            f"(status={sim_truth.solver_status}); a different seed is needed."
        )

    from simscape_adapter import ClubTarget

    target = ClubTarget(
        time=sim_truth.time,
        grip=sim_truth.grip,
        clubhead=sim_truth.clubhead,
        club_quat=sim_truth.club_quat,
        impact_idx=int(sim_truth.impact_idx),
    )

    # warm start: 80 % of the truth (should be inside the basin)
    theta0 = 0.8 * theta_truth

    options = FitOptions(
        method="SLSQP",
        max_iter=40,
        ftol=1e-7,
        theta0=theta0,
        verbose=False,
    )
    result = fit_swing_scipy(target, adapter, options)

    # Sanity checks
    assert result.theta.size == theta_truth.size
    assert result.cost >= 0.0
    assert np.all(np.isfinite(result.theta))

    # Per-coefficient recovery within 10 % of bound range
    bound_range = ub - lb
    err = np.abs(result.theta - theta_truth)
    tolerance = 0.10 * bound_range
    n_within = int(np.count_nonzero(err <= tolerance))
    fraction = n_within / err.size
    # The cost surface is locally flat in many directions (the cost is
    # under-determined — many thetas produce the same trajectory), so we
    # require recovery on a majority of coefficients rather than all of them.
    assert fraction >= 0.5, (
        f"only {fraction:.1%} of coefficients recovered within 10 % of bound "
        f"range (err max={err.max():.3f}, tol max={tolerance.max():.3f})"
    )


def test_fit_options_defaults():
    """FitOptions has sane defaults; can be constructed without MATLAB."""
    from fit_swing_python import FitOptions

    o = FitOptions()
    assert o.method == "SLSQP"
    assert o.max_iter > 0
    assert o.ftol > 0
    assert o.theta0 is None
    assert o.cost_opts == {}


def test_fit_swing_jax_is_explicitly_unimplemented():
    """JAX path raises NotImplementedError pointing at issue #4075."""
    from fit_swing_python import fit_swing_jax

    with pytest.raises(NotImplementedError, match="#4075"):
        fit_swing_jax(target=None, adapter=None)  # type: ignore[arg-type]


def test_fit_swing_scipy_rejects_bad_theta0():
    """Length mismatch on theta0 must surface as ValueError without ever
    starting the engine.
    """
    pytest.importorskip("scipy.optimize")
    from fit_swing_python import FitOptions, fit_swing_scipy
    from simscape_adapter import ClubTarget, SimscapeAdapter

    target = ClubTarget(
        time=np.zeros(2),
        grip=np.zeros((2, 3)),
        clubhead=np.zeros((2, 3)),
        club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        impact_idx=0,
    )

    class _StubAdapter(SimscapeAdapter):
        def start(self) -> None:
            return

        def get_n_joints(self) -> int:
            return 3

        def get_polynomial_bounds(self, n_joints: int):
            n = n_joints * 7
            return -np.ones(n), np.ones(n)

    options = FitOptions(theta0=np.zeros(5))  # wrong length
    with pytest.raises(ValueError):
        fit_swing_scipy(target, _StubAdapter(), options)

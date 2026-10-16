"""Held-out paired Monte Carlo verification of noise-aware optima (NSX-92).

The campaign forward model is the NONLINEAR reduced-coordinate pendulum step
(``_variance_fixtures.pendulum_nonlinear_step``), independent of the kernels
that produced the prediction.  Declared noise is a test hypothesis, not a
measurement.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.noise_aware_allocation import (  # noqa: E402
    minimise_face_variance,
)
from src.shared.python.loop_nullspace.noise_aware_spec import (  # noqa: E402
    NoiseAwareStatus,
)
from src.shared.python.loop_nullspace.noise_aware_verification import (  # noqa: E402
    OPTIMUM,
    REFERENCE,
    verify_against_monte_carlo,
)
from src.shared.python.loop_nullspace.nonlinearity_gate import (  # noqa: E402
    NonlinearityReport,
)
from src.shared.python.loop_nullspace.paired_mc_spec import (  # noqa: E402
    IntegratorSignature,
)
from src.shared.python.loop_nullspace.paired_statistics import (  # noqa: E402
    CrossCheckStatus,
)
from tests.unit.loop_nullspace import _noise_aware_fixtures as fx  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.slow]

INTEGRATOR = IntegratorSignature("explicit Euler, reduced-coordinate pendulum", fx.DT)
N_TRIALS = 3000
N_BOOT = 200


def _gate(exceeded: bool) -> NonlinearityReport:
    return NonlinearityReport(
        ratios=np.zeros((1, 1)),
        max_ratio=np.zeros(1),
        exceeded=(exceeded,),
        gate_threshold=0.1,
        amplitude=1.0,
        n_failed=0,
    )


@pytest.fixture(scope="module")
def optimum():
    sp = fx.strong_spec()
    res = minimise_face_variance(
        [fx.frame()],
        fx.times(),
        fx.SCALE,
        fx.m2_model(sp),
        fx.declaration(True, True, sp),
    )
    assert res.status is NoiseAwareStatus.LOCAL_STATIONARY
    return sp, res


def _verify(res, sp, **kw):
    return verify_against_monte_carlo(
        res,
        fx.mc_campaign_spec(sp, N_TRIALS, N_TRIALS),
        fx.mc_runner_factory(sp),
        INTEGRATOR,
        outcome="y",
        n_boot=N_BOOT,
        **kw,
    )


@pytest.fixture(scope="module")
def report(optimum):
    sp, res = optimum
    return _verify(res, sp)


def test_optimum_lowers_the_nonlinear_mc_sd_on_paired_and_held_out_seeds(
    optimum, report
) -> None:
    _, res = optimum
    assert report.confirmed and report.holdout_agrees
    for comp in (report.paired, report.holdout):
        diff = comp.outcomes["y"]
        assert diff.sd_diff < 0.0 and diff.sd_diff_ci[1] < 0.0
        assert diff.n_pairs == N_TRIALS
    # the predicted ranking and the size of the benefit (conditional on the model)
    pred = report.predicted_sd
    assert pred[OPTIMUM] < pred[REFERENCE]
    hold = report.holdout.outcomes["y"]
    assert hold.sd_a < hold.sd_b
    assert hold.sd_diff == pytest.approx(pred[OPTIMUM] - pred[REFERENCE], abs=0.05)
    assert res.variance < res.reference_variance


def test_linear_gap_is_reported_and_agreement_needs_the_gate(optimum) -> None:
    sp, res = optimum
    plain = _verify(res, sp)
    for name in (OPTIMUM, REFERENCE):
        chk = plain.cross_checks[name][0]
        assert np.isfinite(chk.gap) and np.isfinite(chk.relative_gap)
        assert chk.status is CrossCheckStatus.NOT_ASSESSED  # no gate -> no claim (V7)
        assert abs(chk.relative_gap) < 0.1  # small-noise regime of this fixture
    passed = _verify(res, sp, gate=_gate(False))
    failed = _verify(res, sp, gate=_gate(True))
    for name in (OPTIMUM, REFERENCE):
        assert passed.cross_checks[name][0].status in (
            CrossCheckStatus.AGREES,
            CrossCheckStatus.DISAGREES,
        )
        assert failed.cross_checks[name][0].status is CrossCheckStatus.NOT_LINEAR_VALID
        assert failed.cross_checks[name][0].gap == passed.cross_checks[name][0].gap


def test_flat_negative_control_is_never_confirmed() -> None:
    sp = fx.spec(None, (0.3, 0.3), golden_units=False)
    res = minimise_face_variance(
        [fx.frame()],
        fx.times(),
        fx.SCALE,
        fx.m1_model(sp),
        fx.declaration(False, False, sp),
    )
    assert res.status is NoiseAwareStatus.OBJECTIVE_FLAT_INVARIANT
    rep = verify_against_monte_carlo(
        res,
        fx.mc_campaign_spec(sp, 400, 400),
        fx.mc_runner_factory(sp),
        INTEGRATOR,
        outcome="y",
        n_boot=N_BOOT,
    )
    assert not rep.confirmed
    # same allocation on both arms and the same draws: identical outcomes
    assert rep.paired.outcomes["y"].sd_diff == 0.0


def test_a_worse_allocation_is_not_confirmed(optimum) -> None:
    """Hold-out honesty: swap in a high-variance allocation as the 'optimum'."""
    sp, res = optimum
    z_bad = np.array([20.0])
    tau_bad = (fx.U_P - fx.Q * z_bad[0])[None]
    worse = dataclasses.replace(res, torque=tau_bad, z=(z_bad,))
    model = fx.m2_model(sp)
    worse = dataclasses.replace(worse, variance=model.variance(z_bad, tau_bad))
    rep = verify_against_monte_carlo(
        worse,
        fx.mc_campaign_spec(sp, 1000, 1000),
        fx.mc_runner_factory(sp),
        INTEGRATOR,
        outcome="y",
        n_boot=N_BOOT,
    )
    assert not rep.confirmed
    assert rep.paired.outcomes["y"].sd_diff > 0.0  # the MC sees it is worse


def test_preconditions(optimum) -> None:
    sp, res = optimum
    spec = fx.mc_campaign_spec(sp, 10, 10)
    factory = fx.mc_runner_factory(sp)
    with pytest.raises(TypeError):
        verify_against_monte_carlo("x", spec, factory, INTEGRATOR, outcome="y")
    with pytest.raises(TypeError):
        verify_against_monte_carlo(res, spec, "nope", INTEGRATOR, outcome="y")
    with pytest.raises(ValueError, match="unknown outcome"):
        verify_against_monte_carlo(res, spec, factory, INTEGRATOR, outcome="z")
    infeasible = dataclasses.replace(res, status=NoiseAwareStatus.INFEASIBLE)
    with pytest.raises(ValueError, match="nothing to verify"):
        verify_against_monte_carlo(infeasible, spec, factory, INTEGRATOR, outcome="y")

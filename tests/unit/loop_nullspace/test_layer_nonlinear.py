"""NSX-93 (#12722): nonlinear Monte Carlo against the linear layer variance.

Plant: a hanging pendulum ``th'' = -(g/L) sin th + u / (m L^2)`` about its rest
equilibrium, ``x = (th, om)``.  The linear model is its tangent
(``F = [[0, 1], [-g/L, 0]]``); the Monte Carlo integrates the *nonlinear*
equations with RK4 sub-steps, the impedance acting continuously, the noise and
the delayed feedback held over a step.  At deviations of order 1e-3 rad the
``sin`` nonlinearity is ~1e-6 relative, so the gap is purely statistical: the
tolerance is the sampling half-width of an SD estimate,
``Z99 / sqrt(2 (N - 1))`` (NSX-91).  Deterministic seeds make every assertion
either always hold or always fail.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from src.shared.python.loop_nullspace import paired_monte_carlo as pmc
from src.shared.python.loop_nullspace import paired_statistics as ps
from src.shared.python.loop_nullspace.evidence import (
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.impedance_spec import (
    ImpedanceWindow,
    JointImpedance,
)
from src.shared.python.loop_nullspace.layer_linear import (
    LayerLinearization,
    impedance_state_gain,
    layer_variance,
)
from src.shared.python.loop_nullspace.minimum_intervention import (
    minimum_intervention_design,
)
from src.shared.python.loop_nullspace.paired_mc_spec import (
    CampaignSpec,
    IntegratorSignature,
    OutcomeChannel,
    OutcomeKind,
)
from src.shared.python.loop_nullspace.replay_spec import ReplayMode
from src.shared.python.loop_nullspace.variance_propagation import propagate_covariance
from src.shared.python.loop_nullspace.variance_records import VarianceDeclaration
from src.shared.python.perturbation.disturbance import (
    DisturbanceSpec,
    InjectionStage,
    Spectrum,
)

pytestmark = pytest.mark.unit

Z99 = 2.5758293035489004
MASS, LEN, GRAV = 1.0, 1.0, 10.0
DT, N_STEPS, SUB = 0.01, 100, 2
KP, KD = 30.0, 3.0
DELAY = 2
Q_STEP = 4e-6  # variance of the held torque noise [N^2 m^2]
P0 = np.diag([1e-8, 1e-6])
ROW = np.array([[1.0, 0.0]])
N_TRIALS = 800
RK4 = IntegratorSignature("RK4", DT / SUB)

F = np.array([[0.0, 1.0], [-GRAV / LEN, 0.0]])
G = np.array([[0.0], [1.0 / (MASS * LEN**2)]])


def _step(
    th: float, om: float, u_held: float, kp: float, kd: float
) -> tuple[float, float]:
    """RK4 sub-steps; the impedance ``-(kp th + kd om)`` is evaluated per stage.

    Plain floats: the campaign calls this ~10^7 times.
    """
    h = DT / SUB
    inv = 1.0 / (MASS * LEN**2)
    g_l = GRAV / LEN

    def acc(y_th: float, y_om: float) -> float:
        return -g_l * math.sin(y_th) + (u_held - kp * y_th - kd * y_om) * inv

    for _ in range(SUB):
        k1t, k1o = om, acc(th, om)
        k2t, k2o = om + 0.5 * h * k1o, acc(th + 0.5 * h * k1t, om + 0.5 * h * k1o)
        k3t, k3o = om + 0.5 * h * k2o, acc(th + 0.5 * h * k2t, om + 0.5 * h * k2o)
        k4t, k4o = om + h * k3o, acc(th + h * k3t, om + h * k3o)
        th += h / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        om += h / 6.0 * (k1o + 2 * k2o + 2 * k3o + k4o)
    return th, om


def _runner(kp: float, kd: float, k_fb: np.ndarray | None):
    def run(draw: pmc.TrialDraw) -> np.ndarray:
        w = draw.realise("force")[:, 0]
        th, om = (float(c) for c in draw.realise("x0"))
        states = [(th, om)]
        for t in range(N_STEPS):
            u = float(w[t])
            if k_fb is not None:
                d_th, d_om = states[max(t - DELAY, 0)]
                u -= float(k_fb[t, 0, 0]) * d_th + float(k_fb[t, 0, 1]) * d_om
            th, om = _step(th, om, u, kp, kd)
            states.append((th, om))
        return np.array([th])

    return run


def _spec() -> CampaignSpec:
    return CampaignSpec(
        disturbances=(
            DisturbanceSpec(
                name="force",
                stage=InjectionStage.TORQUE,
                channels=("tau",),
                units="N m",
                covariance=np.array([[Q_STEP * DT]]),
                spectrum=Spectrum.WHITE,
            ),
            DisturbanceSpec(
                name="x0",
                stage=InjectionStage.INITIAL_STATE,
                channels=("th", "om"),
                units="rad|rad/s",
                covariance=P0,
                persistent=True,
            ),
        ),
        n_steps=N_STEPS,
        dt=DT,
        n_trials=N_TRIALS,
        n_holdout=N_TRIALS,
        paired_seed=31,
        holdout_seed=32,
        outcomes=(OutcomeChannel("th_end", "rad", OutcomeKind.SCALAR),),
    )


def _lin(impedance: bool, k_fb: np.ndarray | None) -> LayerLinearization:
    rep = lambda a: np.repeat(a[None], N_STEPS, axis=0)  # noqa: E731
    imp_gain = None
    if impedance:
        win = ImpedanceWindow(0.0, N_STEPS * DT, np.array([KP]), np.array([KD]))
        imp_gain = impedance_state_gain(
            JointImpedance((win,)),
            np.arange(N_STEPS) * DT,
            np.eye(1),
            np.eye(1),
            np.zeros((1, 1)),
        )
    return LayerLinearization(
        rep(F),
        rep(G),
        rep(G),
        DT,
        impedance_gain=imp_gain,
        feedback_gain=k_fb,
        delay_steps=DELAY if k_fb is not None else 0,
    )


def _predicted(lin: LayerLinearization) -> np.ndarray:
    sigma = np.full((N_STEPS, 1, 1), Q_STEP)
    traj = propagate_covariance(lin.closed_loop(), lin.augment_covariance(P0), sigma)
    rows = lin.augment_outcomes(ROW)
    return rows @ traj[-1] @ rows.T


@functools.cache
def _mc_cached(kp: float, kd: float, with_fb: bool):
    k_fb = _design() if with_fb else None
    spec = _spec()
    result = pmc.run_campaign(
        spec, [pmc.Strategy("S", _runner(kp, kd, k_fb), ReplayMode.OPEN_LOOP, RK4)]
    )
    return spec, ps.summarise(
        spec, result.paired, "S", ci_level=0.99, n_boot=500, seed=5
    )


def _design() -> np.ndarray:
    plant = _lin(True, None).plant_dynamics()
    return minimum_intervention_design(plant, ROW, np.eye(1), np.array([[1e-2]])).gains


@pytest.mark.parametrize("case", ["open", "impedance", "impedance_feedback"])
def test_nonlinear_monte_carlo_recovers_the_layer_variance(case: str) -> None:
    """Open loop, impedance only, and impedance plus delayed feedback agree in SD."""
    k_fb = _design() if case == "impedance_feedback" else None
    use_imp = case != "open"
    kp, kd = (KP, KD) if use_imp else (0.0, 0.0)
    cov = _predicted(_lin(use_imp, k_fb))
    spec, summary = _mc_cached(kp, kd, k_fb is not None)
    check = ps.cross_check_linear(summary, spec, cov, gate=None)[0]
    half = Z99 / math.sqrt(2.0 * (N_TRIALS - 1))
    assert check.predicted_sd == pytest.approx(math.sqrt(cov[0, 0]))
    assert check.within_ci
    assert abs(check.relative_gap) < 1.15 * half


def test_the_layers_change_the_face_variance_in_the_predicted_order() -> None:
    """The ordering open > impedance > impedance + feedback is a model prediction.

    The nonlinear campaign reproduces it (the SDs differ by far more than the
    sampling half-width), so the benefit is not a linearisation artefact.
    """
    k_fb = _design()
    pred = [
        _predicted(_lin(False, None))[0, 0],
        _predicted(_lin(True, None))[0, 0],
        _predicted(_lin(True, k_fb))[0, 0],
    ]
    assert pred[0] > pred[1] > pred[2] > 0.0
    sds = [
        _mc_cached(0.0, 0.0, False)[1].dispersions["th_end"].sd,
        _mc_cached(KP, KD, False)[1].dispersions["th_end"].sd,
        _mc_cached(KP, KD, True)[1].dispersions["th_end"].sd,
    ]
    assert sds[0] > sds[1] > sds[2]


def test_zero_layer_is_the_open_loop_exactly() -> None:
    """Negative control: zero impedance and zero feedback gain change nothing."""
    zero_fb = np.zeros((N_STEPS, 1, 2))
    base = _predicted(_lin(False, None))[0, 0]
    zeroed = _predicted(_lin(False, zero_fb))[0, 0]
    assert zeroed == pytest.approx(base, rel=1e-13)


def test_typed_record_carries_the_layer_conditioning() -> None:
    k_fb = _design()
    decl = VarianceDeclaration(
        mechanisms=frozenset(),
        noise_hypotheses=("H-NOISE-ADDITIVE",),
        model_provenance=ModelProvenance("hanging_pendulum", "1"),
        input_provenance=InputProvenance("test", "declared"),
    )
    lv = layer_variance(
        _lin(True, k_fb),
        P0,
        np.full((N_STEPS, 1, 1), Q_STEP),
        ROW,
        decl,
        layer_hypotheses=("H-DECLARED-JOINT-IMPEDANCE", "H-SENSORIMOTOR-DELAY"),
        outcome_units="rad",
    )
    cond = lv.assessment.variance.conditioning
    assert "H-DECLARED-JOINT-IMPEDANCE" in cond and "H-SENSORIMOTOR-DELAY" in cond
    np.testing.assert_allclose(
        np.asarray(lv.assessment.variance.values).reshape(-1)[0],
        _predicted(_lin(True, k_fb))[0, 0],
        rtol=1e-10,
    )

"""NSX-91 (#12720): dispersion, paired comparison and the NSX-90 cross-check.

Tolerances come from the sampling distribution, never from a chosen number:
``SE(s) / s = 1 / sqrt(2 (N - 1))`` for ``N`` Gaussian draws, so a two-sided
99 % interval has half-width ``Z99 s / sqrt(2 (N - 1))``.  A deterministic seed
fixes every draw, so each assertion either always holds or always fails.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.local_policy import LinearizedDynamics
from src.shared.python.loop_nullspace import paired_monte_carlo as pmc
from src.shared.python.loop_nullspace import paired_statistics as ps
from src.shared.python.loop_nullspace import variance_propagation as vp
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.nonlinearity_gate import (
    LinearValidity,
    assess_nonlinearity,
    probe_directions,
)
from src.shared.python.loop_nullspace.paired_mc_spec import (
    CampaignSpec,
    IntegratorSignature,
    OutcomeChannel,
    OutcomeKind,
)
from src.shared.python.loop_nullspace.replay_spec import ReplayMode
from src.shared.python.perturbation.disturbance import (
    DisturbanceSpec,
    InjectionStage,
    Spectrum,
)
from src.shared.python.perturbation.outcome_dispersion import OutcomeDispersion
from tests.unit.loop_nullspace._tangent_fixtures import Pendulum
from tests.unit.loop_nullspace._variance_fixtures import (
    nominal_angle,
    pendulum_nonlinear_step,
)

pytestmark = pytest.mark.unit

Z99 = 2.5758293035489004
RK4 = IntegratorSignature("RK4", 0.0025)
EULER = IntegratorSignature("explicit Euler", 0.01)
N_BOOT = 2000


def _expected_sd_halfwidth(sd: float, n: int, z: float = Z99) -> float:
    return z * sd / math.sqrt(2.0 * (n - 1))


def _gauss_spec(n_trials: int, scale: float = 1.0) -> CampaignSpec:
    return CampaignSpec(
        disturbances=(
            DisturbanceSpec(
                name="w",
                stage=InjectionStage.TORQUE,
                channels=("c",),
                units="N m",
                covariance=np.array([[scale * 0.01]]),
                spectrum=Spectrum.WHITE,
            ),
        ),
        n_steps=1,
        dt=0.01,
        n_trials=n_trials,
        n_holdout=n_trials,
        paired_seed=3,
        holdout_seed=4,
        outcomes=(
            OutcomeChannel("face_angle", "deg", OutcomeKind.ANGLE, tolerance=1.0),
            OutcomeChannel("chs", "m/s", OutcomeKind.SCALAR),
        ),
    )


def _runner(face_gain: float, chs_gain: float = 1.0, centre: float = 0.0):
    def run(draw: pmc.TrialDraw) -> np.ndarray:
        w = float(draw.realise("w")[0, 0])
        return np.array([centre + face_gain * w, 45.0 + chs_gain * w])

    return run


def _campaign(n: int = 3000, gains=(1.0, 0.5), centre: float = 0.0):
    spec = _gauss_spec(n)
    strategies = [
        pmc.Strategy(
            f"S{i}", _runner(g, chs_gain=g, centre=centre), ReplayMode.OPEN_LOOP, RK4
        )
        for i, g in enumerate(gains)
    ]
    return spec, pmc.run_campaign(spec, strategies)


def _sd_of_draws(n: int) -> float:
    spec = _gauss_spec(n)
    return float(
        np.std([pmc.TrialDraw(spec, "paired", i).realise("w")[0, 0] for i in range(n)])
    )


# ----------------------------------------------------------------- dispersion
def test_zero_noise_gives_zero_standard_deviation() -> None:
    spec = _gauss_spec(50, scale=0.0)
    result = pmc.run_campaign(
        spec, [pmc.Strategy("A", _runner(3.0), ReplayMode.OPEN_LOOP, RK4)]
    )
    summary = ps.summarise(spec, result.paired, "A", n_boot=200, seed=1)
    face = summary.dispersions["face_angle"]
    chs = summary.dispersions["chs"]
    assert face.sd == 0.0 and face.sd_ci == (0.0, 0.0)
    assert chs.sd == 0.0 and chs.sd_ci == (0.0, 0.0) and chs.mean == 45.0
    assert summary.tally.failure_rate == 0.0


def test_face_angle_dispersion_is_the_wrapped_outcome_dispersion() -> None:
    """A face angle straddling +-180 deg is wrapped, not blown up (V3)."""
    spec, result = _campaign(400, gains=(1.0,), centre=180.0)
    summary = ps.summarise(spec, result.paired, "S0", n_boot=300, seed=5)
    face = summary.dispersions["face_angle"]
    assert isinstance(face, OutcomeDispersion)
    assert face.n_samples == 400 and face.unit == "deg"
    raw = result.paired.outcome_matrix("S0")[:, 0]
    assert raw.min() < 180.0 < raw.max()  # the sample really crosses the wrap
    # gain 1: the face SD equals the SD of the draw (S / dt = 1 deg^2)
    assert face.sd == pytest.approx(_sd_of_draws(400), rel=0.01)
    lo, hi = face.sd_ci
    assert lo <= face.sd <= hi


def test_scalar_dispersion_interval_matches_the_closed_form_half_width() -> None:
    n = 3000
    spec, result = _campaign(n, gains=(1.0,))
    summary = ps.summarise(
        spec, result.paired, "S0", ci_level=0.99, n_boot=N_BOOT, seed=2
    )
    chs = summary.dispersions["chs"]
    assert isinstance(chs, ps.ScalarDispersion)
    half = 0.5 * (chs.sd_ci[1] - chs.sd_ci[0])
    assert half == pytest.approx(_expected_sd_halfwidth(chs.sd, n), rel=0.15)
    assert chs.sd_ci[0] < chs.sd < chs.sd_ci[1]
    mean_half = 0.5 * (chs.mean_ci[1] - chs.mean_ci[0])
    assert mean_half == pytest.approx(Z99 * chs.sd / math.sqrt(n), rel=0.15)


def test_summary_is_deterministic_for_a_seed_and_counts_failures() -> None:
    spec = _gauss_spec(60)

    def flaky(draw: pmc.TrialDraw) -> np.ndarray:
        if draw.index % 4 == 0:
            raise pmc.TrialMiss("no impact")
        return _runner(1.0)(draw)

    result = pmc.run_campaign(
        spec, [pmc.Strategy("A", flaky, ReplayMode.OPEN_LOOP, RK4)]
    )
    one = ps.summarise(spec, result.paired, "A", n_boot=100, seed=9)
    two = ps.summarise(spec, result.paired, "A", n_boot=100, seed=9)
    assert one.dispersions["chs"] == two.dispersions["chs"]
    assert one.tally.counts[pmc.TrialStatus.MISSED] == 15
    assert one.dispersions["chs"].n_samples == 45  # type: ignore[union-attr]
    assert one.evidence_label == "independent_open_loop_replay"


def test_summary_without_any_ok_trial_reports_no_dispersion_not_zero() -> None:
    spec = _gauss_spec(10)

    def always(draw: pmc.TrialDraw) -> np.ndarray:
        raise pmc.TrialMiss("no impact")

    result = pmc.run_campaign(
        spec, [pmc.Strategy("A", always, ReplayMode.OPEN_LOOP, RK4)]
    )
    summary = ps.summarise(spec, result.paired, "A", n_boot=50, seed=1)
    assert summary.dispersions == {"face_angle": None, "chs": None}
    assert summary.tally.failure_rate == 1.0
    with pytest.raises(ValueError, match="no successful"):
        ps.sd_records(summary, spec, _provenance())


# ------------------------------------------------------------------ comparison
def test_paired_comparison_uses_the_common_draws() -> None:
    n = 2000
    spec, result = _campaign(n, gains=(1.0, 0.5))
    cmp_ = ps.compare_strategies(
        spec, result.paired, "S0", "S1", ci_level=0.99, n_boot=1000, seed=3
    )
    out = cmp_.outcomes["chs"]
    assert cmp_.n_pairs == n and cmp_.seed_set == "paired"
    sd = _sd_of_draws(n)
    assert out.sd_a == pytest.approx(sd, rel=0.05)
    assert out.sd_b == pytest.approx(0.5 * sd, rel=0.05)
    lo, hi = out.sd_diff_ci
    assert lo > 0.0  # S0 is truly more dispersed than S1
    # y_a - y_b = 0.5 w exactly, so the paired mean-difference interval is
    # 0.5 * Z99 sd / sqrt(n); an unpaired interval would be sqrt(1.25) / 0.5 wider
    half = 0.5 * (out.mean_diff_ci[1] - out.mean_diff_ci[0])
    assert half == pytest.approx(0.5 * Z99 * sd / math.sqrt(n), rel=0.15)
    unpaired = Z99 * math.sqrt(out.sd_a**2 + out.sd_b**2) / math.sqrt(n)
    assert half < 0.5 * unpaired


def test_strategies_with_different_failures_keep_every_trial_in_the_tally() -> None:
    spec = _gauss_spec(40)

    def drops_some(draw: pmc.TrialDraw) -> np.ndarray:
        if draw.index < 10:
            raise ValueError("slip")
        return _runner(1.0)(draw)

    strategies = [
        pmc.Strategy("A", _runner(1.0), ReplayMode.OPEN_LOOP, RK4),
        pmc.Strategy("B", drops_some, ReplayMode.OPEN_LOOP, RK4),
    ]
    result = pmc.run_campaign(spec, strategies)
    cmp_ = ps.compare_strategies(spec, result.paired, "A", "B", n_boot=100, seed=1)
    assert cmp_.n_pairs == 30
    assert cmp_.failed_only_b == 10 and cmp_.failed_only_a == 0
    assert cmp_.tally_b.failure_rate == pytest.approx(0.25)
    assert cmp_.tally_a.failure_rate == 0.0


def test_open_loop_and_feedback_are_not_compared_unless_labelled() -> None:
    spec = _gauss_spec(30)
    strategies = [
        pmc.Strategy("ol", _runner(1.0), ReplayMode.OPEN_LOOP, RK4),
        pmc.Strategy("fb", _runner(1.0), ReplayMode.FEEDBACK, RK4),
    ]
    result = pmc.run_campaign(spec, strategies)
    with pytest.raises(ValueError, match="open-loop.*feedback"):
        ps.compare_strategies(spec, result.paired, "ol", "fb", n_boot=50, seed=1)
    cmp_ = ps.compare_strategies(
        spec, result.paired, "ol", "fb", n_boot=50, seed=1, allow_cross_mode=True
    )
    assert cmp_.cross_mode is True


def test_holdout_confirms_a_reproducible_ranking_and_flags_a_reversal() -> None:
    spec, result = _campaign(1500, gains=(1.0, 0.5))
    kw = {"n_boot": 500, "seed": 4, "ci_level": 0.99}
    paired = ps.compare_strategies(spec, result.paired, "S0", "S1", **kw)
    hold = ps.compare_strategies(spec, result.holdout, "S0", "S1", **kw)
    assert ps.holdout_consistent(paired, hold)["chs"] is True
    swapped = ps.compare_strategies(spec, result.holdout, "S1", "S0", **kw)
    assert ps.holdout_consistent(paired, swapped)["chs"] is False


# -------------------------------------------------------- NSX-90 cross-check
DT90 = 0.1
N90 = 3
U_FF = Pendulum.u_particular.copy()
X_NOM = np.array([nominal_angle(), Pendulum.omega])
C_ROW = np.array([[0.4, 1.0]])


def _jacobians() -> LinearizedDynamics:
    a_all, b_all, x = [], [], X_NOM.copy()
    h = 1e-6
    for _ in range(N90):
        a, b = np.zeros((2, 2)), np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            a[:, j] = (
                pendulum_nonlinear_step(x + e, np.zeros(2), U_FF, DT90)
                - pendulum_nonlinear_step(x - e, np.zeros(2), U_FF, DT90)
            ) / (2 * h)
            b[:, j] = (
                pendulum_nonlinear_step(x, e, U_FF, DT90)
                - pendulum_nonlinear_step(x, -e, U_FF, DT90)
            ) / (2 * h)
        a_all.append(a)
        b_all.append(b)
        x = pendulum_nonlinear_step(x, np.zeros(2), U_FF, DT90)
    return LinearizedDynamics(np.array(a_all), np.array(b_all))


def _pendulum_spec(amplitude: float, n_trials: int) -> CampaignSpec:
    """Noise covariance scales with ``amplitude**2``; Sigma_t = S / dt."""
    process = DisturbanceSpec(
        name="force",
        stage=InjectionStage.TORQUE,
        channels=("f1", "f2"),
        units="N",
        covariance=np.diag([0.04, 0.09]) * DT90 * amplitude**2,
        spectrum=Spectrum.WHITE,
    )
    offset = DisturbanceSpec(
        name="x0",
        stage=InjectionStage.INITIAL_STATE,
        channels=("th", "om"),
        units="rad|rad/s",
        covariance=np.diag([0.01, 0.04]) * amplitude**2,
        persistent=True,
    )
    return CampaignSpec(
        disturbances=(process, offset),
        n_steps=N90,
        dt=DT90,
        n_trials=n_trials,
        n_holdout=n_trials,
        paired_seed=21,
        holdout_seed=22,
        outcomes=(OutcomeChannel("y", "rad", OutcomeKind.SCALAR),),
    )


def _pendulum_runner(draw: pmc.TrialDraw) -> np.ndarray:
    w, dx0 = draw.realise("force"), draw.realise("x0")
    x, ref = X_NOM + dx0, X_NOM.copy()
    for t in range(N90):
        x = pendulum_nonlinear_step(x, w[t], U_FF, DT90)
        ref = pendulum_nonlinear_step(ref, np.zeros(2), U_FF, DT90)
    return C_ROW @ (x - ref)


def _linear_prediction(amplitude: float):
    dyn = _jacobians()
    sigma = np.repeat((np.diag([0.04, 0.09]) * amplitude**2)[None], N90, axis=0)
    p0 = np.diag([0.01, 0.04]) * amplitude**2
    kernels = vp.adjoint_kernels(dyn, C_ROW)
    return kernels, p0, sigma, vp.adjoint_covariance(kernels, p0, sigma)


def test_small_amplitude_monte_carlo_recovers_the_nsx90_covariance_within_ci() -> None:
    n = 4000
    amp = 1e-3
    spec = _pendulum_spec(amp, n)
    result = pmc.run_campaign(
        spec, [pmc.Strategy("A", _pendulum_runner, ReplayMode.OPEN_LOOP, RK4)]
    )
    summary = ps.summarise(
        spec, result.paired, "A", ci_level=0.99, n_boot=N_BOOT, seed=7
    )
    _, _, _, cov = _linear_prediction(amp)
    check = ps.cross_check_linear(summary, spec, cov, gate=None)[0]
    assert check.predicted_sd == pytest.approx(math.sqrt(cov[0, 0]))
    assert check.within_ci  # the prediction lies in the 99 % interval
    # nonlinear bias is O(amplitude) ~ 1e-3 relative, far below the statistical
    # half-width (>= Z99 / sqrt(2 (N - 1)) ~ 2.9 %), so the gap is statistical
    assert abs(check.relative_gap) < _expected_sd_halfwidth(1.0, n) * 1.15
    assert check.status is ps.CrossCheckStatus.NOT_ASSESSED  # no gate was run


def test_gate_decides_whether_agreement_is_claimed() -> None:
    n = 3000
    amp = 1e-3
    spec = _pendulum_spec(amp, n)
    result = pmc.run_campaign(
        spec, [pmc.Strategy("A", _pendulum_runner, ReplayMode.OPEN_LOOP, RK4)]
    )
    summary = ps.summarise(
        spec, result.paired, "A", ci_level=0.99, n_boot=N_BOOT, seed=7
    )
    kernels, p0, sigma, cov = _linear_prediction(amp)
    dirs = probe_directions(kernels, p0, sigma, include_axes=False)

    def rollout(theta: np.ndarray) -> np.ndarray:
        w = theta[2:].reshape(N90, 2)
        x, ref = X_NOM + theta[:2], X_NOM.copy()
        for t in range(N90):
            x = pendulum_nonlinear_step(x, w[t], U_FF, DT90)
            ref = pendulum_nonlinear_step(ref, np.zeros(2), U_FF, DT90)
        return C_ROW @ (x - ref)

    gate = assess_nonlinearity(rollout, dirs, amplitude=1.0, gate_threshold=1e-3)
    assert gate.validity is LinearValidity.LINEAR_VALID
    ok = ps.cross_check_linear(summary, spec, cov, gate=gate)[0]
    assert ok.status is ps.CrossCheckStatus.AGREES
    failing = assess_nonlinearity(rollout, dirs, amplitude=1.0, gate_threshold=1e-12)
    assert failing.validity is LinearValidity.NOT_LINEAR_VALID
    flagged = ps.cross_check_linear(summary, spec, cov, gate=failing)[0]
    assert flagged.status is ps.CrossCheckStatus.NOT_LINEAR_VALID
    assert flagged.mc_sd == ok.mc_sd  # the gap is still reported, never hidden


def test_finite_amplitude_gap_is_reported_as_disagreement() -> None:
    n = 3000
    spec = _pendulum_spec(10.0, n)  # amplitude 10: sin(theta) saturates
    result = pmc.run_campaign(
        spec, [pmc.Strategy("A", _pendulum_runner, ReplayMode.OPEN_LOOP, RK4)]
    )
    summary = ps.summarise(spec, result.paired, "A", ci_level=0.99, n_boot=500, seed=7)
    _, _, _, cov = _linear_prediction(10.0)
    check = ps.cross_check_linear(
        summary, spec, cov, gate=_passing_gate(len(spec.outcomes))
    )[0]
    assert check.status is ps.CrossCheckStatus.DISAGREES
    half = 0.5 * (check.mc_sd_ci[1] - check.mc_sd_ci[0])
    assert not check.within_ci and abs(check.gap) > 2.0 * half  # beyond sampling
    assert check.failure_rate == summary.tally.failure_rate


def _passing_gate(ny: int):
    from src.shared.python.loop_nullspace.nonlinearity_gate import NonlinearityReport

    return NonlinearityReport(
        ratios=np.zeros((1, ny)),
        max_ratio=np.zeros(ny),
        exceeded=(False,) * ny,
        gate_threshold=1.0,
        amplitude=1.0,
        n_failed=0,
    )


def test_cross_check_validates_shapes() -> None:
    spec = _pendulum_spec(1e-3, 20)
    result = pmc.run_campaign(
        spec, [pmc.Strategy("A", _pendulum_runner, ReplayMode.OPEN_LOOP, RK4)]
    )
    summary = ps.summarise(spec, result.paired, "A", n_boot=50, seed=1)
    with pytest.raises(ValueError, match="linear_covariance"):
        ps.cross_check_linear(summary, spec, np.eye(3), gate=None)


# --------------------------------------------------------------------- records
def _provenance():
    return ps.RecordProvenance(
        ModelProvenance("pendulum", "1"),
        InputProvenance("fixture", "nsx91 test"),
        conditioning=("H-NOISE-ADDITIVE",),
    )


def test_sd_records_carry_evidence_axes_and_the_mode_label() -> None:
    spec, result = _campaign(200, gains=(1.0,))
    summary = ps.summarise(spec, result.paired, "S0", n_boot=100, seed=1)
    recs = ps.sd_records(summary, spec, _provenance())
    assert set(recs) == {"face_angle", "chs", "failure_rate"}
    chs = recs["chs"]
    assert chs.units == "m/s" and chs.values.shape == (1,)
    assert chs.allocation_dependence is AllocationDependence.INVARIANT
    assert "H-NOISE-ADDITIVE" in chs.conditioning
    assert any("independent_open_loop_replay" in c for c in chs.conventions)
    assert recs["failure_rate"].units == "1"
    hyp = ps.sd_records(
        summary, spec, _provenance(), hypotheses=("H-TORQUE-CAPACITY",)
    )["chs"]
    assert hyp.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT

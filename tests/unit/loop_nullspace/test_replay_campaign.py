"""NSX-91 (#12720): the campaign runner over NSX-31 replay (R10, R13).

The torque noise is a zero-order-hold white disturbance with covariance
``S / dt`` per allocation interval, the same discretisation the NSX-90 linear
prediction uses (V6).  Replays use the fixed-step RK4 of NSX-31; the declared
generator is the DOP853 reference of ``test_replay`` (a different integrator
and an adaptive step), so the R13 gate passes honestly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.feedback_comparison import ReplayPolicy
from src.shared.python.loop_nullspace import paired_monte_carlo as pmc
from src.shared.python.loop_nullspace import paired_statistics as ps
from src.shared.python.loop_nullspace.paired_mc_spec import (
    CampaignSpec,
    GeneratorSpec,
    IntegratorSignature,
    OutcomeChannel,
    OutcomeKind,
)
from src.shared.python.loop_nullspace.planar_model import PlanarGolferProvider
from src.shared.python.loop_nullspace.replay import (
    ReferenceTrajectory,
    ReplayMode,
    ReplayReport,
    ReplaySettings,
    ReplaySetup,
)
from src.shared.python.loop_nullspace.replay_campaign import (
    ReplayCase,
    TorqueNoise,
    replay_integrator,
    replay_strategy,
)
from src.shared.python.perturbation.disturbance import (
    DisturbanceSpec,
    InjectionStage,
    Spectrum,
)
from tests.unit.loop_nullspace.test_replay import (
    LAW,
    N,
    S,
    T_END,
    _exact_allocation,
    _params,
    _reference,
    _state0,
    _tau,
)

pytestmark = pytest.mark.unit

FRAMES = 16
DT = T_END / (FRAMES - 1)
GENERATOR = IntegratorSignature("DOP853 (scipy solve_ivp, rtol 1e-13)", None)


def _torque_noise(sd: float, name: str = "joint_torque") -> DisturbanceSpec:
    return DisturbanceSpec(
        name=name,
        stage=InjectionStage.TORQUE,
        channels=tuple(f"tau{i}" for i in range(N)),
        units="N m",
        covariance=np.eye(N) * sd**2 * DT,  # per-step variance = S / dt = sd^2
        spectrum=Spectrum.WHITE,
    )


def _spec(sd: float, n_trials: int = 6, generator: GeneratorSpec | None = None):
    return CampaignSpec(
        disturbances=(_torque_noise(sd),),
        n_steps=FRAMES - 1,
        dt=DT,
        n_trials=n_trials,
        n_holdout=n_trials,
        paired_seed=5,
        holdout_seed=6,
        outcomes=(
            OutcomeChannel("q0_final", "rad", OutcomeKind.ANGLE, tolerance=0.01),
            OutcomeChannel("v1_final", "rad/s", OutcomeKind.SCALAR),
        ),
        generator=generator,
    )


def _outcomes(report: ReplayReport) -> np.ndarray:
    return np.array([report.q[-1][0], report.v[-1][1]])


def _case(policy: ReplayPolicy, substeps: int = 2) -> ReplayCase:
    q0, v0 = _state0()
    reference = None
    feedback = None
    if policy is ReplayPolicy.STATE_FEEDBACK:
        sol = _reference(_tau)
        times = np.linspace(0.0, T_END, 61)
        states = sol.sol(times)
        reference = ReferenceTrajectory(times, states[:N].T, states[N:].T)
        feedback = LAW
    return ReplayCase(
        allocation=_exact_allocation(FRAMES),
        provider=PlanarGolferProvider(_params()),
        setup=ReplaySetup(actuation=S, q0=q0, v0=v0, reference=reference),
        policy=policy,
        feedback=feedback,
        settings=ReplaySettings(substeps=substeps, divergence_tolerance=1.0),
    )


def _strategies(spec: CampaignSpec, outcome_fn=_outcomes):
    noise = TorqueNoise("joint_torque", np.eye(N))
    return [
        replay_strategy(
            spec, "open", _case(ReplayPolicy.INDEPENDENT_TIME_ONLY), noise, outcome_fn
        ),
        replay_strategy(
            spec, "feedback", _case(ReplayPolicy.STATE_FEEDBACK), noise, outcome_fn
        ),
    ]


def test_zero_noise_gives_zero_dispersion_in_both_modes() -> None:
    spec = _spec(0.0, n_trials=3)
    result = pmc.run_campaign(spec, _strategies(spec))
    for name in ("open", "feedback"):
        summary = ps.summarise(spec, result.paired, name, n_boot=50, seed=1)
        assert summary.tally.failure_rate == 0.0
        for disp in summary.dispersions.values():
            assert disp is not None and disp.sd == 0.0
    nominal = pmc.run_seed_set(spec, _strategies(spec)[:1], "paired").outcome_matrix(
        "open"
    )
    assert np.ptp(nominal, axis=0).max() == 0.0  # every trial is the nominal replay


def test_torque_noise_moves_the_outcomes_and_is_common_to_both_modes() -> None:
    spec = _spec(2.0)
    result = pmc.run_campaign(spec, _strategies(spec))
    open_sum = ps.summarise(spec, result.paired, "open", n_boot=50, seed=1)
    fb_sum = ps.summarise(spec, result.paired, "feedback", n_boot=50, seed=1)
    for disp in open_sum.dispersions.values():
        assert disp is not None and disp.sd > 0.0
    # same draws, different dynamics: the outcomes differ, the draws do not
    assert not np.allclose(
        result.paired.outcome_matrix("open"), result.paired.outcome_matrix("feedback")
    )
    assert open_sum.evidence_label == "independent_open_loop_replay"
    assert fb_sum.evidence_label == "feedback_assisted_tracking"
    assert result.paired.modes == {
        "open": ReplayMode.OPEN_LOOP,
        "feedback": ReplayMode.FEEDBACK,
    }
    with pytest.raises(ValueError, match="open-loop and feedback"):
        ps.compare_strategies(spec, result.paired, "open", "feedback", n_boot=50)


def test_replay_integrator_signature_is_the_rk4_step_and_passes_the_r13_gate() -> None:
    case = _case(ReplayPolicy.INDEPENDENT_TIME_ONLY, substeps=4)
    sig = replay_integrator(case, DT)
    assert sig.step_s == pytest.approx(DT / 4)
    assert "RK4" in sig.name
    gen = GeneratorSpec(GENERATOR, noise=(_torque_noise(3.0),))
    spec = _spec(2.0, n_trials=2, generator=gen)
    result = pmc.run_campaign(spec, _strategies(spec))
    assert "noise_mismatched_to_generator" in result.flags
    for twin in (sig, IntegratorSignature(sig.name, 1.0)):  # same step / same name
        crime = _spec(2.0, 2, GeneratorSpec(twin, noise=(_torque_noise(3.0),)))
        with pytest.raises(ValueError, match="inverse crime"):
            pmc.run_campaign(crime, [_strategies(crime)[0]])


def test_blown_up_replays_are_retained_as_failures() -> None:
    spec = _spec(1e12, n_trials=4)
    result = pmc.run_campaign(spec, _strategies(spec)[:1])
    records = result.paired.records["open"]
    assert len(records) == 4
    tally = pmc.tally(records)
    assert tally.counts[pmc.TrialStatus.FAILED] == 4
    assert any("replay terminated" in (r.reason or "") for r in records)
    summary = ps.summarise(spec, result.paired, "open", n_boot=50, seed=1)
    assert summary.tally.failure_rate == 1.0 and summary.dispersions["v1_final"] is None


def test_a_missed_impact_from_the_outcome_map_is_counted_not_dropped() -> None:
    spec = _spec(2.0, n_trials=8)

    def outcome(report: ReplayReport) -> np.ndarray:
        if report.q[-1][0] > float(np.median(report.q[:, 0])) + 100.0:
            return _outcomes(report)
        raise pmc.TrialMiss("face never reached the ball plane")

    result = pmc.run_campaign(spec, _strategies(spec, outcome)[:1])
    assert pmc.tally(result.paired.records["open"]).counts[pmc.TrialStatus.MISSED] == 8


def test_strategy_construction_checks_the_grid_and_the_noise_map() -> None:
    spec = _spec(1.0)
    case = _case(ReplayPolicy.INDEPENDENT_TIME_ONLY)
    noise = TorqueNoise("joint_torque", np.eye(N))
    other_grid = CampaignSpec(**{**spec.__dict__, "dt": DT * 2.0})
    with pytest.raises(ValueError, match="allocation interval"):
        replay_strategy(other_grid, "x", case, noise, _outcomes)
    with pytest.raises(ValueError, match="unknown disturbance"):
        replay_strategy(spec, "x", case, TorqueNoise("nope", np.eye(N)), _outcomes)
    with pytest.raises(ValueError, match="load_map"):
        replay_strategy(
            spec, "x", case, TorqueNoise("joint_torque", np.eye(3)), _outcomes
        )
    with pytest.raises(ValueError, match="n_steps"):
        replay_strategy(
            CampaignSpec(**{**spec.__dict__, "n_steps": 3}), "x", case, noise, _outcomes
        )
    with pytest.raises(TypeError):
        replay_strategy(spec, "x", case, noise, "not callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="persistent|white"):
        replay_strategy(
            CampaignSpec(
                **{
                    **spec.__dict__,
                    "disturbances": (
                        DisturbanceSpec(
                            name="joint_torque",
                            stage=InjectionStage.TORQUE,
                            channels=tuple(f"t{i}" for i in range(N)),
                            units="N m",
                            covariance=np.eye(N),
                            persistent=True,
                        ),
                    ),
                }
            ),
            "x",
            case,
            noise,
            _outcomes,
        )

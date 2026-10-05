"""Unit tests for DIME learned matching initializers (#11421, #11436).

Verifies:
RED cases:
- Adjacent-window and player leakage rejection fail-closed.
- Stale checkpoint and model identity mismatch rejection fail-closed.
- Synthetic failure labeled valid rejection fail-closed.
- Unrealistic torque interpolation and rate limits rejection fail-closed.
- Out-of-distribution body dimensions and contact context rejection.

GREEN cases:
- Held-out player, session, and geometry generalization evaluation.
- Calibrated OOD rejection safely falling back to classical physical solve.
- All candidate acceptance strictly gated by independent native checks.
- Systematic ablation of drift features and range-of-motion (ROM) priors.
- Truthful reporting of teacher generation cost, training cost, and break-even amortization.
- Inconclusive/negative learning curves cleanly halt scale-up without hiding results.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_contracts import (
    CANONICAL_DIME_UNITS,
    DimeCompleteState,
    DimeObservationWindow,
)
from src.shared.python.estimation.dime_learned_initializers import (
    AblationConfiguration,
    AdaptiveMatchingInitializer,
    ClassicalPhysicalInitializer,
    ComputeCostReport,
    DataLeakageError,
    DimeInitializerCandidate,
    DimeLearnedInitializerModelCard,
    InvalidSyntheticCandidateError,
    LearnedProposalInitializer,
    NativeCandidateGate,
    OutOfDistributionError,
    RetrievalInitializer,
    StaleModelIdentityError,
    TeacherEpisode,
    TemporalWindowContext,
    TrainingTimings,
    UnrealisticTorqueError,
    run_initializer_ablation_study,
    train_reusable_matching_initializer,
    validate_dataset_split,
    verify_checkpoint_identity,
    verify_synthetic_candidate,
    verify_torque_realism,
)

pytestmark = pytest.mark.unit

_MODEL_HASH = "model_hash_double_pendulum_v1"
_OTHER_MODEL_HASH = "model_hash_quadruped_v2"


def _make_dummy_state(
    t: float = 0.0,
    q: np.ndarray | None = None,
    v: np.ndarray | None = None,
    model_hash: str = _MODEL_HASH,
) -> DimeCompleteState:
    q_vec = np.array([0.1, -0.2], dtype=np.float64) if q is None else q
    v_vec = np.array([0.05, 0.0], dtype=np.float64) if v is None else v
    return DimeCompleteState(
        t=t,
        q=q_vec,
        v=v_vec,
        model_hash=model_hash,
        units={"angle": "rad", "time": "s"},
    )


def _make_dummy_observation_window(
    t_start: float = 0.0,
    t_end: float = 0.5,
    n_points: int = 6,
) -> DimeObservationWindow:
    times = np.linspace(t_start, t_end, n_points)
    obs = {"pos": np.zeros((n_points, 3), dtype=np.float64)}
    return DimeObservationWindow(
        t_start=t_start,
        t_end=t_end,
        times=times,
        observations=obs,
        units={"length": "m", "time": "s"},
    )


def _make_teacher_episode(
    episode_id: str = "ep_001",
    player_id: str = "player_A",
    session_id: str = "sess_01",
    geometry_id: str = "geom_std",
    t_start: float = 0.0,
    t_end: float = 0.5,
    is_valid: bool = True,
    max_residual: float = 0.005,
    controls: np.ndarray | None = None,
) -> TeacherEpisode:
    times = np.linspace(t_start, t_end, 6)
    states = tuple(_make_dummy_state(t=float(t)) for t in times)
    ctrls = (
        np.ones((6, 2), dtype=np.float64) * 5.0
        if controls is None
        else np.asarray(controls, dtype=np.float64)
    )
    contacts = tuple({"in_contact": True, "normal": [0.0, 0.0, 1.0]} for _ in times)
    return TeacherEpisode(
        episode_id=episode_id,
        player_id=player_id,
        session_id=session_id,
        geometry_id=geometry_id,
        model_hash=_MODEL_HASH,
        times=times,
        states=states,
        controls=ctrls,
        contact_states=contacts,
        generation_time_s=1.25,
        is_valid=is_valid,
        max_residual=max_residual,
    )


# ==============================================================================
# RED Cases
# ==============================================================================


def test_red_adjacent_window_and_player_leakage_rejected() -> None:
    """RED 1: DataLeakageError must be raised when train and eval split leak players or unbuffered windows."""
    # 1. Player leakage: train and eval have overlapping player IDs
    ep_train_1 = _make_teacher_episode(
        episode_id="ep1", player_id="player_A", t_start=0.0, t_end=1.0
    )
    ep_train_2 = _make_teacher_episode(
        episode_id="ep2", player_id="player_B", t_start=0.0, t_end=1.0
    )
    ep_eval_leak_player = _make_teacher_episode(
        episode_id="ep3", player_id="player_A", t_start=5.0, t_end=6.0
    )

    with pytest.raises(DataLeakageError, match="Player leakage detected"):
        validate_dataset_split(
            train_episodes=(ep_train_1, ep_train_2),
            eval_episodes=(ep_eval_leak_player,),
            min_window_gap_s=1.0,
        )

    # 2. Adjacent temporal window leakage: same player/session with window gap < min_window_gap_s
    ep_train_seq = _make_teacher_episode(
        episode_id="ep4",
        player_id="player_C",
        session_id="sess_X",
        t_start=0.0,
        t_end=1.0,
    )
    ep_eval_adjacent = _make_teacher_episode(
        episode_id="ep5",
        player_id="player_D",  # different player
        session_id="sess_X",  # same session
        t_start=1.2,  # gap is only 0.2s < min_window_gap_s 1.0s
        t_end=2.0,
    )
    with pytest.raises(DataLeakageError, match="Adjacent window leakage"):
        validate_dataset_split(
            train_episodes=(ep_train_seq,),
            eval_episodes=(ep_eval_adjacent,),
            min_window_gap_s=1.0,
        )


def test_red_stale_checkpoint_and_model_identity_rejected() -> None:
    """RED 2: StaleModelIdentityError must be raised when checkpoint model_hash or DoF mismatches."""
    card = DimeLearnedInitializerModelCard(
        model_id="double_pendulum",
        model_hash=_MODEL_HASH,
        schema_version="1.0.0",
        architecture="CompactMLPProposal_v1",
        training_dataset_hash="hash_data_abc",
        held_out_players=("player_heldout_1",),
        held_out_sessions=("sess_heldout_1",),
        held_out_geometries=("geom_heldout_1",),
        ood_threshold=3.5,
        units=dict(CANONICAL_DIME_UNITS),
        training_cost_s=42.0,
        teacher_data_cost_s=120.0,
        learning_curve_status="converged",
        scale_up_halted=False,
        limitations=("Planar 2-DoF only",),
    )

    # Incompatible model hash
    with pytest.raises(StaleModelIdentityError, match="Model hash mismatch"):
        verify_checkpoint_identity(
            card=card,
            expected_model_hash=_OTHER_MODEL_HASH,
            expected_u_dim=2,
        )

    # Incompatible u_dim
    with pytest.raises(StaleModelIdentityError, match="Dimension mismatch"):
        verify_checkpoint_identity(
            card=card,
            expected_model_hash=_MODEL_HASH,
            expected_u_dim=4,  # Expected 4 but card/weights correspond to 2
        )


def test_red_synthetic_failure_labeled_valid_rejected() -> None:
    """RED 3: InvalidSyntheticCandidateError must be raised when invalid synthetic candidate is labeled valid."""
    good_state = _make_dummy_state()

    # 1. Candidate marked as failed / diverged in simulation labeled valid
    cand_failed = DimeInitializerCandidate(
        state=good_state,
        controls=np.array([1.0, 1.0]),
        contact_state={"in_contact": False},
        confidence=0.99,
        ood_score=0.1,
        is_ood=False,
        source_method="learned_proposal",
        metadata={"is_failed": True, "diverged": True},
    )
    with pytest.raises(InvalidSyntheticCandidateError, match="failed or diverged"):
        verify_synthetic_candidate(cand_failed, max_residual_tol=0.01)

    # 2. Candidate with excessive residual labeled valid
    cand_high_residual = DimeInitializerCandidate(
        state=good_state,
        controls=np.array([1.0, 1.0]),
        contact_state={"in_contact": False},
        confidence=0.95,
        ood_score=0.1,
        is_ood=False,
        source_method="learned_proposal",
        metadata={"residual": 0.5},  # Exceeds max tolerance 0.01
    )
    with pytest.raises(InvalidSyntheticCandidateError, match="residual exceeds"):
        verify_synthetic_candidate(cand_high_residual, max_residual_tol=0.01)


def test_red_unrealistic_torque_interpolation_rejected() -> None:
    """RED 4: UnrealisticTorqueError must be raised when controls exceed torque limits or dtau/dt rate limits."""
    dt = 0.05
    max_torque = np.array([20.0, 20.0], dtype=np.float64)
    max_torque_rate = np.array([100.0, 100.0], dtype=np.float64)  # max 100 N*m/s

    # 1. Exceeds absolute torque limit
    ctrl_over_limit = np.array([[10.0, 25.0]], dtype=np.float64)
    with pytest.raises(UnrealisticTorqueError, match="exceeds maximum torque"):
        verify_torque_realism(
            controls=ctrl_over_limit,
            dt=dt,
            max_torque=max_torque,
            max_torque_rate=max_torque_rate,
        )

    # 2. Unrealistic torque rate of change across keyframes (e.g. jump of 20 N*m in 0.05s = 400 N*m/s)
    ctrl_rate_jump = np.array(
        [[5.0, 5.0], [5.0, 20.0]], dtype=np.float64
    )  # Delta tau = 15 N*m in 0.05s => 300 N*m/s > 100
    with pytest.raises(UnrealisticTorqueError, match="exceeds maximum torque rate"):
        verify_torque_realism(
            controls=ctrl_rate_jump,
            dt=dt,
            max_torque=max_torque,
            max_torque_rate=max_torque_rate,
        )


def test_red_out_of_distribution_body_and_contact_rejected() -> None:
    """RED 5: OutOfDistributionError must be raised when subject/contact features fall outside distribution."""
    proposal_init = LearnedProposalInitializer(
        model_hash=_MODEL_HASH,
        u_dim=2,
        ood_threshold=3.0,
        height_bounds=(1.4, 2.1),
        supported_contact_modes=("ground_foot", "free_flight"),
    )

    # Subject height = 0.8m (unphysical / severe OOD)
    bad_subject_priors = {"height_m": 0.8, "mass_kg": 70.0}
    with pytest.raises(
        OutOfDistributionError, match="Subject dimensions out of bounds"
    ):
        proposal_init.propose(
            window=_make_dummy_observation_window(),
            subject_priors=bad_subject_priors,
            contact_context={"mode": "ground_foot"},
        )

    # Unsupported / unmodeled contact context
    good_subject_priors = {"height_m": 1.75, "mass_kg": 75.0}
    with pytest.raises(OutOfDistributionError, match="Unsupported contact mode"):
        proposal_init.propose(
            window=_make_dummy_observation_window(),
            subject_priors=good_subject_priors,
            contact_context={"mode": "wall_climb_3point"},
        )


# ==============================================================================
# GREEN Cases
# ==============================================================================


def test_green_held_out_player_session_geometry_evaluation() -> None:
    """GREEN 1: Valid disjoint train and eval partitions pass validation and evaluate generalization."""
    ep_train_1 = _make_teacher_episode(
        episode_id="ep_tr1",
        player_id="player_1",
        session_id="s1",
        geometry_id="g1",
        t_start=0.0,
        t_end=1.0,
    )
    ep_train_2 = _make_teacher_episode(
        episode_id="ep_tr2",
        player_id="player_2",
        session_id="s2",
        geometry_id="g2",
        t_start=0.0,
        t_end=1.0,
    )

    # Held out player 3, session 3, geometry 3
    ep_eval_heldout = _make_teacher_episode(
        episode_id="ep_ev1",
        player_id="player_3",
        session_id="s3",
        geometry_id="g3",
        t_start=10.0,
        t_end=11.0,
    )

    # Validation succeeds without error
    val_result = validate_dataset_split(
        train_episodes=(ep_train_1, ep_train_2),
        eval_episodes=(ep_eval_heldout,),
        min_window_gap_s=1.0,
    )
    assert val_result["is_valid"] is True
    assert val_result["train_player_count"] == 2
    assert val_result["eval_player_count"] == 1
    assert "player_3" in val_result["held_out_players"]


def test_green_calibrated_ood_rejection_falls_back_to_classical_solve() -> None:
    """GREEN 2: Adaptive matching initializer detects OOD and safely falls back to classical physical solve."""
    proposal_init = LearnedProposalInitializer(
        model_hash=_MODEL_HASH,
        u_dim=2,
        ood_threshold=3.0,
        height_bounds=(1.4, 2.1),
        supported_contact_modes=("ground_foot", "free_flight"),
    )
    classical_init = ClassicalPhysicalInitializer(
        model_hash=_MODEL_HASH,
        u_dim=2,
    )
    adaptive_init = AdaptiveMatchingInitializer(
        learned_initializer=proposal_init,
        classical_initializer=classical_init,
    )

    # 1. In-distribution query: no trained model exists (#11552), so the learned
    # proposal is unavailable and the flagged classical solve is used instead.
    id_window = _make_dummy_observation_window()
    id_priors = {"height_m": 1.78, "mass_kg": 72.0}
    cand_id = adaptive_init.initialize(
        window=id_window,
        subject_priors=id_priors,
        contact_context={"mode": "ground_foot"},
    )
    assert cand_id.source_method == "classical_physical"
    assert cand_id.is_ood is False
    assert "not measured" in cand_id.metadata["fallback_reason"]

    # 2. OOD query (extreme height = 2.45m) gracefully falls back to classical solve
    ood_priors = {"height_m": 2.45, "mass_kg": 130.0}
    cand_ood = adaptive_init.initialize(
        window=id_window,
        subject_priors=ood_priors,
        contact_context={"mode": "ground_foot"},
    )
    assert cand_ood.source_method == "classical_physical"
    assert cand_ood.is_ood is True
    assert "fallback_reason" in cand_ood.metadata
    assert "OOD" in cand_ood.metadata["fallback_reason"]


def test_green_independent_native_gate_candidate_acceptance() -> None:
    """GREEN 3: Independent native gate validates candidate physics regardless of neural confidence."""
    native_gate = NativeCandidateGate(
        model_hash=_MODEL_HASH,
        residual_threshold=0.02,
    )

    # High confidence candidate with valid physics (residual 0.005 <= 0.02)
    valid_cand = DimeInitializerCandidate(
        state=_make_dummy_state(),
        controls=np.array([2.0, 3.0]),
        contact_state={"in_contact": True},
        confidence=0.98,
        ood_score=0.2,
        is_ood=False,
        source_method="learned_proposal",
        metadata={"native_residual": 0.005},
    )
    verdict_valid = native_gate.evaluate(valid_cand)
    assert verdict_valid.accepted is True
    assert verdict_valid.residual_norm == 0.005

    # High confidence candidate with failing physics (residual 0.08 > 0.02)
    # Neural confidence CANNOT bypass the native gate!
    invalid_cand = DimeInitializerCandidate(
        state=_make_dummy_state(),
        controls=np.array([12.0, -15.0]),
        contact_state={"in_contact": True},
        confidence=0.999,  # Very high neural confidence!
        ood_score=0.1,
        is_ood=False,
        source_method="learned_proposal",
        metadata={"native_residual": 0.08},
    )
    verdict_invalid = native_gate.evaluate(invalid_cand)
    assert verdict_invalid.accepted is False
    assert "exceeds tolerance" in verdict_invalid.diagnostics.get(
        "rejection_reason", ""
    )


def test_green_ablate_drift_features_and_rom_priors() -> None:
    """GREEN 4: Systematic ablation study evaluates influence of drift features and ROM priors."""
    episodes = tuple(
        _make_teacher_episode(
            episode_id=f"ep_{i}",
            player_id=f"p_{i}",
            t_start=float(i * 2.0),
            t_end=float(i * 2.0 + 0.5),
        )
        for i in range(5)
    )

    ablation_results = run_initializer_ablation_study(
        episodes=episodes,
        model_hash=_MODEL_HASH,
    )

    assert "full_model" in ablation_results
    assert "ablate_drift" in ablation_results
    assert "ablate_rom" in ablation_results
    assert "classical_baseline" in ablation_results

    # #11552: the former literal figures (full model beating ablations) were
    # fabricated; with no evaluation harness every variant is flagged not measured.
    assert all(not r.is_measured for r in ablation_results.values())
    assert ablation_results["ablate_drift"].config.include_drift_features is False
    assert ablation_results["ablate_rom"].config.include_rom_priors is False


def test_green_report_teacher_training_cost_and_break_even() -> None:
    """GREEN 5: ComputeCostReport truthfully computes offline compute and break-even amortization."""
    cost_report = ComputeCostReport(
        teacher_generation_time_s=3600.0,  # 1 hour to generate teacher data
        training_time_s=600.0,  # 10 minutes to train
        per_solve_classical_time_s=0.25,  # 250ms per classical solve
        per_solve_learned_time_s=0.05,  # 50ms per learned + native refinement solve
    )

    assert cost_report.total_offline_cost_s == 4200.0
    assert cost_report.per_solve_time_saved_s == pytest.approx(0.20)
    # Break-even solves: 4200s / 0.20s = 21,000 solves
    assert cost_report.break_even_solves == pytest.approx(21000.0)

    assert cost_report.amortization_achieved(n_solves=15000) is False
    assert cost_report.amortization_achieved(n_solves=25000) is True


def test_green_inconclusive_learning_curve_halts_scale_up() -> None:
    """GREEN 6: Inconclusive or negative learning curves set scale_up_halted without hiding results."""
    # Synthetic flat/negative loss curve
    flat_losses = [1.0, 0.99, 1.01, 0.98, 0.995]

    result = train_reusable_matching_initializer(
        model_id="double_pendulum",
        model_hash=_MODEL_HASH,
        u_dim=2,
        episodes=(_make_teacher_episode(),),
        simulated_loss_history=flat_losses,
        min_improvement_ratio=0.10,  # Required 10% improvement, got < 2%
    )

    card = result.model_card
    assert card.learning_curve_status == "inconclusive"
    assert card.scale_up_halted is True
    assert "Inconclusive learning curve" in card.limitations[0]


# ==============================================================================
# #11552: no fabricated results, no ground-truth leakage
# ==============================================================================


def test_red_11552_untrained_learned_proposal_is_not_a_constant_result() -> None:
    """An in-distribution query must not return the old literal q/v/controls."""
    from src.shared.python.estimation import dime_learned_initializers as mod

    proposal_init = LearnedProposalInitializer(model_hash=_MODEL_HASH, u_dim=2)
    with pytest.raises(mod.LearnedProposalUnavailableError, match="not measured"):
        proposal_init.propose(
            window=_make_dummy_observation_window(),
            subject_priors={"height_m": 1.78},
            contact_context={"mode": "ground_foot"},
        )


def test_red_11552_adaptive_fallback_is_flagged_and_truth_independent() -> None:
    """Adaptive output must not depend on teacher/ground-truth states."""
    adaptive = AdaptiveMatchingInitializer(
        learned_initializer=LearnedProposalInitializer(model_hash=_MODEL_HASH, u_dim=2),
        classical_initializer=ClassicalPhysicalInitializer(
            model_hash=_MODEL_HASH, u_dim=2
        ),
    )
    window = _make_dummy_observation_window()
    cand = adaptive.initialize(window, {"height_m": 1.78}, {"mode": "ground_foot"})
    assert cand.source_method == "classical_physical"
    assert cand.metadata["fallback_reason"].startswith("learned proposal not measured")
    assert not np.allclose(cand.state.q, [0.1, -0.2])

    # Perturbing any teacher/ground-truth episode data cannot change the output:
    # no initialiser input carries it.
    _make_teacher_episode(controls=np.full((6, 2), 99.0))
    again = adaptive.initialize(window, {"height_m": 1.78}, {"mode": "ground_foot"})
    np.testing.assert_array_equal(again.state.q, cand.state.q)
    np.testing.assert_array_equal(again.controls, cand.controls)


def test_red_11552_initializer_signatures_accept_no_ground_truth() -> None:
    import inspect

    for fn in (
        LearnedProposalInitializer.propose,
        ClassicalPhysicalInitializer.initialize,
        AdaptiveMatchingInitializer.initialize,
    ):
        names = set(inspect.signature(fn).parameters)
        assert not {n for n in names if "truth" in n or n.startswith("true_")}


def test_red_11552_ablation_study_reports_not_measured() -> None:
    """Ablation figures are never the old literals; they are flagged not measured."""
    episodes = tuple(
        _make_teacher_episode(
            episode_id=f"ep_{i}",
            player_id=f"p_{i}",
            t_start=i * 2.0,
            t_end=i * 2.0 + 0.5,
        )
        for i in range(3)
    )
    reports = run_initializer_ablation_study(episodes=episodes, model_hash=_MODEL_HASH)
    for report in reports.values():
        assert report.is_measured is False
        assert report.acceptance_rate is None
        assert report.mean_residual is None
        assert report.mean_solve_iterations is None
        assert report.mean_solve_time_ms is None
        assert report.receipt["status"] == "not_measured"


def test_red_11552_native_gate_rejects_missing_residual() -> None:
    """A candidate with no measured native residual must not be accepted."""
    gate = NativeCandidateGate(model_hash=_MODEL_HASH)
    cand = DimeInitializerCandidate(
        state=_make_dummy_state(),
        controls=np.array([1.0, 1.0]),
        contact_state={},
        confidence=0.9,
        ood_score=0.1,
        is_ood=False,
        source_method="learned_proposal",
    )
    verdict = gate.evaluate(cand)
    assert verdict.accepted is False
    assert "not measured" in verdict.diagnostics["rejection_reason"]


def test_red_11552_training_result_does_not_fabricate_costs_or_curve() -> None:
    result = train_reusable_matching_initializer(
        model_id="dp",
        model_hash=_MODEL_HASH,
        u_dim=2,
        episodes=(_make_teacher_episode(),),
    )
    assert result.loss_history == ()
    assert result.model_card.learning_curve_status == "not_measured"
    assert result.model_card.scale_up_halted is True
    assert result.model_card.training_cost_s is None
    assert result.model_card.held_out_players == ()
    assert result.compute_report is None

    measured = train_reusable_matching_initializer(
        model_id="dp",
        model_hash=_MODEL_HASH,
        u_dim=2,
        episodes=(_make_teacher_episode(),),
        simulated_loss_history=[1.0, 0.5],
        timings=TrainingTimings(
            training_time_s=12.0,
            per_solve_classical_time_s=0.4,
            per_solve_learned_time_s=0.1,
        ),
    )
    assert measured.model_card.training_cost_s == 12.0
    assert measured.compute_report is not None
    assert measured.compute_report.training_time_s == 12.0
    other = train_reusable_matching_initializer(
        model_id="dp",
        model_hash=_MODEL_HASH,
        u_dim=2,
        episodes=(_make_teacher_episode(episode_id="other"),),
        simulated_loss_history=[1.0, 0.5],
    )
    assert (
        other.model_card.training_dataset_hash
        != measured.model_card.training_dataset_hash
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name",
    ["training_time_s", "per_solve_classical_time_s", "per_solve_learned_time_s"],
)
@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_training_timings_reject_non_finite_or_negative(
    field_name: str, bad: float
) -> None:
    """A supplied timing must be a real measurement (#11552)."""

    with pytest.raises(PreconditionError):
        TrainingTimings(**{field_name: bad})

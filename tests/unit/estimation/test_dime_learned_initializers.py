"""Behavioral unit test suite for DIME learned matching initializers (DIME-15, #11436).

Parent: Epic #11421.
Dependencies: #11429, #11430, #11434, #11435.

Required RED cases:
1. Adjacent-window / player leakage detection (reject train/val overlaps).
2. Stale checkpoint or model identity mismatch fails closed.
3. Synthetic solver failure labeled valid is rejected.
4. Unrealistic torque interpolation rejected.
5. Out-of-distribution body geometry or contact context rejected.

Required GREEN cases:
1. Held-out player/session/geometry evaluation passes without data leakage.
2. Calibrated OOD rejection falls back gracefully to classical physical solve.
3. Candidate acceptance verified through independent native gates.
4. Ablation study cleanly evaluates impact of drift features and ROM priors.
5. Reports teacher data generation cost, training compute, and break-even comparison vs. inference time alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_learned_initializers import (
    AblationVariantReport,
    BreakEvenEconomics,
    EarlyStoppingStatus,
    ExperimentBounds,
    InitializerCandidate,
    InitializerComparisonReport,
    InitializerComparisonResult,
    InitializerInput,
    InitializerModelCard,
    InitializerSource,
    TemporalWindow,
    compare_matching_initializers,
    compute_initializer_breakeven,
    detect_out_of_distribution,
    generate_learned_initialization,
    interpolate_and_validate_torques,
    monitor_learning_curve,
    run_initializer_ablation,
    validate_checkpoint_identity,
    validate_split_leakage,
    validate_synthetic_solver_outcome,
    verify_candidate_with_native_gate,
)
from src.shared.python.estimation.dime_providers import (
    AnalyticPendulumProvider,
)

pytestmark = pytest.mark.unit


# =============================================================================
# RED Cases: Contract Enforcement & Defensive Guards
# =============================================================================


def test_red_adjacent_window_player_leakage_detected() -> None:
    """Train and validation splits with player, session, or temporal window overlap must be rejected."""
    # Case 1: Player leakage (same player in train and val)
    train_win_1 = TemporalWindow(
        window_id="w_train_1",
        player_id="player_001",
        session_id="session_A",
        t_start=0.0,
        t_end=1.0,
    )
    val_win_player_leak = TemporalWindow(
        window_id="w_val_1",
        player_id="player_001",
        session_id="session_B",
        t_start=10.0,
        t_end=11.0,
    )
    with pytest.raises(PreconditionError, match="(?i)player leakage"):
        validate_split_leakage([train_win_1], [val_win_player_leak])

    # Case 2: Session leakage (same session across distinct players)
    val_win_session_leak = TemporalWindow(
        window_id="w_val_2",
        player_id="player_002",
        session_id="session_A",
        t_start=20.0,
        t_end=21.0,
    )
    with pytest.raises(PreconditionError, match="(?i)session leakage"):
        validate_split_leakage([train_win_1], [val_win_session_leak])

    # Case 3: Temporal window overlap / adjacent-window leakage (same subject recording)
    w_early = TemporalWindow(
        window_id="w_early",
        player_id="player_100",
        session_id="session_100",
        t_start=0.0,
        t_end=1.0,
    )
    w_overlapping = TemporalWindow(
        window_id="w_overlap",
        player_id="player_100",
        session_id="session_100",
        t_start=0.95,
        t_end=1.95,
    )
    with pytest.raises(PreconditionError, match="(?i)leakage"):
        validate_split_leakage([w_early], [w_overlapping], buffer_s=0.1)


def test_red_stale_checkpoint_or_model_identity_mismatch_fails_closed() -> None:
    """Loading a checkpoint for the wrong model or a stale/deprecated checkpoint must fail closed."""
    valid_card = {
        "model_id": "pendulum_v1",
        "model_hash": "hash_pendulum_v1",
        "schema": "dime-learned-initializers/1.0.0",
        "is_stale": False,
        "created_timestamp_s": 1000.0,
    }

    # Model ID mismatch
    with pytest.raises(PreconditionError, match="(?i)model_id mismatch"):
        validate_checkpoint_identity(
            valid_card,
            expected_model_id="humanoid_golf_v2",
            expected_model_hash="hash_pendulum_v1",
        )

    # Model Hash mismatch
    with pytest.raises(PreconditionError, match="(?i)model_hash mismatch"):
        validate_checkpoint_identity(
            valid_card,
            expected_model_id="pendulum_v1",
            expected_model_hash="hash_other_v1",
        )

    # Stale / deprecated checkpoint
    stale_card = dict(valid_card, is_stale=True)
    with pytest.raises(PreconditionError, match="(?i)stale"):
        validate_checkpoint_identity(
            stale_card,
            expected_model_id="pendulum_v1",
            expected_model_hash="hash_pendulum_v1",
        )


def test_red_synthetic_solver_failure_labeled_valid_rejected() -> None:
    """Synthetic teacher generation where solver failed must never be accepted as valid training data."""
    # Solver did not converge but labeled valid
    with pytest.raises(PreconditionError, match="(?i)synthetic solver failure"):
        validate_synthetic_solver_outcome(
            converged=False,
            residual=0.01,
            tolerance=0.05,
            labeled_valid=True,
        )

    # Rollout residual exceeds tolerance
    with pytest.raises(PreconditionError, match="(?i)residual exceeded tolerance"):
        validate_synthetic_solver_outcome(
            converged=True,
            residual=0.25,
            tolerance=0.05,
            labeled_valid=True,
        )

    # Unphysical torques detected in outcome
    with pytest.raises(PreconditionError, match="(?i)unphysical torques"):
        validate_synthetic_solver_outcome(
            converged=True,
            residual=0.01,
            tolerance=0.05,
            labeled_valid=True,
            has_unphysical_torques=True,
        )


def test_red_unrealistic_torque_interpolation_rejected() -> None:
    """Torque interpolation with unphysical magnitudes or rate-of-change spikes must fail closed."""
    knots_t = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    query_t = np.linspace(0.0, 1.0, 11)

    # Magnitude exceeds torque limit (e.g. 1000 Nm when limit is 500 Nm)
    knots_tau_excessive = np.array([[0.0], [1200.0], [0.0]], dtype=np.float64)
    with pytest.raises(PreconditionError, match="(?i)torque magnitude"):
        interpolate_and_validate_torques(
            knots_t,
            knots_tau_excessive,
            query_t,
            max_torque=500.0,
            max_torque_rate=2000.0,
        )

    # Derivative spike / jerk exceeds physical limit
    knots_tau_spike = np.array([[0.0], [450.0], [-450.0]], dtype=np.float64)
    with pytest.raises(PreconditionError, match="(?i)rate-of-change"):
        interpolate_and_validate_torques(
            np.array([0.0, 0.001, 0.002], dtype=np.float64),
            knots_tau_spike,
            np.linspace(0.0, 0.002, 10),
            max_torque=500.0,
            max_torque_rate=50000.0,  # 900 Nm in 0.001s = 900,000 Nm/s
        )


def test_red_out_of_distribution_geometry_or_contact_rejected() -> None:
    """Extreme out-of-distribution body dimensions or unrecognized contact must be detected and rejected."""
    # Geometry with extreme Mahalanobis distance / unphysical proportions
    ood_dims = {"height": 3.80, "torso": 2.50, "thigh": 0.20}
    is_ood, reason, conf = detect_out_of_distribution(
        subject_dimensions=ood_dims,
        contact_context="stance",
        geometry_mahalanobis_threshold=3.0,
    )
    assert is_ood is True
    assert reason is not None
    assert "geometry" in reason.lower()
    assert conf < 0.5

    # Unrecognized contact context
    in_dist_dims = {"height": 1.78, "torso": 0.60, "thigh": 0.45}
    is_ood_contact, reason_contact, conf_contact = detect_out_of_distribution(
        subject_dimensions=in_dist_dims,
        contact_context="aerial_trampoline_unsupported",
    )
    assert is_ood_contact is True
    assert reason_contact is not None
    assert "contact" in reason_contact.lower()
    assert conf_contact < 0.5

    # In strict mode, generating initialization for OOD input raises PreconditionError
    inp = InitializerInput(
        observation_trajectory=np.zeros((10, 2)),
        observation_mask=np.ones(2),
        sample_times_s=np.linspace(0.0, 0.1, 10),
        model_id="pendulum_v1",
        model_hash="hash_pendulum",
        player_id="p1",
        session_id="s1",
        subject_dimensions=ood_dims,
        contact_context="stance",
    )
    with pytest.raises(PreconditionError, match="(?i)out-of-distribution"):
        generate_learned_initialization(inp, fallback_to_classical=False)


# =============================================================================
# GREEN Cases: Qualified Lifecycle & Behavioral Acceptance
# =============================================================================


def test_green_held_out_player_session_geometry_evaluation() -> None:
    """Disjoint players, sessions, and non-overlapping windows pass split validation cleanly."""
    train_windows = [
        TemporalWindow(
            window_id="w_tr_1",
            player_id="p_alpha",
            session_id="sess_1",
            t_start=0.0,
            t_end=2.0,
        ),
        TemporalWindow(
            window_id="w_tr_2",
            player_id="p_beta",
            session_id="sess_2",
            t_start=0.0,
            t_end=2.0,
        ),
    ]
    val_windows = [
        TemporalWindow(
            window_id="w_val_1",
            player_id="p_gamma",
            session_id="sess_3",
            t_start=5.0,
            t_end=7.0,
        ),
    ]
    # No error raised
    validate_split_leakage(train_windows, val_windows, buffer_s=0.5)


def test_green_calibrated_ood_rejection_falls_back_gracefully() -> None:
    """Calibrated OOD detector flags out-of-distribution input and falls back gracefully to classical solve."""
    ood_dims = {"height": 3.50, "torso": 2.20}
    inp = InitializerInput(
        observation_trajectory=np.ones((8, 2)),
        observation_mask=np.ones(2),
        sample_times_s=np.linspace(0.0, 0.08, 8),
        model_id="pendulum_v1",
        model_hash="hash_pendulum",
        player_id="p_heldout",
        session_id="s_heldout",
        subject_dimensions=ood_dims,
        contact_context="stance",
    )

    candidate = generate_learned_initialization(inp, fallback_to_classical=True)
    assert candidate.is_ood is True
    assert candidate.source == InitializerSource.FALLBACK_CLASSICAL
    assert candidate.calibrated_confidence < 0.5
    assert candidate.q_init.shape == (1,)
    assert candidate.v_init.shape == (1,)
    assert candidate.u_init.shape == (1,)
    assert np.all(np.isfinite(candidate.q_init))
    assert np.all(np.isfinite(candidate.v_init))
    assert np.all(np.isfinite(candidate.u_init))


def test_green_candidate_acceptance_verified_through_independent_native_gates() -> None:
    """Independent native physical gates verify candidate validity."""
    provider = AnalyticPendulumProvider()

    # Valid candidate within physics tolerances
    good_candidate = InitializerCandidate(
        q_init=np.array([0.1], dtype=np.float64),
        v_init=np.array([0.0], dtype=np.float64),
        u_init=np.array([0.0], dtype=np.float64),
        contact_init=None,
        calibrated_confidence=0.95,
        is_ood=False,
        ood_reason=None,
        source=InitializerSource.LEARNED_PROPOSAL,
    )
    passed, msg = verify_candidate_with_native_gate(
        good_candidate, provider, tolerance=0.1
    )
    assert passed is True
    assert "passed" in msg.lower()

    # Infeasible candidate (unrealistically high velocities / forces failing residual)
    bad_candidate = InitializerCandidate(
        q_init=np.array([0.1], dtype=np.float64),
        v_init=np.array([100.0], dtype=np.float64),
        u_init=np.array([500.0], dtype=np.float64),
        contact_init=None,
        calibrated_confidence=0.95,
        is_ood=False,
        ood_reason=None,
        source=InitializerSource.LEARNED_PROPOSAL,
    )
    passed_bad, msg_bad = verify_candidate_with_native_gate(
        bad_candidate, provider, tolerance=0.01
    )
    assert passed_bad is False
    assert "failed" in msg_bad.lower()


def test_green_ablation_evaluates_drift_features_and_rom_priors() -> None:
    """Ablation study cleanly isolates the impact of drift features and ROM priors."""
    provider = AnalyticPendulumProvider()

    inp = InitializerInput(
        observation_trajectory=np.zeros((8, 2)),
        observation_mask=np.ones(2),
        sample_times_s=np.linspace(0.0, 0.08, 8),
        model_id="pendulum_v1",
        model_hash="hash_pendulum",
        player_id="p1",
        session_id="s1",
        subject_dimensions={"height": 1.75},
        drift_features=np.array([0.0, -9.81]),
        q_prior=np.array([0.0]),
    )

    reports = run_initializer_ablation(inp, provider)
    assert len(reports) == 4
    names = {r.variant_name for r in reports}
    assert "full_learned_initializer" in names
    assert "no_drift_features" in names
    assert "no_rom_priors" in names
    assert "classical_warm_start" in names

    full = next(r for r in reports if r.variant_name == "full_learned_initializer")
    no_drift = next(r for r in reports if r.variant_name == "no_drift_features")
    assert full.use_drift_features is True
    assert no_drift.use_drift_features is False
    assert full.acceptance_rate >= 0.0


def test_green_break_even_and_training_compute_reporting() -> None:
    """Break-even economics accurately balances teacher generation, training, and inference time."""
    breakeven = compute_initializer_breakeven(
        teacher_cost_s=120.0,
        train_cost_s=60.0,
        classical_time_ms=50.0,
        learned_time_ms=10.0,
    )
    assert breakeven.teacher_generation_cost_s == 120.0
    assert breakeven.training_compute_cost_s == 60.0
    assert breakeven.speedup_per_query_ms == 40.0
    # Total investment = 180s = 180,000ms. Speedup = 40ms/query. Breakeven = 4500 queries.
    assert breakeven.breakeven_queries == pytest.approx(4500.0)

    # Negative speedup yields infinite break-even
    breakeven_slow = compute_initializer_breakeven(
        teacher_cost_s=120.0,
        train_cost_s=60.0,
        classical_time_ms=20.0,
        learned_time_ms=30.0,
    )
    assert breakeven_slow.speedup_per_query_ms == -10.0
    assert np.isinf(breakeven_slow.breakeven_queries)


def test_green_learning_curve_early_stopping_on_inconclusive_curves() -> None:
    """Inconclusive or negative learning curves automatically halt scale-up without hiding results."""
    # Stagnating/increasing validation loss (overfitting / negative curve)
    bad_curve = [1.0, 0.8, 0.81, 0.82, 0.83]
    status = monitor_learning_curve(bad_curve, patience=3, min_delta=1e-3)
    assert status == EarlyStoppingStatus.INCONCLUSIVE

    # Steadily decreasing curve continues
    good_curve = [1.0, 0.7, 0.5, 0.35]
    status_good = monitor_learning_curve(good_curve, patience=3, min_delta=1e-3)
    assert status_good == EarlyStoppingStatus.CONTINUE


def test_green_experiment_bounds_and_comparison_harness() -> None:
    """Comparison harness benchmarks retrieval, compact learned proposals, and classical warm starts."""
    provider = AnalyticPendulumProvider()

    bounds = ExperimentBounds(
        max_teacher_episodes=50,
        max_teacher_compute_s=300.0,
        max_training_epochs=20,
        max_training_compute_s=150.0,
    )

    inputs = [
        InitializerInput(
            observation_trajectory=np.zeros((6, 2)),
            observation_mask=np.ones(2),
            sample_times_s=np.linspace(0.0, 0.05, 6),
            model_id="pendulum_v1",
            model_hash="hash_pendulum",
            player_id=f"p_{i}",
            session_id=f"s_{i}",
            subject_dimensions={"height": 1.70 + 0.02 * i},
        )
        for i in range(3)
    ]

    report = compare_matching_initializers(inputs, provider, bounds=bounds)
    assert isinstance(report, InitializerComparisonReport)
    assert len(report.results) == 3
    strategies = {r.strategy for r in report.results}
    assert InitializerSource.RETRIEVAL in strategies
    assert InitializerSource.LEARNED_PROPOSAL in strategies
    assert InitializerSource.CLASSICAL_WARM_START in strategies
    assert report.breakeven.teacher_generation_cost_s >= 0.0

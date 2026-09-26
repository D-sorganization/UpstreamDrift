"""NM-07 (#10622): Forward surrogates comparison and physics-structured alternatives."""

from __future__ import annotations

import pytest

from src.shared.python.neural_motion.surrogates import (
    DEFAULT_EVALUATORS,
    SURROGATE_COMPARISON_SCHEMA,
    PhysicsStructuredSurrogate,
    SurrogateAblationResult,
    SurrogateCandidateKind,
    SurrogateComparisonConfig,
    SurrogateComparisonReport,
    compare_surrogates_and_alternatives,
    surrogate_passes_gates,
)

pytestmark = pytest.mark.unit


def test_surrogate_comparison_config_dbc() -> None:
    """DbC: invalid trust radius, negative tolerances, or bad model_id raise ValueError."""
    with pytest.raises(ValueError, match="model_id must be non-empty"):
        SurrogateComparisonConfig(model_id="")

    with pytest.raises(ValueError, match="trust_radius must be positive"):
        SurrogateComparisonConfig(model_id="driven_double_pendulum", trust_radius=-0.1)

    with pytest.raises(ValueError, match="stopping_criterion_tol must be positive"):
        SurrogateComparisonConfig(
            model_id="driven_double_pendulum", stopping_criterion_tol=0.0
        )

    cfg = SurrogateComparisonConfig(model_id="driven_double_pendulum")
    assert cfg.model_id == "driven_double_pendulum"
    assert cfg.trust_radius > 0.0


def test_physics_structured_surrogate_prior_and_residual() -> None:
    """PhysicsStructuredSurrogate combines analytical rigid prior with neural residual."""
    import numpy as np

    model = PhysicsStructuredSurrogate(n_dof=2, n_coeffs=7)
    coeffs = np.zeros(14)
    timegrid = np.linspace(0.0, 0.3, 30)

    # Rollout produces trajectory shape (T, dof)
    traj = model.forward_trajectory(coeffs, timegrid)
    assert traj.shape == (30, 2)
    assert np.all(np.isfinite(traj))


def test_compare_surrogates_config_reaches_evaluators() -> None:
    """Config object actually reaches injected evaluators (#10960 P0-3)."""
    received_configs: list[SurrogateComparisonConfig] = []

    def mock_evaluator(config: SurrogateComparisonConfig) -> SurrogateAblationResult:
        received_configs.append(config)
        return SurrogateAblationResult(
            candidate_kind=SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value,
            sample_efficiency=0.9,
            accepted_query_latency_s=0.04,
            clubhead_rmse_m=0.015,
            butt_rmse_m=0.010,
            orientation_rmse_rad=0.02,
            trust_region_rejected=False,
            gradient_fidelity_rejected=False,
            contact_boundary_failure=False,
            converged=True,
            n_evaluations=10,
        )

    cfg = SurrogateComparisonConfig(
        model_id="pendulum_eval_probe",
        trust_radius=1.8,
        max_accepted_query_latency_s=0.5,
    )

    report = compare_surrogates_and_alternatives(
        cfg,
        evaluators={
            SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value: mock_evaluator
        },
    )

    assert len(received_configs) == 1
    assert received_configs[0] is cfg
    assert received_configs[0].model_id == "pendulum_eval_probe"
    assert received_configs[0].trust_radius == 1.8
    assert (
        report.selected_approach
        == SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value
    )


def test_compare_surrogates_selection_changes_with_measured_metrics() -> None:
    """Injected evaluators with different metrics change selected_approach (argmin) without hardcoded winner (#10960 P0-3)."""
    cfg = SurrogateComparisonConfig(model_id="driven_double_pendulum")

    def make_result(
        kind: str,
        sample_eff: float,
        latency: float,
        ch_rmse: float,
        converged: bool = True,
        rejected: bool = False,
    ) -> SurrogateAblationResult:
        return SurrogateAblationResult(
            candidate_kind=kind,
            sample_efficiency=sample_eff,
            accepted_query_latency_s=latency,
            clubhead_rmse_m=ch_rmse,
            butt_rmse_m=ch_rmse * 0.5,
            orientation_rmse_rad=ch_rmse * 1.5,
            trust_region_rejected=rejected,
            gradient_fidelity_rejected=False,
            contact_boundary_failure=False,
            converged=converged,
            n_evaluations=20,
        )

    # Trial 1: FORWARD_SURROGATE_POLISH has significantly better metrics
    evaluators_trial1 = {
        SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value: lambda c: make_result(
            SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value,
            sample_eff=0.70,
            latency=0.20,
            ch_rmse=0.035,
        ),
        SurrogateCandidateKind.FORWARD_SURROGATE_POLISH.value: lambda c: make_result(
            SurrogateCandidateKind.FORWARD_SURROGATE_POLISH.value,
            sample_eff=0.95,
            latency=0.02,
            ch_rmse=0.005,
        ),
    }

    report1 = compare_surrogates_and_alternatives(cfg, evaluators=evaluators_trial1)
    assert (
        report1.selected_approach
        == SurrogateCandidateKind.FORWARD_SURROGATE_POLISH.value
    )
    assert "forward_surrogate_polish" in report1.selection_rationale
    assert "0.0200" in report1.selection_rationale

    # Trial 2: Change metrics so PHYSICS_STRUCTURED_RESIDUAL has lower cost
    evaluators_trial2 = {
        SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value: lambda c: make_result(
            SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value,
            sample_eff=0.98,
            latency=0.015,
            ch_rmse=0.003,
        ),
        SurrogateCandidateKind.FORWARD_SURROGATE_POLISH.value: lambda c: make_result(
            SurrogateCandidateKind.FORWARD_SURROGATE_POLISH.value,
            sample_eff=0.60,
            latency=0.30,
            ch_rmse=0.040,
        ),
    }

    report2 = compare_surrogates_and_alternatives(cfg, evaluators=evaluators_trial2)
    assert (
        report2.selected_approach
        == SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value
    )
    assert "physics_structured_residual" in report2.selection_rationale
    assert "0.0150" in report2.selection_rationale


def test_compare_surrogates_yields_no_selection_when_unqualified() -> None:
    """When all candidates are rejected or non-converged, yields no selection (#10960 P0-3)."""
    cfg = SurrogateComparisonConfig(model_id="driven_double_pendulum")

    rejected_evaluators = {
        SurrogateCandidateKind.FORWARD_SURROGATE_INVERT.value: lambda c: (
            SurrogateAblationResult(
                candidate_kind=SurrogateCandidateKind.FORWARD_SURROGATE_INVERT.value,
                sample_efficiency=0.5,
                accepted_query_latency_s=0.2,
                clubhead_rmse_m=0.02,
                butt_rmse_m=0.01,
                orientation_rmse_rad=0.03,
                trust_region_rejected=True,
                gradient_fidelity_rejected=False,
                contact_boundary_failure=False,
                converged=True,
                n_evaluations=10,
                rejection_reason="Trust region violation",
            )
        ),
        SurrogateCandidateKind.DIFFUSION_FALLBACK.value: lambda c: (
            SurrogateAblationResult(
                candidate_kind=SurrogateCandidateKind.DIFFUSION_FALLBACK.value,
                sample_efficiency=0.2,
                accepted_query_latency_s=0.9,
                clubhead_rmse_m=0.08,
                butt_rmse_m=0.04,
                orientation_rmse_rad=0.12,
                trust_region_rejected=False,
                gradient_fidelity_rejected=False,
                contact_boundary_failure=False,
                converged=False,
                n_evaluations=100,
                rejection_reason="Did not converge",
            )
        ),
    }

    report = compare_surrogates_and_alternatives(cfg, evaluators=rejected_evaluators)
    assert report.selected_approach is None
    assert "No candidate satisfied" in report.selection_rationale


def test_compare_surrogates_and_alternatives_injected_bounded_ablation() -> None:
    """Runs bounded ablation across forward surrogates, hybrid polish, physics-structured residual, and proposals with injected measured results."""
    cfg = SurrogateComparisonConfig(
        model_id="driven_double_pendulum",
        trust_radius=1.5,
        max_iterations=50,
        stopping_criterion_tol=1e-3,
    )

    injected_evaluators = {
        SurrogateCandidateKind.FORWARD_SURROGATE_INVERT.value: lambda c: (
            SurrogateAblationResult(
                candidate_kind=SurrogateCandidateKind.FORWARD_SURROGATE_INVERT.value,
                sample_efficiency=0.76,
                accepted_query_latency_s=0.082,
                clubhead_rmse_m=0.028,
                butt_rmse_m=0.015,
                orientation_rmse_rad=0.042,
                trust_region_rejected=False,
                gradient_fidelity_rejected=False,
                contact_boundary_failure=False,
                converged=True,
                n_evaluations=45,
            )
        ),
        SurrogateCandidateKind.FORWARD_SURROGATE_POLISH.value: lambda c: (
            SurrogateAblationResult(
                candidate_kind=SurrogateCandidateKind.FORWARD_SURROGATE_POLISH.value,
                sample_efficiency=0.88,
                accepted_query_latency_s=0.145,
                clubhead_rmse_m=0.011,
                butt_rmse_m=0.007,
                orientation_rmse_rad=0.018,
                trust_region_rejected=False,
                gradient_fidelity_rejected=False,
                contact_boundary_failure=False,
                converged=True,
                n_evaluations=32,
            )
        ),
        SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value: lambda c: (
            SurrogateAblationResult(
                candidate_kind=SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value,
                sample_efficiency=0.92,
                accepted_query_latency_s=0.038,
                clubhead_rmse_m=0.019,
                butt_rmse_m=0.010,
                orientation_rmse_rad=0.024,
                trust_region_rejected=False,
                gradient_fidelity_rejected=False,
                contact_boundary_failure=False,
                converged=True,
                n_evaluations=24,
            )
        ),
        SurrogateCandidateKind.MASKED_PROPOSAL.value: lambda c: SurrogateAblationResult(
            candidate_kind=SurrogateCandidateKind.MASKED_PROPOSAL.value,
            sample_efficiency=0.80,
            accepted_query_latency_s=0.012,
            clubhead_rmse_m=0.034,
            butt_rmse_m=0.021,
            orientation_rmse_rad=0.058,
            trust_region_rejected=False,
            gradient_fidelity_rejected=False,
            contact_boundary_failure=False,
            converged=True,
            n_evaluations=1,
        ),
        SurrogateCandidateKind.DIFFUSION_FALLBACK.value: lambda c: (
            SurrogateAblationResult(
                candidate_kind=SurrogateCandidateKind.DIFFUSION_FALLBACK.value,
                sample_efficiency=0.28,
                accepted_query_latency_s=0.820,
                clubhead_rmse_m=0.048,
                butt_rmse_m=0.032,
                orientation_rmse_rad=0.075,
                trust_region_rejected=True,
                gradient_fidelity_rejected=False,
                contact_boundary_failure=False,
                converged=False,
                n_evaluations=120,
                rejection_reason="Excessive iteration latency",
            )
        ),
    }

    report = compare_surrogates_and_alternatives(cfg, evaluators=injected_evaluators)

    assert isinstance(report, SurrogateComparisonReport)
    assert report.schema_version == SURROGATE_COMPARISON_SCHEMA
    assert report.model_id == "driven_double_pendulum"

    # All candidate kinds evaluated
    assert SurrogateCandidateKind.FORWARD_SURROGATE_INVERT.value in report.candidates
    assert SurrogateCandidateKind.FORWARD_SURROGATE_POLISH.value in report.candidates
    assert SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value in report.candidates
    assert SurrogateCandidateKind.MASKED_PROPOSAL.value in report.candidates
    assert SurrogateCandidateKind.DIFFUSION_FALLBACK.value in report.candidates

    # Diffusion fallback is rejected due to excessive accepted-query latency (multi-iteration)
    diffusion_res = report.candidates[SurrogateCandidateKind.DIFFUSION_FALLBACK.value]
    assert (
        diffusion_res.converged is False or diffusion_res.accepted_query_latency_s > 0.5
    )
    assert diffusion_res.rejection_reason is not None

    # Selected approach is determined by argmin of measured metrics
    assert report.selected_approach in [
        SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value,
        SurrogateCandidateKind.FORWARD_SURROGATE_POLISH.value,
        SurrogateCandidateKind.FORWARD_SURROGATE_INVERT.value,
    ]
    assert len(report.selection_rationale) > 10


def _measured(
    kind: str, *, ch: float, latency: float, eff: float
) -> SurrogateAblationResult:
    return SurrogateAblationResult(
        candidate_kind=kind,
        sample_efficiency=eff,
        accepted_query_latency_s=latency,
        clubhead_rmse_m=ch,
        butt_rmse_m=ch * 0.5,
        orientation_rmse_rad=ch * 1.5,
        trust_region_rejected=False,
        gradient_fidelity_rejected=False,
        contact_boundary_failure=False,
        converged=True,
        n_evaluations=20,
    )


def test_selection_never_trades_metres_against_seconds() -> None:
    """Accuracy decides; a faster, less accurate candidate cannot win on a summed score (#10960 P0-3)."""
    cfg = SurrogateComparisonConfig(model_id="driven_double_pendulum")
    accurate = SurrogateCandidateKind.FORWARD_SURROGATE_POLISH.value
    fast = SurrogateCandidateKind.MASKED_PROPOSAL.value
    evaluators = {
        accurate: lambda c: _measured(accurate, ch=0.010, latency=0.90, eff=0.50),
        fast: lambda c: _measured(fast, ch=0.030, latency=0.01, eff=0.95),
    }

    report = compare_surrogates_and_alternatives(cfg, evaluators=evaluators)

    assert report.selected_approach == accurate


def test_candidate_over_the_latency_bound_is_gated_out() -> None:
    cfg = SurrogateComparisonConfig(model_id="driven_double_pendulum")
    slow = SurrogateCandidateKind.FORWARD_SURROGATE_POLISH.value
    ok = SurrogateCandidateKind.MASKED_PROPOSAL.value
    evaluators = {
        slow: lambda c: _measured(slow, ch=0.001, latency=5.0, eff=0.9),
        ok: lambda c: _measured(ok, ch=0.030, latency=0.01, eff=0.9),
    }

    report = compare_surrogates_and_alternatives(cfg, evaluators=evaluators)

    assert not surrogate_passes_gates(report.candidates[slow], cfg)
    assert report.selected_approach == ok


def test_default_evaluators_cover_every_candidate_kind_and_fail_closed() -> None:
    cfg = SurrogateComparisonConfig(model_id="driven_double_pendulum")
    assert set(DEFAULT_EVALUATORS) == {k.value for k in SurrogateCandidateKind}
    for evaluate in DEFAULT_EVALUATORS.values():
        with pytest.raises(NotImplementedError, match="#10960 P0-3"):
            evaluate(cfg)

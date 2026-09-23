"""NM-07 (#10622): Forward surrogates comparison and physics-structured alternatives."""

from __future__ import annotations

import pytest

from src.shared.python.neural_motion.surrogates import (
    SURROGATE_COMPARISON_SCHEMA,
    PhysicsStructuredSurrogate,
    SurrogateAblationResult,
    SurrogateCandidateKind,
    SurrogateComparisonConfig,
    SurrogateComparisonReport,
    compare_surrogates_and_alternatives,
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


def test_compare_surrogates_and_alternatives_bounded_ablation() -> None:
    """Runs bounded ablation across forward surrogates, hybrid polish, physics-structured residual, and proposals."""
    cfg = SurrogateComparisonConfig(
        model_id="driven_double_pendulum",
        trust_radius=1.5,
        max_iterations=50,
        stopping_criterion_tol=1e-3,
    )

    report = compare_surrogates_and_alternatives(cfg)

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

    # Smallest useful approach selected with evidence
    assert report.selected_approach in [
        SurrogateCandidateKind.PHYSICS_STRUCTURED_RESIDUAL.value,
        SurrogateCandidateKind.FORWARD_SURROGATE_POLISH.value,
        SurrogateCandidateKind.FORWARD_SURROGATE_INVERT.value,
    ]
    assert len(report.selection_rationale) > 10

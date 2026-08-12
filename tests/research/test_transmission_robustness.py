"""Contracts for the adversarial transmission and robustness study."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.transmission_robustness import (
    OutcomeLinearization,
    PathwayBudget,
    PerturbationEnsemble,
    compute_pathway_budget,
    finite_difference_outcome_jacobian,
    nondominated_indices,
    perturbation_summary,
    task_variance_partition,
)


def test_pathway_budget_closes_and_preserves_signed_components() -> None:
    time = np.linspace(0.0, 1.0, 11)
    powers = {
        "proximal_actuation": np.full_like(time, 2.0),
        "constraint_transport": np.full_like(time, -0.5),
        "direct_distal_moment": np.full_like(time, 1.0),
        "elastic_release": np.linspace(-1.0, 1.0, time.size),
        "gravity": np.full_like(time, 0.25),
        "dissipation": np.full_like(time, -0.75),
    }
    energy_change = sum(np.trapezoid(value, x=time) for value in powers.values())

    budget = compute_pathway_budget(time, powers, energy_change_j=energy_change)

    assert isinstance(budget, PathwayBudget)
    assert budget.work_j["constraint_transport"] == pytest.approx(-0.5)
    assert budget.work_j["elastic_release"] == pytest.approx(0.0, abs=1e-14)
    assert budget.closure_residual_j == pytest.approx(0.0, abs=1e-14)
    assert budget.cancellation_index > 0.0


def test_pathway_budget_rejects_a_hidden_energy_source() -> None:
    time = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="closure"):
        compute_pathway_budget(
            time,
            {"declared": np.ones(2)},
            energy_change_j=2.0,
            closure_tolerance_j=1e-12,
        )


def test_paired_perturbation_summary_exposes_lower_tail_and_amplification() -> None:
    perturbations = np.array([[-1.0], [0.0], [1.0]])
    baseline = np.array([[10.0, 2.0], [11.0, 1.0], [12.0, 0.0]])
    candidate = np.array([[10.5, 1.0], [11.5, 1.0], [12.5, 1.0]])
    ensemble = PerturbationEnsemble(
        perturbations=perturbations,
        baseline_outcomes=baseline,
        candidate_outcomes=candidate,
        outcome_names=("delivery_speed_m_s", "face_path_error_deg"),
    )

    summary = perturbation_summary(ensemble)

    assert summary["delivery_speed_m_s"]["paired_mean_delta"] == pytest.approx(0.5)
    assert (
        summary["delivery_speed_m_s"]["candidate_q10"]
        > summary["delivery_speed_m_s"]["baseline_q10"]
    )
    assert summary["face_path_error_deg"]["candidate_std"] == pytest.approx(0.0)


def test_local_task_variance_partition_distinguishes_null_and_task_directions() -> None:
    jacobian = np.array([[1.0, 1.0]])
    samples = np.array([[1.0, -1.0], [-1.0, 1.0], [2.0, -2.0], [-2.0, 2.0]])

    result = task_variance_partition(jacobian, samples)

    assert result.task_rank == 1
    assert result.nullity == 1
    assert result.null_variance > 0.0
    assert result.task_relevant_variance == pytest.approx(0.0, abs=1e-14)
    assert result.synergy_index == pytest.approx(1.0)


def test_finite_difference_jacobian_is_central_and_scale_explicit() -> None:
    linearization = finite_difference_outcome_jacobian(
        lambda x: np.array([x[0] + 2.0 * x[1], x[0] ** 2]),
        center=np.array([3.0, 4.0]),
        steps=np.array([1e-4, 2e-4]),
        input_names=("a", "b"),
        outcome_names=("sum", "square"),
    )

    assert isinstance(linearization, OutcomeLinearization)
    np.testing.assert_allclose(
        linearization.jacobian, [[1.0, 2.0], [6.0, 0.0]], atol=1e-8
    )


def test_nondominated_set_retains_speed_variability_tradeoff() -> None:
    # All objectives are minimized: negative speed, dispersion, and load.
    objectives = np.array(
        [
            [-10.0, 2.0, 5.0],
            [-9.0, 1.0, 4.0],
            [-8.0, 3.0, 6.0],
        ]
    )
    assert nondominated_indices(objectives) == (0, 1)


pytestmark = pytest.mark.scientific

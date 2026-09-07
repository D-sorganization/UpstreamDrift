"""Out-of-sample evaluation and promotion ledger for participant twin ensembles.

Posterior contraction and out-of-sample prediction are computed and reported
separately, both held-out protocols run against a population baseline and an
uncalibrated baseline, and every null or nontransportable outcome is retained
rather than filtered. The ledger fails closed: no promotion path in this module
can declare a human, anatomical, equipment, or coaching conclusion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .participant_twin_calibration import (
    ALIASED_PARAMETERS,
    CALIBRATION_TRAJECTORIES,
    DISCREPANCY_PRIOR_SD,
    POPULATION_SD_FRACTION,
    POSTERIOR_PREDICTIVE_COVERAGE_MINIMUM,
    PRIOR_PREDICTIVE_COVERAGE_BAND,
    PRIOR_SD,
    SMALL_STRATUM_CELL,
    TWIN_PARAMETERS,
    Benchmark,
    FloatArray,
    IdentifiabilityScreen,
    PopulationPosterior,
    _condition,
    _joint_prior_for_new_participant,
    _operator,
    _participant_trajectories,
    _predictive_metrics,
    build_benchmark,
    build_synthetic_cohort,
    feature_names,
    fit_population,
    screen_identifiability,
)
from .participant_twin_provenance import (
    STRATIFICATION_VOCABULARY,
    HoldoutBarrier,
    build_holdout_barrier,
    public_facade,
    validate_cohort,
)

_ROUNDING = 9


def _round(value: float) -> float:
    return float(np.round(float(value), _ROUNDING))


def _round_mapping(mapping: dict[str, float]) -> dict[str, float]:
    return {key: _round(value) for key, value in mapping.items()}


def _marginal_prior(channel_count: int) -> tuple[FloatArray, FloatArray]:
    width = len(TWIN_PARAMETERS)
    mean = np.zeros(width + channel_count)
    covariance = np.zeros((width + channel_count, width + channel_count))
    prior = np.asarray(PRIOR_SD, dtype=float)
    covariance[0:width, 0:width] = np.diag(
        prior**2 + (POPULATION_SD_FRACTION * prior) ** 2
    )
    covariance[width:, width:] = np.eye(channel_count) * DISCREPANCY_PRIOR_SD**2
    return mean, covariance


def _joint_block(
    posterior: PopulationPosterior, pseudonym: str
) -> tuple[FloatArray, FloatArray]:
    indices = np.r_[
        posterior.participant_slice(pseudonym), posterior.discrepancy_slice()
    ]
    return posterior.mean[indices], posterior.covariance[np.ix_(indices, indices)]


def _population_block(
    posterior: PopulationPosterior,
) -> tuple[FloatArray, FloatArray]:
    indices = np.r_[posterior.population_slice(), posterior.discrepancy_slice()]
    mean = posterior.mean[indices].copy()
    covariance = posterior.covariance[np.ix_(indices, indices)].copy()
    width = len(TWIN_PARAMETERS)
    covariance[0:width, 0:width] += np.diag(
        (POPULATION_SD_FRACTION * np.asarray(PRIOR_SD)) ** 2
    )
    return mean, covariance


def _calibration_observations(
    benchmark: Benchmark, cohort: dict[str, Any], pseudonyms: tuple[str, ...]
) -> list[FloatArray]:
    return [
        benchmark.observations[trajectory]
        for pseudonym in pseudonyms
        for trajectory in _participant_trajectories(cohort, pseudonym, "calibration")
    ]


def prior_predictive_check(
    benchmark: Benchmark, cohort: dict[str, Any]
) -> dict[str, float | str]:
    """Check the declared prior against the calibration data before fitting."""
    summary = validate_cohort(cohort)
    observations = _calibration_observations(
        benchmark, cohort, tuple(summary["training_participants"])
    )
    mean, covariance = _marginal_prior(len(benchmark.channels))
    metrics = _predictive_metrics(
        observations, mean, covariance, _operator(benchmark), benchmark.noise_sd
    )
    lower, upper = PRIOR_PREDICTIVE_COVERAGE_BAND
    coverage = metrics["interval_coverage_90"]
    return {
        "normalized_rmse": _round(metrics["normalized_rmse"]),
        "interval_coverage_90": _round(coverage),
        "observation_count": int(metrics["observation_count"]),
        "status": "passed" if lower <= coverage <= upper else "failed",
    }


def posterior_predictive_check(
    benchmark: Benchmark, cohort: dict[str, Any], posterior: PopulationPosterior
) -> dict[str, float | str]:
    """Check the fitted posterior against the data it was calibrated on."""
    operator = _operator(benchmark)
    residual_squares: list[float] = []
    covered: list[float] = []
    count = 0
    for pseudonym in posterior.training_participants:
        mean, covariance = _joint_block(posterior, pseudonym)
        observations = [
            benchmark.observations[trajectory]
            for trajectory in _participant_trajectories(
                cohort, pseudonym, "calibration"
            )
        ]
        metrics = _predictive_metrics(
            observations, mean, covariance, operator, benchmark.noise_sd
        )
        residual_squares.append(metrics["normalized_rmse"] ** 2 * len(observations))
        covered.append(metrics["interval_coverage_90"] * len(observations))
        count += len(observations)
    coverage = float(np.sum(covered) / count)
    return {
        "normalized_rmse": _round(float(np.sqrt(np.sum(residual_squares) / count))),
        "interval_coverage_90": _round(coverage),
        "observation_count": count,
        "status": (
            "passed" if coverage >= POSTERIOR_PREDICTIVE_COVERAGE_MINIMUM else "failed"
        ),
    }


def posterior_contraction(posterior: PopulationPosterior) -> dict[str, Any]:
    """Report contraction separately from prediction, per parameter and block."""
    prior = np.asarray(PRIOR_SD, dtype=float)
    marginal = np.sqrt(prior**2 + (POPULATION_SD_FRACTION * prior) ** 2)
    population_sd = np.sqrt(np.diag(posterior.covariance)[posterior.population_slice()])
    participant_sd = np.vstack(
        [
            np.sqrt(np.diag(posterior.covariance)[posterior.participant_slice(name)])
            for name in posterior.training_participants
        ]
    )
    discrepancy_sd = np.sqrt(
        np.diag(posterior.covariance)[posterior.discrepancy_slice()]
    )
    return {
        "population_mean": _round_mapping(
            dict(zip(TWIN_PARAMETERS, 1.0 - population_sd / prior, strict=True))
        ),
        "participant_twin": _round_mapping(
            dict(
                zip(
                    TWIN_PARAMETERS,
                    1.0 - participant_sd.mean(axis=0) / marginal,
                    strict=True,
                )
            )
        ),
        "discrepancy_term": [
            _round(value) for value in (1.0 - discrepancy_sd / DISCREPANCY_PRIOR_SD)
        ],
        "reporting_boundary": (
            "contraction measures what the design and prior constrain; it is "
            "not evidence of out-of-sample predictive skill"
        ),
    }


def _evaluate_participant(
    benchmark: Benchmark,
    cohort: dict[str, Any],
    mean: FloatArray,
    covariance: FloatArray,
    population: tuple[FloatArray, FloatArray],
    pseudonym: str,
) -> dict[str, dict[str, float]]:
    operator = _operator(benchmark)
    observations = [
        benchmark.observations[trajectory]
        for trajectory in _participant_trajectories(cohort, pseudonym, "evaluation")
    ]
    prior_mean, prior_covariance = _marginal_prior(len(benchmark.channels))
    return {
        "calibrated_twin": _predictive_metrics(
            observations, mean, covariance, operator, benchmark.noise_sd
        ),
        "population_baseline": _predictive_metrics(
            observations, population[0], population[1], operator, benchmark.noise_sd
        ),
        "uncalibrated_baseline": _predictive_metrics(
            observations, prior_mean, prior_covariance, operator, benchmark.noise_sd
        ),
    }


def _pool(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    pooled: dict[str, dict[str, float]] = {}
    for arm in ("calibrated_twin", "population_baseline", "uncalibrated_baseline"):
        total = sum(row[arm]["observation_count"] for row in rows)
        squares = sum(
            row[arm]["normalized_rmse"] ** 2 * row[arm]["observation_count"]
            for row in rows
        )
        covered = sum(
            row[arm]["interval_coverage_90"] * row[arm]["observation_count"]
            for row in rows
        )
        pooled[arm] = {
            "normalized_rmse": _round(float(np.sqrt(squares / total))),
            "interval_coverage_90": _round(float(covered / total)),
            "observation_count": int(total),
        }
    return pooled


def evaluate_holdouts(
    benchmark: Benchmark,
    cohort: dict[str, Any],
    posterior: PopulationPosterior,
    barrier: HoldoutBarrier,
) -> dict[str, Any]:
    """Evaluate trajectory-held-out and participant-held-out prediction.

    Held-out participants are calibrated only on their own calibration-role
    trajectories, read through ``barrier``, and conditioned on the population
    posterior fitted without them, so no evaluation outcome can enter a
    calibration path.
    """
    summary = validate_cohort(cohort)
    population = _population_block(posterior)
    operator = _operator(benchmark)
    trajectory_rows: list[dict[str, dict[str, float]]] = []
    for pseudonym in posterior.training_participants:
        mean, covariance = _joint_block(posterior, pseudonym)
        trajectory_rows.append(
            _evaluate_participant(
                benchmark, cohort, mean, covariance, population, pseudonym
            )
        )
    participant_rows: list[dict[str, dict[str, float]]] = []
    per_participant_rmse: dict[str, float] = {}
    prior_mean, prior_covariance = _joint_prior_for_new_participant(posterior)
    for pseudonym in summary["participant_held_out"]:
        observations = [
            barrier.calibration_outcome(trajectory, benchmark.observations)
            for trajectory in _participant_trajectories(
                cohort, pseudonym, "calibration"
            )
        ]
        mean, covariance = _condition(
            prior_mean, prior_covariance, operator, observations, benchmark.noise_sd
        )
        row = _evaluate_participant(
            benchmark, cohort, mean, covariance, population, pseudonym
        )
        participant_rows.append(row)
        per_participant_rmse[pseudonym] = row["calibrated_twin"]["normalized_rmse"]
    return {
        "trajectory_held_out": _pool(trajectory_rows),
        "participant_held_out": _pool(participant_rows),
        "participant_held_out_normalized_rmse": _round_mapping(per_participant_rmse),
        "protocol": {
            "trajectory_held_out": (
                "training participants; the twin sees only calibration-role "
                "trajectories of the same participant"
            ),
            "participant_held_out": (
                "participants excluded from the population fit; the twin sees "
                "only that participant's calibration-role trajectories"
            ),
        },
    }


def transport_audit(
    cohort: dict[str, Any], per_participant_rmse: dict[str, float]
) -> dict[str, Any]:
    """Audit stratification and transport across every declared stratum."""
    summary = validate_cohort(cohort)
    training = set(summary["training_participants"])
    strata = {
        row["participant_pseudonym"]: row["strata"] for row in cohort["participants"]
    }
    audit: dict[str, dict[str, Any]] = {}
    nontransportable: list[str] = []
    for field_name in STRATIFICATION_VOCABULARY:
        levels: dict[str, dict[str, Any]] = {}
        for pseudonym, values in strata.items():
            level = values[field_name]
            cell = levels.setdefault(
                level,
                {
                    "training_participants": 0,
                    "evaluated_participants": 0,
                    "normalized_rmse": None,
                    "transport_status": "represented",
                },
            )
            if pseudonym in training:
                cell["training_participants"] += 1
            if pseudonym in per_participant_rmse:
                cell["evaluated_participants"] += 1
        for level, cell in levels.items():
            errors = [
                per_participant_rmse[pseudonym]
                for pseudonym, values in strata.items()
                if values[field_name] == level and pseudonym in per_participant_rmse
            ]
            if errors:
                cell["normalized_rmse"] = _round(float(np.mean(errors)))
            if cell["training_participants"] == 0:
                cell["transport_status"] = "nontransportable_unrepresented_in_training"
                nontransportable.append(f"{field_name}={level}")
            elif cell["training_participants"] < SMALL_STRATUM_CELL:
                cell["transport_status"] = "represented_small_cell"
        audit[field_name] = dict(sorted(levels.items()))
    return {
        "strata": audit,
        "nontransportable_cells": sorted(nontransportable),
        "retention_policy": (
            "nontransportable and small-cell results are retained, never "
            "dropped, and never averaged into a transportable claim"
        ),
    }


def parameter_recovery(
    benchmark: Benchmark, posterior: PopulationPosterior, screen: IdentifiabilityScreen
) -> dict[str, Any]:
    """Report recovery only for parameters the pre-fit screen declared estimable."""
    estimable = screen.estimable_parameters
    errors: dict[str, list[float]] = {name: [] for name in estimable}
    for pseudonym in posterior.training_participants:
        estimate = posterior.mean[posterior.participant_slice(pseudonym)]
        truth = benchmark.participant_truth[pseudonym]
        for name in estimable:
            index = TWIN_PARAMETERS.index(name)
            errors[name].append(float(estimate[index] - truth[index]))
    return {
        "estimable_parameters": list(estimable),
        "withheld_parameters": [
            name for name in TWIN_PARAMETERS if name not in estimable
        ],
        "absolute_error_rms": _round_mapping(
            {
                name: float(np.sqrt(np.mean(np.square(values))))
                for name, values in errors.items()
            }
        ),
        "reporting_boundary": (
            "parameters excluded by the pre-fit identifiability screen are "
            "never reported as recovered, even though the prior keeps the "
            "posterior proper"
        ),
    }


def _gate(condition: bool, satisfied: str, failed: str) -> str:
    return satisfied if condition else failed


def build_evaluation_ledger(
    cohort: dict[str, Any],
    screen: IdentifiabilityScreen,
    barrier_held_out_reads: int,
    prior_check: dict[str, Any],
    posterior_check: dict[str, Any],
    contraction: dict[str, Any],
    holdouts: dict[str, Any],
    transport: dict[str, Any],
    *,
    include_discrepancy: bool,
) -> dict[str, Any]:
    """Assemble the fail-closed promotion ledger for this workflow run."""
    summary = validate_cohort(cohort)
    trajectory = holdouts["trajectory_held_out"]
    participant = holdouts["participant_held_out"]
    facade = public_facade(cohort)
    gates = {
        "private_governed_authority_contract": _gate(
            summary["human_calibration_authority"] == "available",
            "satisfied",
            "blocked_no_governed_participant_data",
        ),
        "public_facade_contract": _gate(
            "participants" not in facade and "stratum_counts" in facade,
            "satisfied",
            "failed",
        ),
        "calibration_never_reads_held_out_outcomes": _gate(
            barrier_held_out_reads == 0, "satisfied", "failed"
        ),
        "identifiability_screened_before_fitting": _gate(
            screen.structural_rank < screen.parameter_count
            and bool(screen.aliased_directions),
            "satisfied_and_non_identifiable_directions_retained",
            "satisfied",
        ),
        "prior_predictive_check": prior_check["status"],
        "posterior_predictive_check": posterior_check["status"],
        "explicit_discrepancy_term": _gate(
            include_discrepancy, "declared_and_estimated", "omitted"
        ),
        "contraction_and_prediction_reported_separately": _gate(
            "population_mean" in contraction
            and "normalized_rmse" in trajectory["calibrated_twin"],
            "satisfied",
            "failed",
        ),
        "participant_held_out_beats_population_baseline": _gate(
            participant["calibrated_twin"]["normalized_rmse"]
            < participant["population_baseline"]["normalized_rmse"],
            "satisfied",
            "not_satisfied",
        ),
        "trajectory_held_out_beats_uncalibrated_baseline": _gate(
            trajectory["calibrated_twin"]["normalized_rmse"]
            < trajectory["uncalibrated_baseline"]["normalized_rmse"],
            "satisfied",
            "not_satisfied",
        ),
        "null_and_nontransportable_results_retained": _gate(
            bool(transport["nontransportable_cells"]),
            "satisfied_with_retained_nontransportable_cells",
            "no_nontransportable_cell_in_this_cohort",
        ),
        "no_personalized_recommendation_emitted": "satisfied",
    }
    return {
        "gates": gates,
        "promotion_decision": (
            "synthetic_benchmark_qualified"
            if gates["private_governed_authority_contract"] != "satisfied"
            else "eligible_for_governed_review"
        ),
        "human_promotion": "blocked_no_governed_participant_data",
        "boundaries": {
            "human_digital_twin": "untested",
            "population_ensemble_transport": "declared_strata_only",
            "anatomical_or_muscle_strategy": "not_identified",
            "equipment_effect": "not_identified",
            "coaching_inference": "unsupported",
            "identity_inference": "prohibited_by_provenance_contract",
        },
    }


def run_workflow(
    *, include_activation: bool = True, include_discrepancy: bool = True
) -> dict[str, Any]:
    """Run the full calibration workflow and return the reviewer dashboard.

    Preconditions: none beyond the frozen synthetic cohort.
    Postcondition: the returned record is deterministic, contains no
    participant-level identifier outside the pseudonymous cohort, and never
    reports a promoted human conclusion.
    """
    cohort = build_synthetic_cohort()
    benchmark = build_benchmark(cohort, include_activation=include_activation)
    screen = screen_identifiability(
        benchmark.design,
        benchmark.discrepancy,
        benchmark.noise_sd,
        calibration_trajectory_count=CALIBRATION_TRAJECTORIES,
    )
    barrier = build_holdout_barrier(cohort)
    posterior = fit_population(
        benchmark,
        cohort,
        barrier,
        screen,
        include_discrepancy=include_discrepancy,
    )
    prior_check = prior_predictive_check(benchmark, cohort)
    posterior_check = posterior_predictive_check(benchmark, cohort, posterior)
    contraction = posterior_contraction(posterior)
    holdouts = evaluate_holdouts(benchmark, cohort, posterior, barrier)
    transport = transport_audit(
        cohort, holdouts["participant_held_out_normalized_rmse"]
    )
    recovery = parameter_recovery(benchmark, posterior, screen)
    ledger = build_evaluation_ledger(
        cohort,
        screen,
        barrier.held_out_outcomes_read(),
        prior_check,
        posterior_check,
        contraction,
        holdouts,
        transport,
        include_discrepancy=include_discrepancy,
    )
    return {
        "schema_version": "participant-twin-calibration/v1",
        "analysis_type": "synthetic_hierarchical_participant_twin_calibration",
        "cohort_id": cohort["cohort_id"],
        "channels": [
            {
                "name": channel.name,
                "features": list(channel.features),
                "noise_fraction": channel.noise_fraction,
                "optional": channel.optional,
            }
            for channel in benchmark.channels
        ],
        "feature_names": list(feature_names(benchmark.channels)),
        "design_digest": benchmark.digest(),
        "twin_parameters": list(TWIN_PARAMETERS),
        "prior": {
            "twin_parameter_sd": list(PRIOR_SD),
            "population_sd_fraction": POPULATION_SD_FRACTION,
            "discrepancy_sd": DISCREPANCY_PRIOR_SD,
        },
        "identifiability_screen": {
            "structural_rank": screen.structural_rank,
            "parameter_count": screen.parameter_count,
            "aliased_directions": [list(group) for group in screen.aliased_directions],
            "structurally_identifiable": list(screen.structurally_identifiable),
            "predicted_contraction": _round_mapping(screen.predicted_contraction),
            "practically_identifiable": list(screen.practically_identifiable),
            "estimable_parameters": list(screen.estimable_parameters),
            "declared_alias": list(ALIASED_PARAMETERS),
        },
        "prior_predictive_check": prior_check,
        "posterior_predictive_check": posterior_check,
        "posterior_contraction": contraction,
        "held_out_evaluation": holdouts,
        "parameter_recovery": recovery,
        "transport_audit": transport,
        "public_facade": public_facade(cohort),
        "calibration_access_ledger": {
            "calibration_outcomes_read": len(barrier.access_ledger()),
            "held_out_outcomes_read": barrier.held_out_outcomes_read(),
        },
        "evaluation_ledger": ledger,
        "inference_boundary": cohort["inference_boundary"],
    }


def _ablation_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "posterior_predictive_coverage_90": record["posterior_predictive_check"][
            "interval_coverage_90"
        ],
        "posterior_predictive_normalized_rmse": record["posterior_predictive_check"][
            "normalized_rmse"
        ],
        "participant_held_out_normalized_rmse": record["held_out_evaluation"][
            "participant_held_out"
        ]["calibrated_twin"]["normalized_rmse"],
        "parameter_absolute_error_rms": record["parameter_recovery"][
            "absolute_error_rms"
        ],
    }


def discrepancy_ablation() -> dict[str, Any]:
    """Quantify what omitting the explicit model-discrepancy term costs.

    The omitted systematic error does not disappear. The twin parameters absorb
    it, so the fit still looks acceptable while the estimated parameters move
    away from their generating values.
    """
    return {
        "with_discrepancy_term": _ablation_summary(
            run_workflow(include_discrepancy=True)
        ),
        "without_discrepancy_term": _ablation_summary(
            run_workflow(include_discrepancy=False)
        ),
        "finding": (
            "omitting the declared discrepancy term shifts systematic channel "
            "error into the twin parameters and inflates their error against "
            "the generating values"
        ),
    }


def build_evidence_record() -> dict[str, Any]:
    """Compose the committed evidence bundle, including the discrepancy ablation."""
    record = run_workflow()
    record["discrepancy_ablation"] = discrepancy_ablation()
    return record

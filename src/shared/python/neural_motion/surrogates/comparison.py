"""Ablation comparison of forward surrogates and alternatives (NM-07 #10622)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import datetime

from .types import (
    SURROGATE_COMPARISON_SCHEMA,
    SurrogateAblationResult,
    SurrogateCandidateKind,
    SurrogateComparisonConfig,
    SurrogateComparisonReport,
)

__all__ = [
    "DEFAULT_EVALUATORS",
    "compare_surrogates_and_alternatives",
    "surrogate_passes_gates",
    "surrogate_selection_key",
]


def _unwired_evaluator(
    kind: SurrogateCandidateKind,
) -> Callable[[SurrogateComparisonConfig], SurrogateAblationResult]:
    """Return an evaluator for ``kind`` that fails closed until it is wired (#10960 P0-3).

    No candidate may report metrics it did not measure, so the default
    evaluators validate their input and then raise instead of returning literals.
    """

    def evaluate(config: SurrogateComparisonConfig) -> SurrogateAblationResult:
        if not isinstance(config, SurrogateComparisonConfig):
            raise TypeError(
                f"expected SurrogateComparisonConfig, got {type(config).__name__}"
            )
        raise NotImplementedError(  # tracked: #10960 P0-3
            f"surrogate evaluation for {kind.value!r} is not wired (#10960 P0-3)"
        )

    return evaluate


DEFAULT_EVALUATORS: dict[
    str, Callable[[SurrogateComparisonConfig], SurrogateAblationResult]
] = {kind.value: _unwired_evaluator(kind) for kind in SurrogateCandidateKind}


def surrogate_passes_gates(
    result: SurrogateAblationResult,
    config: SurrogateComparisonConfig,
) -> bool:
    """Return True when ``result`` clears every pre-registered rejection gate.

    A candidate is rejected when it did not converge, tripped the trust-region,
    gradient-fidelity or contact-boundary check, carries a rejection reason, or
    exceeds the configured latency, clubhead-RMSE or orientation-RMSE bound.
    """
    return not (
        not result.converged
        or result.trust_region_rejected
        or result.gradient_fidelity_rejected
        or result.contact_boundary_failure
        or result.rejection_reason is not None
        or result.accepted_query_latency_s > config.max_accepted_query_latency_s
        or result.clubhead_rmse_m > config.max_clubhead_rmse_m
        or result.orientation_rmse_rad > config.max_orientation_rmse_rad
    )


def surrogate_selection_key(
    result: SurrogateAblationResult,
) -> tuple[float, float, float, str]:
    """Order gate-passing candidates by measured metrics (lower is better).

    Lexicographic: clubhead RMSE, then orientation RMSE, then accepted-query
    latency, then candidate kind. Quantities with different units are never
    summed, so no weighting has to be invented.
    """
    return (
        result.clubhead_rmse_m,
        result.orientation_rmse_rad,
        result.accepted_query_latency_s,
        result.candidate_kind,
    )


def compare_surrogates_and_alternatives(
    config: SurrogateComparisonConfig,
    evaluators: (
        Mapping[str, Callable[[SurrogateComparisonConfig], SurrogateAblationResult]]
        | None
    ) = None,
) -> SurrogateComparisonReport:
    """Run bounded ablation comparing forward surrogates and alternatives.

    Parameters
    ----------
    config : SurrogateComparisonConfig
        Pre-registered configuration for forward surrogate and alternative comparison.
    evaluators : Mapping[str, Callable[[SurrogateComparisonConfig], SurrogateAblationResult]] | None
        Optional mapping from candidate kind string to an evaluator callable.
        Defaults to DEFAULT_EVALUATORS, whose members raise NotImplementedError until wired (#10960 P0-3).

    Returns
    -------
    SurrogateComparisonReport
        Evidence bundle containing candidate ablation results and selected approach.

    Raises
    ------
    TypeError
        If config is not a SurrogateComparisonConfig.
    NotImplementedError
        If an unwired default evaluator is invoked (#10960 P0-3).
    """
    if not isinstance(config, SurrogateComparisonConfig):
        raise TypeError(
            f"expected SurrogateComparisonConfig, got {type(config).__name__}"
        )

    evaluator_map = DEFAULT_EVALUATORS if evaluators is None else evaluators

    candidates: dict[str, SurrogateAblationResult] = {}
    for kind, eval_fn in evaluator_map.items():
        candidates[kind] = eval_fn(config)

    qualified = [
        (kind, res)
        for kind, res in candidates.items()
        if surrogate_passes_gates(res, config)
    ]

    if not qualified:
        selected: str | None = None
        rationale = (
            "No candidate satisfied convergence and validation criteria "
            "within configured tolerances."
        )
    else:
        selected, best_res = min(
            qualified, key=lambda item: surrogate_selection_key(item[1])
        )
        rationale = (
            f"Selected approach '{selected}': lowest measured clubhead RMSE among "
            f"gate-passing candidates (ties broken by orientation RMSE, then latency): "
            f"clubhead_rmse_m={best_res.clubhead_rmse_m:.4f}, "
            f"orientation_rmse_rad={best_res.orientation_rmse_rad:.4f}, "
            f"accepted_query_latency_s={best_res.accepted_query_latency_s:.4f}, "
            f"sample_efficiency={best_res.sample_efficiency:.4f}, "
            f"converged={best_res.converged} over {best_res.n_evaluations} evaluations."
        )

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return SurrogateComparisonReport(
        schema_version=SURROGATE_COMPARISON_SCHEMA,
        model_id=config.model_id,
        candidates=candidates,
        selected_approach=selected,
        selection_rationale=rationale,
        timestamp=now,
    )

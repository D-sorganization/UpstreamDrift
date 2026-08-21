"""Canonical evidence-bearing player covariation and population analysis."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.shared.python.launch_monitor.contract_v2 import (
    AnalysisContextV2,
    AvailabilityState,
    AvailabilityV2,
    analysis_lineage_v2,
    metric_units_v2,
    vendor_provenance_v2,
)
from src.shared.python.launch_monitor.player_covariation_core import (
    PairStatistics,
    compute_pair_statistics,
)
from src.shared.python.launch_monitor.player_covariation_types import (
    CovariationPairRankV1,
    CovariationUncertaintyV1,
    PlayerCovariationContractV1,
    PlayerCovariationRequestV1,
    PlayerCovariationResultV1,
    PlayerCovariationScanRequestV1,
    PlayerCovariationScanResultV1,
)
from src.shared.python.launch_monitor.schema import IDENTITY_COLUMNS

_DEFAULT_SCAN_EXCLUSIONS = frozenset(
    (*IDENTITY_COLUMNS, "source_id", "row_order", "filename", "file_name")
)


def _assert_player_identity(player_column: str, context: AnalysisContextV2) -> None:
    identity = context.player_identity
    if identity.trust_level in {"not_provided", "untrusted_inferred"}:
        raise ValueError("player covariation requires explicit trusted player identity")
    if identity.identifier_column != player_column:
        raise ValueError(
            "player_column must match the declared player identifier_column"
        )


def _estimate_availability(
    path: str,
    state: str,
    reason: str | None,
    observed: int,
    required: int,
) -> AvailabilityV2:
    if state == "available":
        return AvailabilityV2(result_path=path, state="available")
    return AvailabilityV2(
        result_path=path,
        state="unavailable",
        reason_code=reason,
        message=f"{path} is unavailable: {reason}.",
        observed_count=observed,
        required_count=required,
    )


def _availability(
    statistics: PairStatistics, request: PlayerCovariationRequestV1
) -> tuple[AvailabilityV2, ...]:
    estimates = (
        ("pooled", statistics.pooled, request.min_samples),
        ("within_player", statistics.within_player, request.min_samples),
        ("between_player", statistics.between_player, 2),
    )
    items = [
        _estimate_availability(
            path,
            estimate.state,
            estimate.reason_code,
            estimate.sample_count,
            required,
        )
        for path, estimate, required in estimates
    ]
    for index, player in enumerate(statistics.per_player):
        estimate = player.estimate
        items.append(
            _estimate_availability(
                f"per_player.{index}",
                estimate.state,
                estimate.reason_code,
                estimate.sample_count,
                request.min_samples,
            )
        )
    meta = statistics.meta_analysis
    items.append(
        _estimate_availability(
            "meta_analysis",
            meta.state,
            meta.reason_code,
            meta.contributor_count,
            2,
        )
    )
    return tuple(items)


def _overall_status(items: tuple[AvailabilityV2, ...]) -> AvailabilityState:
    available = sum(item.state == "available" for item in items)
    if available == 0:
        return "unavailable"
    if available < len(items):
        return "partial"
    return "available"


def _uncertainty(request: PlayerCovariationRequestV1) -> CovariationUncertaintyV1:
    return CovariationUncertaintyV1(
        confidence_level=request.confidence_level,
        assumptions=(
            "Player effects are descriptive and do not establish causality.",
            "Fisher-z intervals assume approximately independent observations.",
            "The pooled interval is not adjusted for repeated shots by player.",
            "DerSimonian-Laird estimates between-player effect heterogeneity.",
            "Population generalization requires a representative player sample.",
        ),
    )


def analyze_player_covariation_v1(
    frame: pd.DataFrame,
    request: PlayerCovariationRequestV1,
    *,
    context: AnalysisContextV2 | None = None,
) -> PlayerCovariationResultV1:
    """Analyze one pair without mutating the input frame.

    Postcondition: all input rows have content-addressed backing references and
    every unavailable statistical scope has a reason code.
    """

    resolved_context = context or AnalysisContextV2()
    _assert_player_identity(request.player_column, resolved_context)
    statistics = compute_pair_statistics(frame, request)
    availability = _availability(statistics, request)
    selected = (request.player_column, request.x_column, request.y_column)
    return PlayerCovariationResultV1(
        status=_overall_status(availability),
        request=request,
        pooled=statistics.pooled,
        within_player=statistics.within_player,
        between_player=statistics.between_player,
        per_player=statistics.per_player,
        meta_analysis=statistics.meta_analysis,
        missingness=statistics.missingness,
        units={
            request.x_column: metric_units_v2(request.x_column, resolved_context),
            request.y_column: metric_units_v2(request.y_column, resolved_context),
        },
        lineage=analysis_lineage_v2(frame, resolved_context, selected),
        availability=availability,
        uncertainty=_uncertainty(request),
        player_identity=resolved_context.player_identity,
        vendor_provenance=vendor_provenance_v2(
            frame, (request.x_column, request.y_column)
        ),
        definitions={
            "pooled": "Association across all pairwise-complete shots.",
            "within_player": "Association after centering x and y by player.",
            "between_player": "Association between unweighted player means.",
            "meta_analysis": (
                "Fixed and DerSimonian-Laird random effects in Fisher-z space."
            ),
        },
        warnings=statistics.warnings,
    )


def _has_finite_numeric(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    return bool(np.isfinite(values).any())


def _scan_columns(
    frame: pd.DataFrame, request: PlayerCovariationScanRequestV1
) -> tuple[str, ...]:
    if request.numeric_columns:
        missing = sorted(set(request.numeric_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"Columns not present in dataset: {', '.join(missing)}")
        columns = request.numeric_columns
    else:
        columns = tuple(
            column
            for column in frame.columns
            if column != request.player_column
            and column not in _DEFAULT_SCAN_EXCLUSIONS
            and _has_finite_numeric(frame[column])
        )
    unique = tuple(sorted(dict.fromkeys(columns)))
    if len(unique) < 2:
        raise ValueError("pair scan requires at least two numeric columns")
    if len(unique) > 20:
        raise ValueError("pair scan accepts at most 20 numeric columns")
    return unique


def _pair_rank(
    frame: pd.DataFrame,
    request: PlayerCovariationScanRequestV1,
    context: AnalysisContextV2,
    x_column: str,
    y_column: str,
) -> CovariationPairRankV1:
    pair_request = PlayerCovariationRequestV1(
        x_column=x_column,
        y_column=y_column,
        player_column=request.player_column,
        min_samples=request.min_samples,
        confidence_level=request.confidence_level,
    )
    statistics = compute_pair_statistics(frame, pair_request)
    meta = statistics.meta_analysis
    valid = [
        item.estimate.pearson_r
        for item in statistics.per_player
        if item.estimate.pearson_r is not None
    ]
    consistency = None
    if meta.random_effect_r is not None and valid:
        consistency = float(np.mean(np.sign(valid) == np.sign(meta.random_effect_r)))
    return CovariationPairRankV1(
        rank=1,
        state=meta.state,
        reason_code=meta.reason_code,
        x_column=x_column,
        y_column=y_column,
        x_unit=metric_units_v2(x_column, context),
        y_unit=metric_units_v2(y_column, context),
        random_effect_r=meta.random_effect_r,
        fixed_effect_r=meta.fixed_effect_r,
        within_player_r=statistics.within_player.pearson_r,
        between_player_r=statistics.between_player.pearson_r,
        contributor_count=meta.contributor_count,
        total_sample_count=meta.total_sample_count,
        input_row_count=statistics.missingness.input_row_count,
        pairwise_complete_row_count=statistics.missingness.pairwise_complete_row_count,
        excluded_row_count=(
            statistics.missingness.input_row_count
            - statistics.missingness.pairwise_complete_row_count
        ),
        i_squared_pct=meta.i_squared_pct,
        direction_consistency=consistency,
    )


def _rank_pairs(
    items: list[CovariationPairRankV1],
) -> tuple[CovariationPairRankV1, ...]:
    ordered = sorted(
        items,
        key=lambda item: (
            item.state == "unavailable",
            -abs(item.random_effect_r) if item.random_effect_r is not None else 0,
            -item.contributor_count,
            item.x_column,
            item.y_column,
        ),
    )
    return tuple(
        item.model_copy(update={"rank": rank})
        for rank, item in enumerate(ordered, start=1)
    )


def scan_player_covariation_v1(
    frame: pd.DataFrame,
    request: PlayerCovariationScanRequestV1,
    *,
    context: AnalysisContextV2 | None = None,
) -> PlayerCovariationScanResultV1:
    """Rank bounded numeric pairs without copying source values into output."""

    resolved_context = context or AnalysisContextV2()
    _assert_player_identity(request.player_column, resolved_context)
    columns = _scan_columns(frame, request)
    items = [
        _pair_rank(frame, request, resolved_context, x_column, y_column)
        for x_column, y_column in combinations(columns, 2)
    ]
    ranking = _rank_pairs(items)
    available = sum(item.state == "available" for item in ranking)
    unavailable = len(ranking) - available
    status: AvailabilityState = (
        "available"
        if unavailable == 0
        else "unavailable"
        if available == 0
        else "partial"
    )
    return PlayerCovariationScanResultV1(
        status=status,
        request=request,
        pair_count=len(ranking),
        available_pair_count=available,
        unavailable_pair_count=unavailable,
        ranking=ranking,
        lineage=analysis_lineage_v2(
            frame, resolved_context, (request.player_column, *columns)
        ),
        player_identity=resolved_context.player_identity,
        vendor_provenance=vendor_provenance_v2(frame, columns),
        warnings=(
            f"Exploratory scan evaluated {len(ranking)} pairs; rankings are not confirmatory.",
            "Multiplicity increases false-positive risk; validate selected relationships on held-out data.",
            "Correlation does not imply causality or population generalizability.",
        ),
        method_description=(
            "Pairs rank by absolute random-effects Pearson correlation, then "
            "eligible-player count and lexical variable names."
        ),
    )


def player_covariation_contract_json_schema() -> dict[str, object]:
    """Return the selected-pair/pair-scan JSON Schema bundle."""

    return PlayerCovariationContractV1.model_json_schema()


__all__ = [
    "analyze_player_covariation_v1",
    "player_covariation_contract_json_schema",
    "scan_player_covariation_v1",
]

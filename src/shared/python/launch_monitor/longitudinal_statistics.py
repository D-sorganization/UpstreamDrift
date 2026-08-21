"""Session-unit descriptive and player-clustered longitudinal statistics."""

from __future__ import annotations

from math import sqrt
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.shared.python.launch_monitor.longitudinal_types import (
    LongitudinalPlayerAssociationV1,
    LongitudinalSessionRequestV1,
    PooledAssociationV1,
)


def _direction(estimate: float) -> Literal["increasing", "decreasing", "flat"]:
    if estimate > 1e-12:
        return "increasing"
    if estimate < -1e-12:
        return "decreasing"
    return "flat"


def player_associations(
    cells: pd.DataFrame, request: LongitudinalSessionRequestV1
) -> tuple[LongitudinalPlayerAssociationV1, ...]:
    """Fit descriptive slopes after collapsing every player-session equally."""
    session_means = (
        cells.groupby(["player_id", "session_id", "order_value"], as_index=False)[
            "metric_value"
        ]
        .mean()
        .sort_values(["player_id", "order_value", "session_id"])
    )
    results: list[LongitudinalPlayerAssociationV1] = []
    for player_id, group in session_means.groupby("player_id", sort=True):
        session_count = len(group)
        if (
            session_count < request.minimum_sessions_per_player
            or group["order_value"].nunique() < 2
        ):
            results.append(
                LongitudinalPlayerAssociationV1(
                    player_id=str(player_id),
                    session_count=session_count,
                    direction="unavailable",
                    state="unavailable",
                    reason_code="insufficient_ordered_sessions",
                )
            )
            continue
        estimate = float(
            stats.linregress(group["order_value"], group["metric_value"]).slope
        )
        results.append(
            LongitudinalPlayerAssociationV1(
                player_id=str(player_id),
                session_count=session_count,
                estimate_per_order_unit=estimate,
                direction=_direction(estimate),
                state="available",
            )
        )
    return tuple(results)


def _design_matrix(
    cells: pd.DataFrame, request: LongitudinalSessionRequestV1
) -> tuple[np.ndarray, tuple[str, ...]]:
    numeric = cells[["order_value", *request.confounders]].astype(float)
    categorical_columns = ["player_id", *request.strata]
    categorical = pd.get_dummies(
        cells[categorical_columns].astype(str),
        columns=categorical_columns,
        drop_first=True,
        dtype=float,
    )
    design = pd.concat(
        [
            pd.Series(1.0, index=cells.index, name="intercept"),
            numeric,
            categorical,
        ],
        axis=1,
    )
    return design.to_numpy(dtype=float), tuple(str(item) for item in design.columns)


def clustered_pooled_association(
    cells: pd.DataFrame, request: LongitudinalSessionRequestV1
) -> tuple[PooledAssociationV1 | None, str | None, tuple[str, ...]]:
    """Fit player-FE OLS with a finite-cluster corrected sandwich covariance."""
    cluster_labels = cells["player_id"].astype(str).to_numpy()
    clusters = tuple(sorted(set(cluster_labels)))
    if len(clusters) < request.minimum_player_clusters:
        return None, "insufficient_player_clusters", ()
    matrix, terms = _design_matrix(cells, request)
    values = cells["metric_value"].to_numpy(dtype=float)
    observations, parameter_count = matrix.shape
    if (
        observations <= parameter_count
        or np.linalg.matrix_rank(matrix) < parameter_count
    ):
        return None, "rank_deficient_session_design", terms
    bread = np.linalg.inv(matrix.T @ matrix)
    coefficients = bread @ matrix.T @ values
    residuals = values - matrix @ coefficients
    meat = np.zeros((parameter_count, parameter_count), dtype=float)
    for cluster in clusters:
        selected = cluster_labels == cluster
        score = matrix[selected].T @ residuals[selected]
        meat += np.outer(score, score)
    correction = (len(clusters) / (len(clusters) - 1)) * (
        (observations - 1) / (observations - parameter_count)
    )
    covariance = correction * bread @ meat @ bread
    variance = float(covariance[1, 1])
    if not np.isfinite(variance) or variance <= 0:
        return None, "degenerate_clustered_variance", terms
    standard_error = sqrt(variance)
    estimate = float(coefficients[1])
    degrees_of_freedom = len(clusters) - 1
    critical = float(
        stats.t.ppf(0.5 + request.confidence_level / 2.0, degrees_of_freedom)
    )
    statistic = estimate / standard_error
    p_value = float(2 * stats.t.sf(abs(statistic), degrees_of_freedom))
    return (
        PooledAssociationV1(
            estimate_per_order_unit=estimate,
            standard_error=standard_error,
            confidence_interval_low=estimate - critical * standard_error,
            confidence_interval_high=estimate + critical * standard_error,
            p_value=p_value,
            confidence_level=request.confidence_level,
            cluster_count=len(clusters),
            session_cell_count=observations,
        ),
        None,
        terms,
    )


__all__ = ["clustered_pooled_association", "player_associations"]

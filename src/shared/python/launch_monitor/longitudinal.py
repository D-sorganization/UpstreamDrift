"""Attested longitudinal analysis with player-session as the inference unit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.shared.python.launch_monitor.contract_v2 import (
    AnalysisContextV2,
    AvailabilityV2,
    build_analysis_lineage_v2,
)
from src.shared.python.launch_monitor.longitudinal_statistics import (
    clustered_pooled_association,
    player_associations,
)
from src.shared.python.launch_monitor.longitudinal_types import (
    LongitudinalDesignV1,
    LongitudinalMissingnessV1,
    LongitudinalSessionRequestV1,
    LongitudinalSessionResultV1,
    SessionAggregateV1,
)


@dataclass(frozen=True)
class _QualifiedColumns:
    player: str
    session: str
    order: str


def _qualified_columns(
    frame: pd.DataFrame, context: AnalysisContextV2
) -> tuple[_QualifiedColumns | None, str | None]:
    player = context.player_identity
    session = context.session_identity
    order = context.order_evidence
    if player.trust_level in {"not_provided", "untrusted_inferred"}:
        return None, "untrusted_player_identity"
    if session.trust_level in {"not_provided", "untrusted_inferred"}:
        return None, "untrusted_session_identity"
    if order.trust_level in {"not_provided", "untrusted_inferred"}:
        return None, "untrusted_order_evidence"
    assert player.identifier_column is not None
    assert session.identifier_column is not None
    assert order.order_column is not None
    columns = _QualifiedColumns(
        player.identifier_column, session.identifier_column, order.order_column
    )
    required = {columns.player, columns.session, columns.order}
    if not required <= set(frame.columns):
        return None, "identity_or_order_column_missing"
    if len(required) != 3:
        return None, "identity_and_order_columns_must_be_distinct"
    return columns, None


def _order_values(
    values: pd.Series, context: AnalysisContextV2
) -> tuple[pd.Series | None, str | None]:
    evidence = context.order_evidence
    if evidence.order_kind != "timestamp":
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric, None
    unit_seconds = {"second": 1.0, "hour": 3600.0, "day": 86400.0}
    divisor = unit_seconds.get(str(evidence.unit).lower())
    if divisor is None:
        return None, "unsupported_timestamp_order_unit"
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    numeric = timestamps.map(
        lambda item: np.nan if pd.isna(item) else item.timestamp() / divisor
    )
    return numeric.astype(float), None


def _prepare_rows(
    frame: pd.DataFrame,
    request: LongitudinalSessionRequestV1,
    context: AnalysisContextV2,
    columns: _QualifiedColumns,
) -> tuple[pd.DataFrame | None, dict[str, int], str | None]:
    selected = [
        columns.player,
        columns.session,
        columns.order,
        request.metric,
        *request.strata,
        *request.confounders,
    ]
    missing_columns = sorted(set(selected) - set(frame.columns))
    if missing_columns:
        return None, {}, "analysis_column_missing"
    prepared = frame[selected].copy()
    for column in (columns.player, columns.session, *request.strata):
        blank = prepared[column].map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        prepared.loc[blank, column] = pd.NA
    prepared["_order_value"], order_error = _order_values(
        prepared[columns.order], context
    )
    if order_error:
        return None, {}, order_error
    prepared[request.metric] = pd.to_numeric(prepared[request.metric], errors="coerce")
    for confounder in request.confounders:
        prepared[confounder] = pd.to_numeric(prepared[confounder], errors="coerce")
    numeric_columns = ["_order_value", request.metric, *request.confounders]
    prepared[numeric_columns] = prepared[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    incomplete = prepared.isna().any(axis=1)
    excluded = {"incomplete_or_nonfinite_selected_fields": int(incomplete.sum())}
    prepared = prepared.loc[~incomplete].copy()
    if prepared.empty:
        return None, excluded, "no_complete_finite_shots"
    group = prepared.groupby([columns.player, columns.session], dropna=False)
    if (group["_order_value"].nunique() != 1).any():
        return None, excluded, "nonconstant_session_order"
    return prepared, excluded, None


def _aggregate_sessions(
    prepared: pd.DataFrame,
    request: LongitudinalSessionRequestV1,
    columns: _QualifiedColumns,
) -> pd.DataFrame:
    group_columns = [columns.player, columns.session, *request.strata]
    aggregation: dict[str, Any] = {
        "_order_value": "first",
        request.metric: request.session_aggregate,
        **dict.fromkeys(request.confounders, "mean"),
    }
    grouped = prepared.groupby(group_columns, as_index=False, dropna=False)
    cells = grouped.agg(aggregation)
    cells["shot_count"] = grouped.size()["size"]
    return cells.rename(
        columns={
            columns.player: "player_id",
            columns.session: "session_id",
            "_order_value": "order_value",
            request.metric: "metric_value",
        }
    )


def _session_records(
    cells: pd.DataFrame,
    request: LongitudinalSessionRequestV1,
    context: AnalysisContextV2,
) -> tuple[SessionAggregateV1, ...]:
    order_unit = context.order_evidence.unit
    assert order_unit is not None
    records: list[SessionAggregateV1] = []
    for row in cells.sort_values(["player_id", "order_value", "session_id"]).to_dict(
        orient="records"
    ):
        records.append(
            SessionAggregateV1(
                player_id=str(row["player_id"]),
                session_id=str(row["session_id"]),
                order_value=float(row["order_value"]),
                order_unit=order_unit,
                stratum={column: str(row[column]) for column in request.strata},
                shot_count=int(row["shot_count"]),
                metric_value=float(row["metric_value"]),
                confounder_values={
                    column: float(row[column]) for column in request.confounders
                },
            )
        )
    return tuple(records)


def _unavailable(
    frame: pd.DataFrame,
    request: LongitudinalSessionRequestV1,
    context: AnalysisContextV2,
    reason: str,
    excluded: dict[str, int] | None = None,
) -> LongitudinalSessionResultV1:
    return LongitudinalSessionResultV1(
        status="unavailable",
        request=request,
        design=LongitudinalDesignV1(
            session_aggregate=request.session_aggregate,
            strata=request.strata,
            confounders=request.confounders,
            pooled_terms=(),
        ),
        session_aggregates=(),
        player_associations=(),
        pooled_association=None,
        availability=(
            AvailabilityV2(
                result_path="analysis",
                state="unavailable",
                reason_code=reason,
                message="The attested longitudinal design could not be qualified.",
            ),
        ),
        missingness=LongitudinalMissingnessV1(
            input_row_count=len(frame),
            included_shot_count=0,
            session_cell_count=0,
            excluded_by_reason=excluded or {},
        ),
        lineage=build_analysis_lineage_v2(frame, context),
        player_identity=context.player_identity,
        session_identity=context.session_identity,
        order_evidence=context.order_evidence,
        warnings=("No causal improvement claim is available.",),
    )


def analyze_longitudinal_sessions(
    frame: pd.DataFrame,
    request: LongitudinalSessionRequestV1,
    *,
    context: AnalysisContextV2,
) -> LongitudinalSessionResultV1:
    """Estimate descriptive direction after one equal-weight cell per session/stratum."""
    columns, qualification_error = _qualified_columns(frame, context)
    if qualification_error or columns is None:
        return _unavailable(frame, request, context, str(qualification_error))
    prepared, excluded, preparation_error = _prepare_rows(
        frame, request, context, columns
    )
    if preparation_error or prepared is None:
        return _unavailable(frame, request, context, str(preparation_error), excluded)
    cells = _aggregate_sessions(prepared, request, columns)
    per_player = player_associations(cells, request)
    pooled, pooled_reason, pooled_terms = clustered_pooled_association(cells, request)
    availability = [
        AvailabilityV2(
            result_path=f"player_associations.{item.player_id}",
            state=item.state,
            reason_code=item.reason_code,
            observed_count=item.session_count,
            required_count=request.minimum_sessions_per_player,
        )
        for item in per_player
    ]
    availability.append(
        AvailabilityV2(
            result_path="pooled_association",
            state="available" if pooled is not None else "unavailable",
            reason_code=pooled_reason,
            message=(
                None
                if pooled is not None
                else "Clustered uncertainty was not estimable for this design."
            ),
            observed_count=int(cells["player_id"].nunique()),
            required_count=request.minimum_player_clusters,
        )
    )
    unavailable_count = sum(item.state == "unavailable" for item in availability)
    status = "available" if unavailable_count == 0 else "partial"
    included_count = len(prepared)
    return LongitudinalSessionResultV1(
        status=status,
        request=request,
        design=LongitudinalDesignV1(
            session_aggregate=request.session_aggregate,
            strata=request.strata,
            confounders=request.confounders,
            pooled_terms=pooled_terms,
        ),
        session_aggregates=_session_records(cells, request, context),
        player_associations=per_player,
        pooled_association=pooled,
        availability=tuple(availability),
        missingness=LongitudinalMissingnessV1(
            input_row_count=len(frame),
            included_shot_count=included_count,
            session_cell_count=len(cells),
            excluded_by_reason=excluded,
        ),
        lineage=build_analysis_lineage_v2(frame, context),
        player_identity=context.player_identity,
        session_identity=context.session_identity,
        order_evidence=context.order_evidence,
        warnings=(
            "Associations are descriptive and do not establish player improvement.",
            "Shot rows were aggregated before inference to avoid pseudo-replication.",
        ),
    )


def longitudinal_session_contract_json_schema() -> dict[str, Any]:
    """Return the strict versioned result schema for API and generated clients."""
    return LongitudinalSessionResultV1.model_json_schema()


__all__ = ["analyze_longitudinal_sessions", "longitudinal_session_contract_json_schema"]

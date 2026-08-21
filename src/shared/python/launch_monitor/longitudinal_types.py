"""Wire types for attested session-unit longitudinal analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.shared.python.launch_monitor.contract_v2 import (
    AnalysisLineageV2,
    AvailabilityV2,
    OrderEvidenceV2,
    PlayerIdentityV2,
    SessionIdentityV2,
)

LONGITUDINAL_SESSION_CONTRACT_VERSION: Literal[
    "launch-monitor-longitudinal-session/1.0.0"
] = "launch-monitor-longitudinal-session/1.0.0"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LongitudinalSessionRequestV1(_StrictModel):
    """Scientific choices for session-unit directional association."""

    metric: str = Field(min_length=1)
    direction: Literal["higher_is_better", "lower_is_better", "descriptive_only"] = (
        "descriptive_only"
    )
    session_aggregate: Literal["mean", "median"] = "mean"
    strata: tuple[str, ...] = ()
    confounders: tuple[str, ...] = ()
    minimum_sessions_per_player: int = Field(default=3, ge=3)
    minimum_player_clusters: int = Field(default=4, ge=4)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)

    @model_validator(mode="after")
    def validate_design_terms(self) -> LongitudinalSessionRequestV1:
        terms = (*self.strata, *self.confounders)
        if len(terms) != len(set(terms)) or set(self.strata) & set(self.confounders):
            raise ValueError("strata and confounders must be unique and disjoint")
        if self.metric in terms:
            raise ValueError(
                "metric, strata, and confounders must be unique and disjoint"
            )
        if any(not term.strip() for term in terms):
            raise ValueError("strata and confounder names must be non-empty")
        return self


class LongitudinalDesignV1(_StrictModel):
    primary_unit: Literal["player_session_stratum"] = "player_session_stratum"
    session_aggregate: Literal["mean", "median"]
    strata: tuple[str, ...]
    confounders: tuple[str, ...]
    pooled_terms: tuple[str, ...]


class LongitudinalClaimsV1(_StrictModel):
    association_scope: Literal["descriptive_directional"] = "descriptive_directional"
    primary_unit: Literal["player_session_stratum"] = "player_session_stratum"
    shot_level_inference: bool = False
    causal_improvement: bool = False
    confounder_control_is_causal: bool = False


class LongitudinalMissingnessV1(_StrictModel):
    input_row_count: int = Field(ge=0)
    included_shot_count: int = Field(ge=0)
    session_cell_count: int = Field(ge=0)
    excluded_by_reason: dict[str, int]


class SessionAggregateV1(_StrictModel):
    player_id: str
    session_id: str
    order_value: float
    order_unit: str
    stratum: dict[str, str]
    shot_count: int = Field(ge=1)
    metric_value: float
    confounder_values: dict[str, float]


class LongitudinalPlayerAssociationV1(_StrictModel):
    player_id: str
    session_count: int = Field(ge=0)
    estimate_per_order_unit: float | None = None
    direction: Literal["increasing", "decreasing", "flat", "unavailable"]
    state: Literal["available", "unavailable"]
    reason_code: str | None = None


class PooledAssociationV1(_StrictModel):
    method: Literal["player_fixed_effects_ols_clustered_by_player"] = (
        "player_fixed_effects_ols_clustered_by_player"
    )
    estimate_per_order_unit: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float
    p_value: float
    confidence_level: float
    cluster_count: int = Field(ge=1)
    session_cell_count: int = Field(ge=1)
    uncertainty_state: Literal["available"] = "available"


class LongitudinalSessionResultV1(_StrictModel):
    """Evidence-bearing result that never labels association as improvement."""

    contract_version: Literal["launch-monitor-longitudinal-session/1.0.0"] = (
        LONGITUDINAL_SESSION_CONTRACT_VERSION
    )
    status: Literal["available", "partial", "unavailable"]
    request: LongitudinalSessionRequestV1
    design: LongitudinalDesignV1
    session_aggregates: tuple[SessionAggregateV1, ...]
    player_associations: tuple[LongitudinalPlayerAssociationV1, ...]
    pooled_association: PooledAssociationV1 | None
    availability: tuple[AvailabilityV2, ...]
    missingness: LongitudinalMissingnessV1
    lineage: AnalysisLineageV2
    player_identity: PlayerIdentityV2
    session_identity: SessionIdentityV2
    order_evidence: OrderEvidenceV2
    claims: LongitudinalClaimsV1 = Field(default_factory=LongitudinalClaimsV1)
    warnings: tuple[str, ...] = ()


__all__ = [
    "LONGITUDINAL_SESSION_CONTRACT_VERSION",
    "LongitudinalClaimsV1",
    "LongitudinalDesignV1",
    "LongitudinalMissingnessV1",
    "LongitudinalSessionRequestV1",
    "LongitudinalSessionResultV1",
    "LongitudinalPlayerAssociationV1",
    "PooledAssociationV1",
    "SessionAggregateV1",
]

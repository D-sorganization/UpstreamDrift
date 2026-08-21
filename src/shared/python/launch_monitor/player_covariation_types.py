"""Versioned wire models for player covariation and population synthesis."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from src.shared.python.launch_monitor.contract_v2 import (
    AnalysisLineageV2,
    AvailabilityState,
    AvailabilityV2,
    ClaimsV2,
    MetricUnitsV2,
    PlayerIdentityV2,
    VendorProvenanceV2,
)

PLAYER_COVARIATION_CONTRACT_VERSION: Literal[
    "launch-monitor-player-covariation/1.0.0"
] = "launch-monitor-player-covariation/1.0.0"
AssociationState = Literal["available", "unavailable"]
AssociationUnavailableReason = Literal[
    "insufficient_samples",
    "insufficient_groups",
    "constant_x",
    "constant_y",
    "constant_both",
]


class _CovariationModel(BaseModel):
    """Strict immutable base for the public covariation records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PlayerCovariationRequestV1(_CovariationModel):
    """Select one variable pair and the player-level inference rules."""

    x_column: str = Field(min_length=1)
    y_column: str = Field(min_length=1)
    player_column: str = Field(min_length=1)
    min_samples: int = Field(default=4, ge=4)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)

    @model_validator(mode="after")
    def require_distinct_variables(self) -> PlayerCovariationRequestV1:
        if self.x_column == self.y_column:
            raise ValueError("x_column and y_column must differ")
        if self.player_column in {self.x_column, self.y_column}:
            raise ValueError("player_column cannot also be an analyzed variable")
        return self


class PlayerCovariationScanRequestV1(_CovariationModel):
    """Select a bounded exploratory all-pairs scan."""

    player_column: str = Field(min_length=1)
    numeric_columns: tuple[str, ...] = Field(default=(), max_length=20)
    min_samples: int = Field(default=4, ge=4)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)

    @model_validator(mode="after")
    def require_unique_columns(self) -> PlayerCovariationScanRequestV1:
        if any(not column.strip() for column in self.numeric_columns):
            raise ValueError("numeric_columns values must be non-empty")
        if len(set(self.numeric_columns)) != len(self.numeric_columns):
            raise ValueError("numeric_columns values must be unique")
        return self


class AssociationEstimateV1(_CovariationModel):
    """One descriptive association or a typed unavailable state."""

    state: AssociationState
    reason_code: AssociationUnavailableReason | None = None
    sample_count: int = Field(ge=0)
    group_count: int = Field(ge=0)
    pearson_r: float | None = Field(default=None, ge=-1, le=1)
    spearman_r: float | None = Field(default=None, ge=-1, le=1)
    slope: float | None = None
    intercept: float | None = None
    r_squared: float | None = Field(default=None, ge=0, le=1)
    ci_lower: float | None = Field(default=None, ge=-1, le=1)
    ci_upper: float | None = Field(default=None, ge=-1, le=1)

    @model_validator(mode="after")
    def require_consistent_state(self) -> AssociationEstimateV1:
        estimates = (
            self.pearson_r,
            self.spearman_r,
            self.slope,
            self.intercept,
            self.r_squared,
        )
        if self.state == "available" and (
            self.reason_code is not None or any(value is None for value in estimates)
        ):
            raise ValueError("available association requires all point estimates")
        if self.state == "unavailable" and (
            self.reason_code is None or any(value is not None for value in estimates)
        ):
            raise ValueError(
                "unavailable association requires reason_code and null estimates"
            )
        if (self.ci_lower is None) != (self.ci_upper is None):
            raise ValueError("association interval bounds must be supplied together")
        return self


class PlayerAssociationV1(_CovariationModel):
    """A player-level estimate and normalized meta-analysis weights."""

    player_id: str = Field(min_length=1)
    estimate: AssociationEstimateV1
    fixed_weight: float | None = Field(default=None, ge=0, le=1)
    random_weight: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_consistent_weights(self) -> PlayerAssociationV1:
        if (self.fixed_weight is None) != (self.random_weight is None):
            raise ValueError("fixed and random weights must be supplied together")
        if self.estimate.state == "unavailable" and self.fixed_weight is not None:
            raise ValueError("unavailable player estimates cannot carry weights")
        return self


class MetaAnalysisSummaryV1(_CovariationModel):
    """Fixed/random Fisher-z synthesis with heterogeneity diagnostics."""

    state: AssociationState
    reason_code: Literal["insufficient_eligible_players"] | None = None
    contributor_count: int = Field(ge=0)
    total_sample_count: int = Field(ge=0)
    fixed_effect_r: float | None = Field(default=None, ge=-1, le=1)
    fixed_ci_lower: float | None = Field(default=None, ge=-1, le=1)
    fixed_ci_upper: float | None = Field(default=None, ge=-1, le=1)
    random_effect_r: float | None = Field(default=None, ge=-1, le=1)
    random_ci_lower: float | None = Field(default=None, ge=-1, le=1)
    random_ci_upper: float | None = Field(default=None, ge=-1, le=1)
    tau_squared: float | None = Field(default=None, ge=0)
    q_statistic: float | None = Field(default=None, ge=0)
    i_squared_pct: float | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def require_consistent_state(self) -> MetaAnalysisSummaryV1:
        values = (
            self.fixed_effect_r,
            self.fixed_ci_lower,
            self.fixed_ci_upper,
            self.random_effect_r,
            self.random_ci_lower,
            self.random_ci_upper,
            self.tau_squared,
            self.q_statistic,
            self.i_squared_pct,
        )
        if self.state == "available" and (
            self.reason_code is not None or any(value is None for value in values)
        ):
            raise ValueError("available meta-analysis requires all estimates")
        if self.state == "unavailable" and (
            self.reason_code is None or any(value is not None for value in values)
        ):
            raise ValueError(
                "unavailable meta-analysis requires reason_code and null estimates"
            )
        return self


class CovariationMissingnessV1(_CovariationModel):
    """Row and player exclusions used by a selected-pair analysis."""

    input_row_count: int = Field(ge=0)
    pairwise_complete_row_count: int = Field(ge=0)
    missing_by_variable: dict[str, int]
    non_numeric_by_variable: dict[str, int]
    non_finite_by_variable: dict[str, int]
    excluded_by_reason: dict[str, int]
    eligible_player_count: int = Field(ge=0)
    excluded_player_count_by_reason: dict[str, int]
    policy: Literal["pairwise"] = "pairwise"


class CovariationUncertaintyV1(_CovariationModel):
    """Named uncertainty methods and their scientific limits."""

    confidence_level: float = Field(gt=0.5, lt=1.0)
    per_player_interval: Literal["fisher-z"] = "fisher-z"
    pooled_interval: Literal["fisher-z-unclustered"] = "fisher-z-unclustered"
    within_player_interval: Literal["unavailable-clustered"] = "unavailable-clustered"
    fixed_effect_method: Literal["inverse-variance-fisher-z"] = (
        "inverse-variance-fisher-z"
    )
    random_effect_method: Literal["dersimonian-laird-fisher-z"] = (
        "dersimonian-laird-fisher-z"
    )
    assumptions: tuple[str, ...]


class CovariationPairRankV1(_CovariationModel):
    """One deterministically ranked pair from an exploratory scan."""

    rank: int = Field(ge=1)
    state: AssociationState
    reason_code: Literal["insufficient_eligible_players"] | None = None
    x_column: str = Field(min_length=1)
    y_column: str = Field(min_length=1)
    x_unit: MetricUnitsV2
    y_unit: MetricUnitsV2
    random_effect_r: float | None = Field(default=None, ge=-1, le=1)
    fixed_effect_r: float | None = Field(default=None, ge=-1, le=1)
    within_player_r: float | None = Field(default=None, ge=-1, le=1)
    between_player_r: float | None = Field(default=None, ge=-1, le=1)
    contributor_count: int = Field(ge=0)
    total_sample_count: int = Field(ge=0)
    input_row_count: int = Field(ge=0)
    pairwise_complete_row_count: int = Field(ge=0)
    excluded_row_count: int = Field(ge=0)
    i_squared_pct: float | None = Field(default=None, ge=0, le=100)
    direction_consistency: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_consistent_state(self) -> CovariationPairRankV1:
        if self.state == "available" and self.reason_code is not None:
            raise ValueError("available ranked pair cannot have a reason_code")
        if self.state == "unavailable" and self.reason_code is None:
            raise ValueError("unavailable ranked pair requires a reason_code")
        if (
            self.pairwise_complete_row_count + self.excluded_row_count
            != self.input_row_count
        ):
            raise ValueError("ranked pair row counts must reconcile to input_row_count")
        return self


class PlayerCovariationResultV1(_CovariationModel):
    """Evidence-bearing result for one selected variable pair."""

    contract_version: Literal["launch-monitor-player-covariation/1.0.0"] = (
        PLAYER_COVARIATION_CONTRACT_VERSION
    )
    analysis_kind: Literal["selected_pair"] = "selected_pair"
    status: AvailabilityState
    request: PlayerCovariationRequestV1
    pooled: AssociationEstimateV1
    within_player: AssociationEstimateV1
    between_player: AssociationEstimateV1
    per_player: tuple[PlayerAssociationV1, ...]
    meta_analysis: MetaAnalysisSummaryV1
    missingness: CovariationMissingnessV1
    units: dict[str, MetricUnitsV2]
    lineage: AnalysisLineageV2
    availability: tuple[AvailabilityV2, ...]
    uncertainty: CovariationUncertaintyV1
    player_identity: PlayerIdentityV2
    vendor_provenance: tuple[VendorProvenanceV2, ...]
    claims: ClaimsV2 = Field(default_factory=ClaimsV2)
    definitions: dict[str, str]
    warnings: tuple[str, ...]


class PlayerCovariationScanResultV1(_CovariationModel):
    """Evidence-bearing deterministic exploratory pair ranking."""

    contract_version: Literal["launch-monitor-player-covariation/1.0.0"] = (
        PLAYER_COVARIATION_CONTRACT_VERSION
    )
    analysis_kind: Literal["pair_scan"] = "pair_scan"
    status: AvailabilityState
    request: PlayerCovariationScanRequestV1
    pair_count: int = Field(ge=0)
    available_pair_count: int = Field(ge=0)
    unavailable_pair_count: int = Field(ge=0)
    ranking: tuple[CovariationPairRankV1, ...]
    lineage: AnalysisLineageV2
    player_identity: PlayerIdentityV2
    vendor_provenance: tuple[VendorProvenanceV2, ...]
    claims: ClaimsV2 = Field(default_factory=ClaimsV2)
    warnings: tuple[str, ...]
    method_description: str

    @model_validator(mode="after")
    def require_consistent_counts(self) -> PlayerCovariationScanResultV1:
        available = sum(item.state == "available" for item in self.ranking)
        unavailable = len(self.ranking) - available
        if (
            self.pair_count != len(self.ranking)
            or self.available_pair_count != available
            or self.unavailable_pair_count != unavailable
        ):
            raise ValueError("pair counts must match the ranked pair states")
        if tuple(item.rank for item in self.ranking) != tuple(
            range(1, self.pair_count + 1)
        ):
            raise ValueError("ranked pairs must use consecutive one-based ranks")
        expected_status: AvailabilityState = (
            "available"
            if unavailable == 0
            else "unavailable"
            if available == 0
            else "partial"
        )
        if self.status != expected_status:
            raise ValueError("scan status must match the ranked pair states")
        return self


CovariationResultUnion = Annotated[
    PlayerCovariationResultV1 | PlayerCovariationScanResultV1,
    Field(discriminator="analysis_kind"),
]


class PlayerCovariationContractV1(RootModel[CovariationResultUnion]):
    """Schema root covering selected-pair and exploratory-scan results."""

    model_config = ConfigDict(frozen=True)


__all__ = [
    "PLAYER_COVARIATION_CONTRACT_VERSION",
    "AssociationEstimateV1",
    "CovariationMissingnessV1",
    "CovariationPairRankV1",
    "CovariationUncertaintyV1",
    "MetaAnalysisSummaryV1",
    "PlayerAssociationV1",
    "PlayerCovariationContractV1",
    "PlayerCovariationRequestV1",
    "PlayerCovariationResultV1",
    "PlayerCovariationScanRequestV1",
    "PlayerCovariationScanResultV1",
]

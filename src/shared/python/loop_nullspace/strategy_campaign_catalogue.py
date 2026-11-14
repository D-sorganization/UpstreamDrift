"""Named NSX-95 comparison campaigns as data (#12724).

The owner's comparisons are controlled: matched clubhead speed, matched effort
within one basis, unconstrained Pareto fronts, alternative-disturbance priors
and held-out Monte Carlo seeds (NSX-91).  This module names those campaigns so
callers and docs refer to one definition instead of re-stating controls.

The first campaign, ``variability_toy_matched_chs``, is the fixture-backed suite
exercised in ``tests/unit/loop_nullspace/test_strategy_campaign.py``.  Campaigns
that depend on not-yet-implemented catalogue arms stay ``DEFERRED`` with an
explicit reason; no experimental numbers are invented here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from src.shared.python.loop_nullspace.strategy_campaign import ComparisonMode
from src.shared.python.loop_nullspace.strategy_spec import StrategyId

__all__ = [
    "CAMPAIGN_CATALOGUE",
    "CampaignStatus",
    "ComparisonCampaignSpec",
    "IMPLEMENTED_STRATEGY_IDS",
    "campaign_for",
    "export_campaign_catalogue",
]

_IMPLEMENTED: tuple[StrategyId, ...] = (
    StrategyId.S1,
    StrategyId.S2,
    StrategyId.S3,
    StrategyId.S5,
    StrategyId.S7,
    StrategyId.S8,
)

IMPLEMENTED_STRATEGY_IDS: tuple[str, ...] = tuple(s.value for s in _IMPLEMENTED)

_DEFERRED_ARMS = "S4, S6, S9, S11, S12"


class CampaignStatus(str, Enum):
    """Whether a campaign can be run with the current evaluator set."""

    AVAILABLE = "available"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class ComparisonCampaignSpec:
    """One named campaign: controls, scope and status.

    ``comparison_modes`` are :class:`~.strategy_campaign.ComparisonMode` values.
    ``strategy_ids`` lists catalogue ids included when the campaign runs; an
    empty tuple means every *implemented* id at run time (never the deferred arms).
    """

    id: str
    description: str
    status: CampaignStatus
    comparison_modes: tuple[ComparisonMode, ...]
    requires_holdout_seeds: bool
    requires_prior_variants: bool
    strategy_ids: tuple[str, ...] = ()
    defer_reason: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.description.strip():
            raise ValueError("a campaign needs an id and a description")
        if not self.comparison_modes:
            raise ValueError(f"{self.id}: comparison_modes must be non-empty")
        if self.status is CampaignStatus.DEFERRED and not self.defer_reason.strip():
            raise ValueError(f"{self.id}: a deferred campaign needs defer_reason")


def _catalogue() -> dict[str, ComparisonCampaignSpec]:
    return {
        "variability_toy_matched_chs": ComparisonCampaignSpec(
            id="variability_toy_matched_chs",
            description=(
                "Toy pendulum / layer / muscle fixtures: matched-CHS ranking, "
                "matched-effort cap, unconstrained fronts, prior reversals and "
                "held-out paired Monte Carlo on implemented arms S1–S3, S5, S7, S8."
            ),
            status=CampaignStatus.AVAILABLE,
            comparison_modes=(
                ComparisonMode.MATCHED_CHS,
                ComparisonMode.MATCHED_EFFORT,
                ComparisonMode.UNCONSTRAINED,
            ),
            requires_holdout_seeds=True,
            requires_prior_variants=True,
            strategy_ids=IMPLEMENTED_STRATEGY_IDS,
        ),
        "variability_full_catalogue_matched_chs": ComparisonCampaignSpec(
            id="variability_full_catalogue_matched_chs",
            description=(
                "Same controls as the toy campaign on every catalogue arm S1–S12 "
                "once missing evaluators exist."
            ),
            status=CampaignStatus.DEFERRED,
            comparison_modes=(
                ComparisonMode.MATCHED_CHS,
                ComparisonMode.MATCHED_EFFORT,
            ),
            requires_holdout_seeds=True,
            requires_prior_variants=True,
            strategy_ids=(),
            defer_reason=(
                f"Not implemented: {_DEFERRED_ARMS} need NSX-00 parameterisation; "
                "S10 remains UNAVAILABLE on rigid clubs."
            ),
        ),
        "muscle_biarticular_matched_effort": ComparisonCampaignSpec(
            id="muscle_biarticular_matched_effort",
            description=(
                "Muscle-driven matched-effort comparison for S9 biarticular coupling "
                "on the NSX-94 antagonist frame."
            ),
            status=CampaignStatus.DEFERRED,
            comparison_modes=(ComparisonMode.MATCHED_EFFORT,),
            requires_holdout_seeds=False,
            requires_prior_variants=False,
            strategy_ids=(StrategyId.S9.value,),
            defer_reason="S9: no biarticular comparison on the NSX-94 frame.",
        ),
    }


CAMPAIGN_CATALOGUE: Mapping[str, ComparisonCampaignSpec] = MappingProxyType(
    _catalogue()
)


def campaign_for(campaign_id: str) -> ComparisonCampaignSpec:
    """Return a catalogue campaign (``ValueError`` if unknown)."""
    if not campaign_id.strip():
        raise ValueError("campaign_id must be non-empty")
    try:
        return CAMPAIGN_CATALOGUE[campaign_id]
    except KeyError as exc:
        raise ValueError(f"unknown campaign {campaign_id!r}") from exc


def export_campaign_catalogue() -> list[dict[str, object]]:
    """JSON-serialisable campaign rows (no quantities, no invented results)."""
    rows: list[dict[str, object]] = []
    for spec in CAMPAIGN_CATALOGUE.values():
        rows.append(
            {
                "id": spec.id,
                "description": spec.description,
                "status": spec.status.value,
                "comparison_modes": [m.value for m in spec.comparison_modes],
                "requires_holdout_seeds": spec.requires_holdout_seeds,
                "requires_prior_variants": spec.requires_prior_variants,
                "strategy_ids": list(spec.strategy_ids),
                "defer_reason": spec.defer_reason,
            }
        )
    return rows

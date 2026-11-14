"""NSX-95 (#12724): comparison campaign catalogue is data; first campaign is named."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.loop_nullspace.strategy_campaign import ComparisonMode
from src.shared.python.loop_nullspace.strategy_campaign_catalogue import (
    CAMPAIGN_CATALOGUE,
    CampaignStatus,
    IMPLEMENTED_STRATEGY_IDS,
    campaign_for,
    export_campaign_catalogue,
)

pytestmark = pytest.mark.unit

_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "config"
    / "nsx_comparison_campaigns.json"
)


def test_first_campaign_is_available_and_lists_implemented_arms_only() -> None:
    first = campaign_for("variability_toy_matched_chs")
    assert first.status is CampaignStatus.AVAILABLE
    assert ComparisonMode.MATCHED_CHS in first.comparison_modes
    assert first.requires_holdout_seeds and first.requires_prior_variants
    assert set(first.strategy_ids) == set(IMPLEMENTED_STRATEGY_IDS)
    assert "S4" not in first.strategy_ids


def test_deferred_campaigns_name_missing_evaluators_not_fake_results() -> None:
    full = CAMPAIGN_CATALOGUE["variability_full_catalogue_matched_chs"]
    assert full.status is CampaignStatus.DEFERRED
    assert "S4" in full.defer_reason and "S10" in full.defer_reason
    muscle = CAMPAIGN_CATALOGUE["muscle_biarticular_matched_effort"]
    assert muscle.status is CampaignStatus.DEFERRED
    assert muscle.strategy_ids == ("S9",)


def test_unknown_campaign_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown campaign"):
        campaign_for("not_a_campaign")


def test_committed_campaign_json_matches_python_export() -> None:
    on_disk = json.loads(_CONFIG.read_text(encoding="utf-8"))
    assert on_disk == export_campaign_catalogue()

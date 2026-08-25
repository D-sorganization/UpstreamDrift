"""Canonical player-covariation and population-synthesis contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_launch_monitor_contract import player_covariation_golden_fixture
from src.shared.python.launch_monitor import (
    CONTRACT_VERSION,
    CONTRACT_VERSION_V2,
    PLAYER_COVARIATION_CONTRACT_VERSION,
    AnalysisContextV2,
    PlayerCovariationRequestV1,
    PlayerCovariationScanRequestV1,
    PlayerIdentityV2,
    SourceFileReferenceV2,
    analyze_player_covariation_v1,
    player_covariation_contract_json_schema,
    scan_player_covariation_v1,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = (
    REPO_ROOT
    / "docs"
    / "api"
    / "contracts"
    / "fixtures"
    / "launch-monitor-player-covariation-v1.golden.json"
)


def _context() -> AnalysisContextV2:
    return AnalysisContextV2(
        player_identity=PlayerIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column="player_id",
            evidence="The fixture owner attests these stable player labels.",
        ),
        sources=(
            SourceFileReferenceV2(
                source_id="synthetic-source",
                file_sha256="1" * 64,
                rights_status="public_redistributable",
            ),
        ),
    )


def _confounded_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [f"shot-{index}" for index in range(10)],
            "source_id": ["synthetic-source"] * 10,
            "source_row": list(range(10)),
            "player_id": ["A"] * 5 + ["B"] * 5,
            "face_angle": [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
            "club_path": [4, 3, 2, 1, 0, 14, 13, 12, 11, 10],
            "ball_speed": [100, 102, 104, 106, 108, 120, 122, 124, 126, 128],
            "monitor_vendor": ["TrackMan"] * 10,
            "monitor_model": ["fixture-comparable"] * 10,
            "software_version": ["fixture-1"] * 10,
        }
    )


def test_separates_pooled_within_between_and_population_effects() -> None:
    result = analyze_player_covariation_v1(
        _confounded_frame(),
        PlayerCovariationRequestV1(
            x_column="face_angle",
            y_column="club_path",
            player_column="player_id",
        ),
        context=_context(),
    )

    assert result.contract_version == PLAYER_COVARIATION_CONTRACT_VERSION
    assert result.status == "available"
    assert result.pooled.pearson_r == pytest.approx(0.8518518519)
    assert result.within_player.pearson_r == pytest.approx(-1.0)
    assert result.within_player.slope == -1.0
    assert result.within_player.intercept == 0.0
    assert result.within_player.ci_lower is None
    assert result.between_player.pearson_r == pytest.approx(1.0)
    assert result.meta_analysis.state == "available"
    assert result.meta_analysis.contributor_count == 2
    assert result.meta_analysis.fixed_effect_r == pytest.approx(-1.0, abs=1e-5)
    assert [item.player_id for item in result.per_player] == ["A", "B"]
    assert sum(item.fixed_weight or 0 for item in result.per_player) == pytest.approx(1)
    assert any("aggregation reversal" in warning for warning in result.warnings)

    player_payload = result.per_player[0].model_dump(mode="json")
    player_payload["random_weight"] = None
    with pytest.raises(ValueError, match="weights must be supplied together"):
        type(result.per_player[0]).model_validate(player_payload)


def test_result_retains_units_vendor_provenance_and_source_joinable_rows() -> None:
    result = analyze_player_covariation_v1(
        _confounded_frame(),
        PlayerCovariationRequestV1(
            x_column="face_angle",
            y_column="club_path",
            player_column="player_id",
        ),
        context=_context(),
    )

    assert result.units["face_angle"].canonical_unit == "rad"
    assert result.units["face_angle"].display_unit == "deg"
    assert result.units["face_angle"].authority == "canonical_registry"
    assert result.vendor_provenance[0].vendor == "TrackMan"
    assert len(result.lineage.backing_records) == 10
    assert {row.source_id for row in result.lineage.backing_records} == {
        "synthetic-source"
    }
    assert all(len(row.record_sha256) == 64 for row in result.lineage.backing_records)
    assert result.claims.causal_inference is False
    assert result.claims.device_emulation is False


def test_population_synthesis_reports_material_heterogeneity() -> None:
    frame = pd.DataFrame(
        {
            "source_id": ["synthetic-source"] * 18,
            "player_id": ["positive"] * 6 + ["mixed"] * 6 + ["negative"] * 6,
            "x": list(range(6)) * 3,
            "y": list(range(6)) + [0, 1, 0, 1, 0, 1] + list(reversed(range(6))),
        }
    )
    result = analyze_player_covariation_v1(
        frame,
        PlayerCovariationRequestV1(
            x_column="x", y_column="y", player_column="player_id"
        ),
        context=_context().model_copy(
            update={"source_units": {"x": "deg", "y": "deg"}}
        ),
    )

    assert result.meta_analysis.state == "available"
    assert result.meta_analysis.q_statistic is not None
    assert result.meta_analysis.q_statistic > 1
    assert result.meta_analysis.tau_squared is not None
    assert result.meta_analysis.tau_squared > 0
    assert result.meta_analysis.i_squared_pct is not None
    assert result.meta_analysis.i_squared_pct > 50


def test_player_analysis_requires_matching_trusted_identity() -> None:
    request = PlayerCovariationRequestV1(
        x_column="face_angle",
        y_column="club_path",
        player_column="player_id",
    )

    with pytest.raises(ValueError, match="trusted player identity"):
        analyze_player_covariation_v1(_confounded_frame(), request)

    mismatched = AnalysisContextV2(
        player_identity=PlayerIdentityV2(
            trust_level="verified_external",
            identifier_column="athlete_id",
            evidence="Joined against the governed participant register.",
        )
    )
    with pytest.raises(ValueError, match="must match"):
        analyze_player_covariation_v1(_confounded_frame(), request, context=mismatched)


def test_missing_constant_and_small_groups_are_structurally_unavailable() -> None:
    frame = pd.DataFrame(
        {
            "source_id": ["synthetic-source"] * 12,
            "player_id": ["good"] * 4 + ["small"] * 3 + ["constant"] * 4 + [""],
            "x": [1, 2, 3, 4, 1, 2, np.nan, 5, 5, 5, 5, 9],
            "y": [2, 4, 6, 8, 2, 4, 6, 1, 2, 3, 4, np.inf],
        }
    )
    context = _context().model_copy(update={"source_units": {"x": "mph", "y": "deg"}})

    result = analyze_player_covariation_v1(
        frame,
        PlayerCovariationRequestV1(
            x_column="x", y_column="y", player_column="player_id"
        ),
        context=context,
    )

    states = {item.player_id: item.estimate.reason_code for item in result.per_player}
    assert states == {
        "constant": "constant_x",
        "good": None,
        "small": "insufficient_samples",
    }
    assert result.meta_analysis.state == "unavailable"
    assert result.meta_analysis.reason_code == "insufficient_eligible_players"
    assert result.missingness.missing_by_variable["x"] == 1
    assert result.missingness.non_finite_by_variable["y"] == 1
    assert result.missingness.excluded_by_reason["blank_player_identity"] == 1
    assert result.status == "partial"
    assert any(
        item.result_path == "meta_analysis" and item.state == "unavailable"
        for item in result.availability
    )
    unavailable = next(item for item in result.per_player if item.player_id == "small")
    unavailable_payload = unavailable.model_dump(mode="json")
    unavailable_payload.update({"fixed_weight": 0.5, "random_weight": 0.5})
    with pytest.raises(ValueError, match="unavailable player"):
        type(unavailable).model_validate(unavailable_payload)


def test_pair_scan_is_deterministic_and_carries_multiplicity_boundary() -> None:
    frame = _confounded_frame().assign(constant_metric=1.0)
    result = scan_player_covariation_v1(
        frame,
        PlayerCovariationScanRequestV1(
            player_column="player_id",
            numeric_columns=(
                "club_path",
                "ball_speed",
                "face_angle",
                "constant_metric",
            ),
        ),
        context=_context(),
    )

    assert result.pair_count == 6
    assert result.ranking[0].rank == 1
    assert result.ranking[0].x_column == "ball_speed"
    assert result.ranking[0].y_column == "club_path"
    assert result.ranking[0].random_effect_r == pytest.approx(-1.0, abs=1e-5)
    assert result.ranking[0].input_row_count == len(frame)
    assert result.ranking[0].pairwise_complete_row_count == len(frame)
    assert result.ranking[0].excluded_row_count == 0
    assert result.ranking[-1].state == "unavailable"
    assert len(result.lineage.backing_records) == len(frame)
    assert any("multiplicity" in warning.lower() for warning in result.warnings)
    assert any("exploratory" in warning.lower() for warning in result.warnings)


def test_pair_scan_contract_rejects_inconsistent_states_and_counts() -> None:
    result = scan_player_covariation_v1(
        _confounded_frame(),
        PlayerCovariationScanRequestV1(
            player_column="player_id",
            numeric_columns=("club_path", "ball_speed", "face_angle"),
        ),
        context=_context(),
    )
    payload = result.model_dump(mode="json")
    payload["available_pair_count"] = 0
    with pytest.raises(ValueError, match="pair counts"):
        type(result).model_validate(payload)

    rank_payload = result.ranking[0].model_dump(mode="json")
    rank_payload["reason_code"] = "insufficient_eligible_players"
    with pytest.raises(ValueError, match="available ranked pair"):
        type(result.ranking[0]).model_validate(rank_payload)


def test_default_scan_excludes_numeric_source_structure() -> None:
    result = scan_player_covariation_v1(
        _confounded_frame(),
        PlayerCovariationScanRequestV1(player_column="player_id"),
        context=_context(),
    )

    selected = {item.x_column for item in result.ranking} | {
        item.y_column for item in result.ranking
    }
    assert selected == {"ball_speed", "club_path", "face_angle"}
    assert "source_row" not in selected


def test_contract_schema_and_consumer_golden_are_fresh() -> None:
    schema = player_covariation_contract_json_schema()
    assert schema["title"] == "PlayerCovariationContractV1"
    assert "PlayerCovariationResultV1" in schema["$defs"]
    assert "PlayerCovariationScanResultV1" in schema["$defs"]
    assert GOLDEN_PATH.is_file()
    committed = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert committed == player_covariation_golden_fixture()
    assert committed["expected_result"]["analysis_kind"] == "selected_pair"
    assert committed["expected_scan_result"]["analysis_kind"] == "pair_scan"
    assert committed["expected_scan_result"]["pair_count"] == 3


def test_generic_analysis_contract_versions_remain_unchanged() -> None:
    assert CONTRACT_VERSION == "1.0.0"
    assert CONTRACT_VERSION_V2 == "2.0.0"

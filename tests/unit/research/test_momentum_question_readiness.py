from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.momentum_question_readiness import (
    build_readiness_audit,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"


def _load(name: str) -> dict:
    return json.loads((ARTICLE / "data" / name).read_text(encoding="utf-8"))


def test_repository_agenda_has_complete_point_and_experiment_coverage() -> None:
    result = build_readiness_audit(
        _load("momentum_transfer_question_registry.json"),
        _load("momentum_transfer_experiment_registry.json"),
    )
    assert result["summary"]["critical_point_count"] == 9
    assert result["summary"]["model_plan_registered_for_all"] is True
    assert result["summary"]["human_plan_registered_for_all"] is True
    assert result["summary"]["human_execution_blocked"] is True
    assert result["summary"]["unresolved_point_ids"] == ["MTQ-06"]
    assert set(result["question_coverage"]) == {f"Q{i}" for i in range(1, 8)}
    assert all(result["question_coverage"].values())
    assert all(point["falsifier"] for point in result["critical_points"])
    assert all(point["evidence_artifacts"] for point in result["critical_points"])


def test_agenda_fails_closed_when_a_source_point_is_missing() -> None:
    registry = _load("momentum_transfer_question_registry.json")
    registry["critical_points"] = registry["critical_points"][:-1]
    with pytest.raises(ValueError, match="MTQ-01 through MTQ-09"):
        build_readiness_audit(
            registry, _load("momentum_transfer_experiment_registry.json")
        )


def test_agenda_fails_closed_for_mislinked_experiment() -> None:
    registry = _load("momentum_transfer_question_registry.json")
    registry["critical_points"][0]["experiment_ids"] = ["MT-E06", "MT-H01"]
    with pytest.raises(ValueError, match="does not cover"):
        build_readiness_audit(
            registry, _load("momentum_transfer_experiment_registry.json")
        )

"""Keep the narrative falsification matrix and typed predictions aligned."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from src.shared.python.biomechanics.interaction_evidence import (
    load_evidence_manifest,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"


def test_narrative_hypotheses_have_exactly_one_typed_prediction() -> None:
    matrix = (ARTICLE / "MODEL_COMPLETION_FALSIFICATION_MATRIX.md").read_text(
        encoding="utf-8"
    )
    narrative_ids = set(re.findall(r"\| H(\d+):", matrix))
    manifest = load_evidence_manifest(
        ARTICLE / "data/model_completion_predictions.json"
    )
    typed_ids = [
        prediction.hypothesis_id.removeprefix("H")
        for prediction in manifest.predictions
    ]

    assert narrative_ids == {str(value) for value in range(1, 12)}
    assert set(typed_ids) == narrative_ids
    assert len(typed_ids) == len(set(typed_ids))


def test_typed_predictions_retain_falsifiability_and_boundaries() -> None:
    record = json.loads(
        (ARTICLE / "data/model_completion_predictions.json").read_text(encoding="utf-8")
    )
    by_id = {row["hypothesis_id"]: row for row in record["predictions"]}

    assert by_id["H9"]["status"] == "contradicted"
    assert by_id["H10"]["status"] == "inconclusive"
    for row in record["predictions"]:
        assert row["falsifier"]
        assert row["competing_explanations"]
        assert row["negative_controls"]
        assert row["status_scope"]
        assert row["remaining_gate"]

"""Deterministic rendering checks for structural-factorial reviewer evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from scripts.research.proximal_distal_energy import (
    make_articulated_structural_factorial_figure as figure_module,
)

pytestmark = pytest.mark.scientific


def _aggregate(contrast_id: str, values: list[float]) -> dict[str, object]:
    count = len(values)
    if values:
        ordered = sorted(values)
        median = ordered[len(ordered) // 2]
        effect = {
            "minimum": min(values),
            "median": median,
            "maximum": max(values),
        }
    else:
        effect = {"minimum": None, "median": None, "maximum": None}
    return {
        "contrast_id": contrast_id,
        "estimand_class": "primary",
        "order": 1,
        "outcome": "final_club_translation_speed_m_s",
        "expected_block_count": 2,
        "eligible_block_count": count,
        "missing_block_count": 2 - count,
        "support_fraction": count / 2,
        "exact_sign_counts": {
            "negative": sum(value < 0.0 for value in values),
            "zero": sum(value == 0.0 for value in values),
            "positive": sum(value > 0.0 for value in values),
        },
        "sign_reversal": min(values) < 0.0 < max(values) if values else False,
        "walsh_coefficient": effect,
        "high_minus_low_effect": effect,
    }


def _summary() -> dict[str, object]:
    return {
        "schema_version": figure_module.SCHEMA,
        "contrast_aggregates": [
            _aggregate("shaft_bending", [-1.0, 2.0]),
            _aggregate("shaft_torsion", []),
        ],
        "factorial_contrasts": [
            {
                "contrast_id": "shaft_bending",
                "outcome": "final_club_translation_speed_m_s",
                "block": [0, 0, 1.0, 0.001, "mujoco", 0.05],
                "high_minus_low_effect": -1.0,
            },
            {
                "contrast_id": "shaft_bending",
                "outcome": "final_club_translation_speed_m_s",
                "block": [1, 0, 1.0, 0.001, "mujoco", 0.05],
                "high_minus_low_effect": 2.0,
            },
        ],
    }


def test_load_summary_rejects_an_old_or_incomplete_schema(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"schema_version": "old"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        figure_module.load_summary(path)


def test_render_figure_retains_sign_reversal_and_missing_support(
    tmp_path: Path,
) -> None:
    output = tmp_path / "structural-factorial"

    figure_module.render_figure(
        _summary(), outcome="final_club_translation_speed_m_s", output=output
    )

    assert output.with_suffix(".pdf").stat().st_size > 1_000
    svg = output.with_suffix(".svg").read_text(encoding="utf-8")
    assert "Structural Pathway Factorial: Effects, Direction, and Support" in svg
    assert "Missing Support" in svg
    assert svg.endswith("\n")
    plt.close("all")

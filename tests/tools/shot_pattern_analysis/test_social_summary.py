"""Contracts for the Facebook-sized conditional scoring summary."""

import json
from pathlib import Path

import pytest

try:
    import matplotlib
    import matplotlib.image as mpimg
except (ImportError, OSError) as exc:
    pytest.skip(
        f"optional plotting dependency unavailable: {exc}", allow_module_level=True
    )

matplotlib.use("Agg")

from docs.research.shot_pattern_analysis.phase2.presentation_v2 import summary

pytestmark = pytest.mark.unit

TEX = summary.Path(__file__).resolve().parents[3] / (
    "docs/research/shot_pattern_analysis/shot_pattern_analysis.tex"
)
COLORS = {"Straight": "#2563a6", "Draw": "#c76a23", "Fade": "#804c9e"}


def test_social_summary_writes_exact_facebook_preview_dimensions(tmp_path):
    destination = tmp_path / "shot-pattern-facebook.png"

    result = summary.render_social_summary(TEX, destination, COLORS)

    assert result == destination
    assert result.exists()
    assert mpimg.imread(result).shape[:2] == (630, 1200)


def drifted_article(tmp_path: Path) -> Path:
    """An article path whose headline dataset moves the Driver Draw maximum."""
    dataset = json.loads((TEX.parent / summary.HEADLINE_DATASET).read_text("utf-8"))
    for cell in dataset["cells"]:
        if cell["key"] == "C07":
            cell["paired_benefit_vs_straight"]["Draw"]["estimate"] += 1e-6
    target = tmp_path / summary.HEADLINE_DATASET
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(dataset), encoding="utf-8")
    return tmp_path / "article.tex"


def test_social_summary_rejects_ranges_that_drift_from_the_tex_table(tmp_path):
    modified = drifted_article(tmp_path)
    destination = tmp_path / "must-not-exist.png"

    with pytest.raises(ValueError, match="summary range differs"):
        summary.render_social_summary(modified, destination, COLORS)

    assert not destination.exists()

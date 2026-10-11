"""Contracts for the Facebook-sized conditional scoring summary."""

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


def test_social_summary_rejects_ranges_that_drift_from_the_tex_table(tmp_path):
    source = TEX.read_text(encoding="utf-8")
    modified = tmp_path / "modified.tex"
    modified.write_text(
        source.replace("+0.007797655", "+0.007797656"), encoding="utf-8"
    )
    destination = tmp_path / "must-not-exist.png"

    with pytest.raises(ValueError, match="summary range differs"):
        summary.render_social_summary(modified, destination, COLORS)

    assert not destination.exists()

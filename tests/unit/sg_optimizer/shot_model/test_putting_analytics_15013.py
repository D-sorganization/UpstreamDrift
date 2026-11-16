"""Regression tests for issue #15013 — SG putting docs and model boundaries."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from src.shared.python.putting_validation.reference_data import get_series
from src.shared.python.sg_optimizer.shot_model import putting as putting_mod
from src.shared.python.sg_optimizer.shot_model.putting import _baseline_make

_REPO_ROOT = Path(__file__).resolve().parents[4]

pytestmark = pytest.mark.unit

# Tour logistic calibrated to 50 % at 8 ft and 7 % at 30 ft (see putting_validation registry).
_BASELINE_SERIES = get_series("sg_optimizer_documented_anchors")
_BROADIE_2011_URL = "https://www.columbia.edu/~mnb2/broadie/Assets/strokes_gained_pga_broadie_20110408.pdf"


def test_sg_logistic_matches_documented_anchors() -> None:
    """Fitted curve matches the registry (not the retired 99 % / 6.0 − 2.85 ln d comment)."""
    for distance_ft, expected in zip(
        _BASELINE_SERIES.distances_ft,
        _BASELINE_SERIES.values,
        strict=True,
    ):
        assert _baseline_make(distance_ft) == pytest.approx(expected, rel=0, abs=1e-12)
    assert _baseline_make(8.0) == pytest.approx(0.50, abs=1e-12)
    assert _baseline_make(30.0) == pytest.approx(0.07, abs=1e-12)


def test_putting_module_documents_shot_pattern_as_separate_broadie_curve() -> None:
    doc = inspect.getdoc(putting_mod) or ""
    assert "shot_pattern_analysis" in doc
    assert _BROADIE_2011_URL in doc


def test_shot_pattern_scoring_cites_same_broadie_source() -> None:
    scoring_path = _REPO_ROOT / "src/tools/shot_pattern_analysis/scoring.py"
    text = scoring_path.read_text(encoding="utf-8")
    assert _BROADIE_2011_URL in text
    assert "Equation" in text or "Eq" in text


def test_green_terminal_and_expected_strokes_docs_state_scope() -> None:
    strategy_src = (
        _REPO_ROOT / "src/shared/python/sg_optimizer/strategy.py"
    ).read_text(encoding="utf-8")
    vi_src = (
        _REPO_ROOT / "src/shared/python/sg_optimizer/mdp/value_iteration.py"
    ).read_text(encoding="utf-8")
    assert "strokes to reach the green" in strategy_src.lower()
    assert "strokes to reach the green" in vi_src.lower()

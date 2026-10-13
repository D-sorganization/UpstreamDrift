"""Keep the club marker acquisition requirements doc in sync with the code (#10382)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_matching import club_face_target as cft
from src.shared.python.motion_matching import club_observability as obs
from src.shared.python.motion_matching import output_policy

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "motion_capture" / "club_marker_acquisition_requirements.md"

pytestmark = pytest.mark.gate


def _text() -> str:
    assert DOC.is_file(), f"missing governed document: {DOC.relative_to(ROOT)}"
    return DOC.read_text(encoding="utf-8")


def _reason_names() -> list[str]:
    return sorted(n for n in obs.__all__ if n.startswith("REASON_"))


def test_every_reason_constant_is_documented_by_name_and_text() -> None:
    """Each REASON_* constant and its user-facing text appears in the doc."""
    text = _text()
    names = _reason_names()
    assert len(names) >= 6
    for name in names:
        assert f"`{name}`" in text, f"{name} missing from the acquisition doc"
        assert getattr(obs, name) in text, f"text of {name} missing from the doc"


def test_threshold_and_contract_constants_are_named_with_source() -> None:
    """Limits are cited by constant name and source module, never by number alone."""
    text = _text()
    for name in (
        "MIN_TRIAD_SPREAD_M",
        "MAX_TRIAD_SHAPE_RESIDUAL_M",
        "HEAD_TRIAD_LABELS",
    ):
        assert hasattr(cft, name)
        assert f"`{name}`" in text
    assert "club_face_target.py" in text
    assert "club_observability.py" in text


def test_head_triad_labels_and_outputs_are_documented() -> None:
    """The required head markers and the output names track the code."""
    text = _text()
    for label in cft.HEAD_TRIAD_LABELS:
        assert label in text
    for output in output_policy.ADVERTISED_OUTPUTS + output_policy.WITHHELD_OUTPUTS:
        assert f"`{output}`" in text
    for term in ("non-collinear", "calibration", "evidence", "frame coverage"):
        assert term in text.casefold()

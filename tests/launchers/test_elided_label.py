"""ElidedLabel shows an ellipsis at narrow widths and keeps the full text."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

try:
    from src.launchers.elided_label import ElidedLabel
except (ImportError, OSError) as exc:  # PyQt6 missing or not loadable
    pytest.skip(f"PyQt6 not loadable: {exc}", allow_module_level=True)

NAME = "Experimental Spine Fit Long Name"


def test_narrow_width_elides_and_keeps_full_text(qapp) -> None:
    label = ElidedLabel(NAME)
    label.resize(60, 20)
    label.show()
    assert label.text().endswith("…")
    assert len(label.text()) < len(NAME)
    assert label.fullText() == NAME
    assert label.toolTip() == NAME


def test_wide_width_shows_full_text(qapp) -> None:
    label = ElidedLabel(NAME)
    label.resize(label.sizeHint().width() + 20, 20)
    label.show()
    assert label.text() == NAME


def test_min_width_does_not_force_full_text(qapp) -> None:
    label = ElidedLabel(NAME)
    assert label.minimumSizeHint().width() < label.sizeHint().width()


def test_set_text_updates_full_text(qapp) -> None:
    label = ElidedLabel("a")
    label.setText("b")
    assert label.fullText() == "b"

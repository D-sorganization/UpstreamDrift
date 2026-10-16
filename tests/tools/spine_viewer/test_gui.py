"""Offscreen GUI tests for the Spine Viewer (SPN-15b, #12772)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.unit


def _load_widget_class():
    try:
        import matplotlib  # noqa: F401
        import PyQt6.QtWidgets  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6/matplotlib not loadable: {exc}")
    from src.tools.spine_viewer.gui import SpineViewerWidget

    return SpineViewerWidget


@pytest.fixture
def qapp():
    _load_widget_class()
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_empty_widget_shows_no_data_not_qualified(qapp) -> None:
    widget = _load_widget_class()()
    assert "no study" in widget.status_text().lower()
    assert "QUALIFIED" not in widget.status_text().replace("UNQUALIFIED", "")
    widget.cleanup()


def test_loaded_unqualified_study_is_prominent(qapp, study_factory) -> None:
    from .conftest import SOURCE

    widget = _load_widget_class()()
    widget.load_study(study_factory())
    assert "UNQUALIFIED" in widget.status_text()
    assert SOURCE in widget.distribution_text()
    assert widget.current_quantity() == "compression"
    widget.set_quantity("ap_shear")
    assert widget.current_quantity() == "ap_shear"
    assert widget.current_figure() is not None
    widget.cleanup()


def test_bad_receipt_shows_error_and_clears_data(qapp, study_factory) -> None:
    widget = _load_widget_class()()
    widget.load_study(study_factory())
    widget.load_study(widget_bad := study_factory().parent / "missing.json")
    assert widget_bad
    assert "ERROR" in widget.status_text()
    assert widget.current_figure() is None
    widget.cleanup()


def test_cleanup_is_idempotent(qapp, study_factory) -> None:
    widget = _load_widget_class()()
    widget.load_study(study_factory())
    widget.cleanup()
    widget.cleanup()

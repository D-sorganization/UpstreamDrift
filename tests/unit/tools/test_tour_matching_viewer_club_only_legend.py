"""CO-09: observed-versus-inferred legend on Tour Matching Viewer."""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("PyQt6", reason="tour matching viewer needs PyQt6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.motion_matching import club_only_ui as cui  # noqa: E402
from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv[:1])
    return app


def test_viewer_exposes_observed_versus_inferred_legend(qapp: QApplication) -> None:
    widget = TourMatchingViewerWidget()
    legend = widget.observed_versus_inferred_legend()
    assert legend == cui.observed_versus_inferred_legend()
    assert widget.legend_label is not None
    text = widget.legend_label.text().lower()
    assert "observed" in text
    assert "inferred" in text or "plausible" in text
    widget.set_club_only_legend_visible(True)
    assert widget.legend_label.isVisibleTo(widget) is True
    widget.set_club_only_legend_visible(False)

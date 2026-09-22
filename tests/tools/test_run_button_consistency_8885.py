"""The primary "run" action shares one colour across tools (issue #8885).

Before this change the four tools below each hardcoded a different literal
for their primary action button: putting_green_gui (#2E7D32),
ball_flight_gui (#1565C0), swing_flight_pipeline (#4CAF50), and
training_controller (#0A84FF). They now all source
``PRIMARY_ACTION_COLOR`` from the shared
``src.shared.python.theme.tool_stylesheet`` module.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
import pytest

from src.shared.python.theme.tool_stylesheet import PRIMARY_ACTION_COLOR

pytestmark = [pytest.mark.unit, pytest.mark.ui]

_STALE_RUN_BUTTON_LITERALS = ("#2E7D32", "#1565C0", "#4CAF50")

_APP: QApplication | None = None


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app
    return app


@pytest.fixture(autouse=True)
def _qapp() -> None:
    _ensure_qapp()


def test_putting_green_run_button_uses_the_shared_colour() -> None:
    from src.tools.putting_green_gui.gui import PuttingGreenWidget

    widget = PuttingGreenWidget()
    sheet = widget._run_btn.styleSheet()
    assert PRIMARY_ACTION_COLOR in sheet
    widget.deleteLater()


def test_ball_flight_run_button_uses_the_shared_colour() -> None:
    from src.tools.ball_flight_gui.gui import BallFlightWidget

    widget = BallFlightWidget()
    sheet = widget._run_btn.styleSheet()
    assert PRIMARY_ACTION_COLOR in sheet
    widget.deleteLater()


def test_swing_flight_pipeline_run_button_uses_the_shared_colour() -> None:
    from src.tools.swing_flight_pipeline.gui import SwingFlightWidget

    widget = SwingFlightWidget()
    sheet = widget._run_btn.styleSheet()
    assert PRIMARY_ACTION_COLOR in sheet
    widget.deleteLater()


def test_training_controller_submit_button_uses_the_shared_colour() -> None:
    from src.tools.training_controller.gui import DARK_STYLE

    assert PRIMARY_ACTION_COLOR in DARK_STYLE


def test_no_stale_per_tool_run_button_literals_remain() -> None:
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    offenders = []
    for rel in (
        "src/tools/putting_green_gui/gui.py",
        "src/tools/ball_flight_gui/gui.py",
        "src/tools/swing_flight_pipeline/gui.py",
    ):
        text = (repo_root / rel).read_text(encoding="utf-8")
        for literal in _STALE_RUN_BUTTON_LITERALS:
            if literal in text:
                offenders.append(f"{rel}: {literal}")
    assert offenders == []

"""Offscreen: ball, ball flight, aim line and shot shape of live playback (#15150).

The panel plays a small cut of the committed demo bundle. The flight model is
reported missing, so the flight and aim-line boxes must be disabled with a
tooltip saying why while the ball at address still draws.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
mujoco = pytest.importorskip("mujoco", reason="golfer playback needs mujoco")
pytest.importorskip("PyQt6", reason="the panel needs PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.motion_matching.same_input.bundle import (  # noqa: E402
    InputBundle,
)
from src.tools.model_explorer.default_match import DEMO_BUNDLE  # noqa: E402
from src.tools.model_explorer.results_playback_panel import (  # noqa: E402
    ResultsPlaybackPanel,
)
from src.tools.model_explorer.results_session import ResultsSession  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ui]

STRIDE = 40
REASON = "the upstream_physics Rust flight wheel is missing or out of date"
WAIT_S = 120.0


def _require_offscreen_gl() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    try:
        mujoco.Renderer(model, 16, 16).close()
    except Exception as exc:  # noqa: BLE001 - any GL backend failure means skip
        pytest.skip(f"no offscreen OpenGL for mujoco.Renderer: {exc}")


@pytest.fixture(scope="module")
def panel(tmp_path_factory: pytest.TempPathFactory) -> Generator[Any, None, None]:
    _require_offscreen_gl()
    app = QApplication.instance() or QApplication(["ball"])
    assert isinstance(app, QApplication)
    demo = InputBundle.load(DEMO_BUNDLE)
    q = np.asarray(demo.reference_q[::STRIDE], dtype=float)
    rest = np.zeros_like(q)
    order, dt_s = demo.coordinate_order, float(demo.dt_s) * STRIDE
    path = tmp_path_factory.mktemp("ball") / "demo_small.npz"
    InputBundle(
        demo.spec_bytes, order, dt_s, q[0], rest[0], rest[1:], q, rest, "mujoco"
    ).save(path)
    target = "src.tools.model_explorer.results_playback_panel.flight_unavailable_reason"
    with patch(target, return_value=REASON):
        widget = ResultsPlaybackPanel(ResultsSession.load(Path(path)))
    yield widget
    widget.stop()
    widget.deleteLater()


def test_flight_controls_are_disabled_with_the_reason(panel: Any) -> None:
    for box in (panel.flight_check, panel.aim_check):
        assert not box.isEnabled() and REASON in box.toolTip()
    panel.set_ball_flight_visible(True)
    panel.set_aim_line_visible(True)
    assert not panel.flight_check.isChecked() and not panel.aim_check.isChecked()
    assert [panel.shape_combo.itemText(i) for i in range(3)] == [
        "straight",
        "draw",
        "fade",
    ]
    with pytest.raises(ValueError, match="shot shape"):
        panel.set_shot_shape("hook")


def test_ball_at_address_draws_and_hides(panel: Any) -> None:
    layout = panel.wait_for_shot(WAIT_S)
    assert layout.ball is not None, layout.note
    assert panel.ball_status.text() == layout.note
    panel.show_frame(0)
    panel.set_ball_visible(True)
    with_ball = panel.current_image()
    panel.set_ball_visible(False)
    assert not np.array_equal(with_ball, panel.current_image())

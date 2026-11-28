"""Offscreen: the club-speed plot shows the Shot card's impact speed (#15150).

The card reads the club speed where the face meets the ball (a fit of the
samples before contact); the plotted curve is a central difference that
straddles the strike. The plot must title and mark the card's number so the
two never disagree.
"""

from __future__ import annotations

import math
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
from src.tools.model_explorer.golfer_playback import CLUB_HEAD_SPEED  # noqa: E402
from src.tools.model_explorer.playback_view import (  # noqa: E402
    load_club_speed_plot_settings,
)
from src.tools.model_explorer.results_playback_panel import (  # noqa: E402
    ResultsPlaybackPanel,
)
from src.tools.model_explorer.results_session import ResultsSession  # noqa: E402
from src.tools.model_explorer.shot_card_rows import ShotSummary  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ui]

STRIDE = 40


def _require_offscreen_gl() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    try:
        mujoco.Renderer(model, 16, 16).close()
    except Exception as exc:  # noqa: BLE001 - any GL backend failure means skip
        pytest.skip(f"no offscreen OpenGL for mujoco.Renderer: {exc}")


@pytest.fixture(scope="module")
def panel(tmp_path_factory: pytest.TempPathFactory) -> Generator[Any, None, None]:
    _require_offscreen_gl()
    app = QApplication.instance() or QApplication(["speed"])
    assert isinstance(app, QApplication)
    demo = InputBundle.load(DEMO_BUNDLE)
    q = np.asarray(demo.reference_q[::STRIDE], dtype=float)
    rest = np.zeros_like(q)
    order, dt_s = demo.coordinate_order, float(demo.dt_s) * STRIDE
    path = tmp_path_factory.mktemp("speed") / "demo_small.npz"
    InputBundle(
        demo.spec_bytes, order, dt_s, q[0], rest[0], rest[1:], q, rest, "mujoco"
    ).save(path)
    target = "src.tools.model_explorer.results_playback_panel.flight_unavailable_reason"
    with patch(target, return_value=None):
        widget = ResultsPlaybackPanel(ResultsSession.load(Path(path)))
    yield widget
    widget.stop()
    widget.deleteLater()


def _marker(panel: Any) -> tuple[float, float] | None:
    for line in panel._axes.get_lines():
        if line.get_gid() == "impact_speed":
            x, y = line.get_data()
            return float(x[0]), float(y[0])
    return None


def test_the_plot_shows_the_card_impact_speed(panel: Any) -> None:
    if CLUB_HEAD_SPEED not in panel.series:
        pytest.skip("the demo specification has no club to measure")
    cfg = load_club_speed_plot_settings()
    panel.select_channel(CLUB_HEAD_SPEED)
    assert panel._axes.get_title().startswith("Peak") and _marker(panel) is None
    t = float(panel._pose.time_s[len(panel._pose.time_s) // 2])
    panel.show_shot(ShotSummary("at impact", club_speed_mph=114.7, impact_time_s=t))
    assert panel._axes.get_title() == cfg.impact_title.format(speed=114.7, time=t)
    assert _marker(panel) == (t, 114.7)
    panel.show_shot(ShotSummary("no impact detected"))  # no number: back to peak
    assert panel._axes.get_title().startswith("Peak") and _marker(panel) is None


def test_show_shot_rejects_a_non_finite_impact(panel: Any) -> None:
    bad = ShotSummary("at impact", club_speed_mph=114.7, impact_time_s=math.nan)
    with pytest.raises(ValueError, match="finite"):
        panel.show_shot(bad)

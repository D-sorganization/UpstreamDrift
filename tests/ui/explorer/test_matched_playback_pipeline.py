"""Offscreen pipeline: launcher -> Model Explorer -> matched swing (#15150).

Open the launcher, open the Model Explorer tile (it loads the default matched
swing itself), play frames, switch scenario, toggle two features and grab a
screenshot per step. CI uses two small bundles cut from the committed demo
bundle; a user who configured real matched swings runs the flow on those.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
mujoco = pytest.importorskip("mujoco", reason="golfer playback needs mujoco")
pytest.importorskip("PyQt6", reason="the launcher needs PyQt6")

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from src.shared.python.motion_matching.same_input.bundle import (  # noqa: E402
    InputBundle,
)
from src.tools.model_explorer.default_match import (  # noqa: E402
    BUNDLE_ENV,
    CONFIG_ENV,
    DEMO_BUNDLE,
    match_scenarios,
)
from src.tools.model_explorer.results_playback_panel import (  # noqa: E402
    ResultsPlaybackPanel,
)

pytestmark = [pytest.mark.integration, pytest.mark.ui, pytest.mark.slow]

#: Keep every n-th demo sample so the CI bundles stay small and fast.
DEMO_STRIDE = 12
#: The second scenario is the demo's last part (downswing and finish).
SECOND_SCENARIO_START = 0.5
LOAD_DEADLINE_S = 240.0
PLAY_SECONDS = 0.5


def _require_offscreen_gl() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    try:
        mujoco.Renderer(model, 16, 16).close()
    except Exception as exc:  # noqa: BLE001 - any GL backend failure means skip
        pytest.skip(f"no offscreen OpenGL for mujoco.Renderer: {exc}")


def _cut(source: InputBundle, start: int, out: Path) -> Path:
    """Save every ``DEMO_STRIDE``-th sample of ``source`` from ``start``."""
    q = np.asarray(source.reference_q[start::DEMO_STRIDE], dtype=float)
    v, dt_s = np.zeros_like(q), float(source.dt_s) * DEMO_STRIDE
    efforts = np.zeros((len(q) - 1, q.shape[1]))
    order, engine = source.coordinate_order, source.reference_engine
    bundle = InputBundle(
        source.spec_bytes, order, dt_s, q[0], v[0], efforts, q, v, engine
    )
    bundle.save(out)
    return out


@pytest.fixture(scope="module")
def user_setting(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[list[str], None, None]:
    """Scenario labels the explorer will list; synthetic unless configured."""
    real = [s for s in match_scenarios() if not s.is_demo]
    if len(real) >= 2:
        yield [s.label for s in real]
        return
    root = tmp_path_factory.mktemp("matched")
    demo = InputBundle.load(DEMO_BUNDLE)
    second_start = int(len(demo.reference_q) * SECOND_SCENARIO_START)
    driver = _cut(demo, 0, root / "driver_demo_small.npz")
    finish = _cut(demo, second_start, root / "finish_demo_small.npz")
    config = root / "matched_swings.json"
    scenarios = {"Driver (CI)": str(driver), "Finish (CI)": str(finish)}
    config.write_text(json.dumps({"scenarios": scenarios}))
    with patch.dict(os.environ, {BUNDLE_ENV: str(driver), CONFIG_ENV: str(config)}):
        yield [s.label for s in match_scenarios()]


@pytest.fixture(scope="module")
def launcher(user_setting: list[str]) -> Generator[Any, None, None]:
    _require_offscreen_gl()
    app = QApplication.instance() or QApplication(["pipeline"])
    assert isinstance(app, QApplication)
    from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

    with patch.object(QMessageBox, "warning", return_value=0):
        window = UpstreamDriftLauncher()
        window.resize(1280, 860)
        window.show()
        app.processEvents()
        yield window
        window.hide()
        window.deleteLater()


def _grab(widget: Any, folder: Path, name: str) -> None:
    path = folder / f"{name}.png"
    assert widget.grab().save(str(path)) and path.stat().st_size > 0, path


def _explorer_tab(launcher: Any) -> Any:
    tabs = launcher.workspace_tabs
    for index in range(tabs.count()):
        if tabs.tabText(index) == "Model Explorer":
            return tabs.widget(index)
    raise AssertionError("the Model Explorer tab did not open")


def _wait_for_results(explorer: Any) -> None:
    """Spin events until the queued background load has started and finished."""
    deadline = time.monotonic() + LOAD_DEADLINE_S
    while explorer.results_frame_count == 0 and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.05)
    assert explorer.results_frame_count > 0, "default matched swing never loaded"
    assert explorer.wait_for_results_load(LOAD_DEADLINE_S), "load did not finish"


def test_launcher_opens_explorer_plays_and_switches_matched_swings(
    launcher: Any, user_setting: list[str], tmp_path: Path
) -> None:
    shots = tmp_path
    _grab(launcher, shots, "01_launcher")
    launcher.launch_model_direct("model_explorer")
    explorer = _explorer_tab(launcher)
    _wait_for_results(explorer)
    panel = explorer.playback_panel
    assert panel is not None
    assert panel.title_label.text().startswith(user_setting[0])
    _grab(launcher, shots, "02_default_match")

    # Play several frames: scrub, then let the timeline run.
    first = panel.current_image()
    count = panel.session.frame_count
    for index in (count // 4, count // 2, (3 * count) // 4):
        panel.show_frame(index)
    assert not np.array_equal(first, panel.current_image())
    panel.show_frame(0)
    panel.play_button.setChecked(True)
    stop = time.monotonic() + PLAY_SECONDS
    while time.monotonic() < stop:
        QApplication.processEvents()
        time.sleep(0.01)
    panel.play_button.setChecked(False)
    assert panel.current_frame > 0, "timeline did not advance while playing"
    _grab(launcher, shots, "03_played")

    # Switch scenario through File > Open Matched Swing.
    actions = explorer.matched_swing_menu.actions()
    assert [a.text() for a in actions] == user_setting
    before = (panel.title_label.text(), count)
    actions[1].trigger()
    assert explorer.is_loading_results, "the switch must load in the background"
    assert explorer.wait_for_results_load(LOAD_DEADLINE_S)
    panel = explorer.playback_panel
    assert panel.title_label.text().startswith(user_setting[1])
    assert (panel.title_label.text(), panel.session.frame_count) != before
    _grab(launcher, shots, "04_second_scenario")

    # Feature 1: camera view changes the rendered golfer.
    # It opens behind the golfer (the default_view setting, #15150).
    assert panel.view_combo.currentText() == "down_the_line"
    panel.show_frame(panel.session.frame_count // 2)
    behind = panel.current_image()
    panel.view_combo.setCurrentText("face_on")
    assert not np.array_equal(behind, panel.current_image())
    # Feature 2: the plotted channel changes the plot axis.
    joint = panel.channel_combo.itemText(panel.channel_combo.count() - 1)
    panel.select_channel(joint)
    assert joint in panel.canvas.fig.axes[0].get_ylabel()
    _grab(launcher, shots, "05_features")
    explorer.cleanup()


def test_live_playback_offers_ball_flight_and_shot_shape() -> None:
    for name in ("set_ball_flight_visible", "set_shot_shape"):
        assert callable(getattr(ResultsPlaybackPanel, name, None)), name

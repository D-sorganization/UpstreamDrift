"""Offscreen: Results Playback gives the swing the window (#15150).

Playing a matched swing hides the segment editor and Properties docks and
returns them when the user goes back to the 3D Viewport tab; inside the panel
the render gets most of the height, in the ratio the settings file names.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Generator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
mujoco = pytest.importorskip("mujoco", reason="golfer playback needs mujoco")
pytest.importorskip("PyQt6", reason="explorer needs PyQt6")

from PyQt6.QtWidgets import QApplication, QSplitter  # noqa: E402

from src.tools.model_explorer.gui import MainWidget  # noqa: E402
from src.tools.model_explorer.playback_view import load_layout_settings  # noqa: E402

pytestmark = [pytest.mark.ui, pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
FIXTURES = REPO_ROOT / "tests/fixtures/results_playback"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    try:
        mujoco.Renderer(model, 16, 16).close()
    except Exception as exc:  # noqa: BLE001 - any GL backend failure means skip
        pytest.skip(f"no offscreen OpenGL for mujoco.Renderer: {exc}")
    inst = QApplication.instance()
    return inst if isinstance(inst, QApplication) else QApplication(["explorer"])


@pytest.fixture(scope="module")
def fit(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("layout") / "mujoco_fit"
    shutil.copytree(FIXTURES / "mujoco_fit", root)
    (root / "full_body_spec_hipcal_scaled.json").write_bytes(SPEC.read_bytes())
    return root / "dynamics_record.npz"


@pytest.fixture
def explorer(qapp: QApplication) -> Generator[MainWidget, None, None]:
    widget = MainWidget()
    widget.resize(1280, 800)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.cleanup()
    widget.deleteLater()


def _editor_docks(widget: MainWidget) -> tuple:
    return widget.segment_dock, widget.properties_dock


def test_loading_a_run_hides_the_editor_docks(
    explorer: MainWidget, fit: Path, qapp: QApplication
) -> None:
    assert all(d.isVisible() for d in _editor_docks(explorer))
    assert explorer.load_results(fit) is True
    qapp.processEvents()
    assert explorer.playback_dock is not None
    assert explorer.playback_dock.isVisible()
    assert not any(d.isVisible() for d in _editor_docks(explorer))


def test_viewport_tab_restores_the_editor_docks(
    explorer: MainWidget, fit: Path, qapp: QApplication
) -> None:
    assert explorer.load_results(fit) is True
    explorer.visualization_dock.raise_()
    qapp.processEvents()
    assert all(d.isVisible() for d in _editor_docks(explorer))
    explorer.playback_dock.raise_()
    qapp.processEvents()
    assert not any(d.isVisible() for d in _editor_docks(explorer))


def test_view_menu_action_can_reshow_a_hidden_dock(
    explorer: MainWidget, fit: Path, qapp: QApplication
) -> None:
    assert explorer.load_results(fit) is True
    qapp.processEvents()
    explorer.segment_dock.toggleViewAction().trigger()
    qapp.processEvents()
    assert explorer.segment_dock.isVisible()
    assert explorer.playback_dock.isVisible()


def test_a_dock_the_user_closed_stays_closed_after_the_round_trip(
    explorer: MainWidget, fit: Path, qapp: QApplication
) -> None:
    explorer.properties_dock.hide()
    assert explorer.load_results(fit) is True
    explorer.visualization_dock.raise_()
    qapp.processEvents()
    assert explorer.segment_dock.isVisible()
    assert not explorer.properties_dock.isVisible()


def test_render_gets_the_settings_share_of_the_panel(
    explorer: MainWidget, fit: Path, qapp: QApplication
) -> None:
    assert explorer.load_results(fit) is True
    qapp.processEvents()
    qapp.processEvents()
    panel = explorer.playback_panel
    splitter = panel.findChild(QSplitter)
    cfg = load_layout_settings()
    render, plot = splitter.sizes()
    assert render > plot
    assert render / plot == pytest.approx(cfg.render_stretch / cfg.plot_stretch, 0.25)


def test_transport_controls_share_one_row(
    explorer: MainWidget, fit: Path, qapp: QApplication
) -> None:
    assert explorer.load_results(fit) is True
    qapp.processEvents()
    panel = explorer.playback_panel
    controls = (
        panel.play_button,
        panel.slider,
        panel.view_combo,
        panel.speed_combo,
        panel.export_button,
    )
    tops = {w.mapTo(panel, w.rect().center()).y() for w in controls}
    assert max(tops) - min(tops) <= 4


def test_timeline_slider_keeps_its_minimum_width_when_narrow(
    explorer: MainWidget, fit: Path, qapp: QApplication
) -> None:
    assert explorer.load_results(fit) is True
    explorer.resize(300, 800)  # clamps to the window's minimum
    for _ in range(3):
        qapp.processEvents()
    assert (
        explorer.playback_panel.slider.width()
        >= load_layout_settings().slider_min_width_px
    )

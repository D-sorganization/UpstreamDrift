"""Offscreen: the playback render is closer and fills its label's shape (#15150)."""

from __future__ import annotations

import os
import shutil
from collections.abc import Generator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
mujoco = pytest.importorskip("mujoco", reason="golfer playback needs mujoco")
pytest.importorskip("PyQt6", reason="explorer needs PyQt6")

from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.model_explorer.gui import MainWidget  # noqa: E402
from src.tools.model_explorer.playback_view import (  # noqa: E402
    load_camera_distances,
    load_framing_settings,
)

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
    root = tmp_path_factory.mktemp("framing") / "mujoco_fit"
    shutil.copytree(FIXTURES / "mujoco_fit", root)
    (root / "full_body_spec_hipcal_scaled.json").write_bytes(SPEC.read_bytes())
    return root / "dynamics_record.npz"


@pytest.fixture
def explorer(qapp: QApplication, fit: Path) -> Generator[MainWidget, None, None]:
    widget = MainWidget()
    widget.resize(1280, 800)
    widget.show()
    assert widget.load_results(fit) is True
    yield widget
    widget.cleanup()
    widget.deleteLater()


def _wait(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()
    QCoreApplication.processEvents()


def test_down_the_line_uses_the_configured_camera_distance(
    explorer: MainWidget,
) -> None:
    panel = explorer.playback_panel
    assert panel.view_combo.currentText() == "down_the_line"
    cam = panel._playback._camera
    assert cam.distance == pytest.approx(load_camera_distances()["down_the_line"])
    panel.set_view("face_on")
    assert cam.distance == pytest.approx(load_camera_distances()["face_on"])


def test_render_aspect_follows_the_label_size(explorer: MainWidget) -> None:
    cfg = load_framing_settings()
    panel = explorer.playback_panel
    for width, height in ((1500, 800), (1100, 800)):
        explorer.resize(width, height)
        _wait(cfg.resize_debounce_ms * 3)
        label = panel.view_label
        w, h = panel._playback.size
        expected = cfg.render_size(label.width() / label.height())
        assert (w, h) == expected
        shown = label.pixmap()
        assert shown is not None and not shown.isNull()
        # the scaled picture reaches both label edges (no gutter) unless clamped
        gutter_w = label.width() - shown.width()
        gutter_h = label.height() - shown.height()
        clamped = expected[0] / expected[1] in (cfg.min_aspect, cfg.max_aspect)
        if not clamped:
            assert min(gutter_w, gutter_h) <= 2 + cfg.aspect_step * label.height()


def test_the_renderer_is_only_rebuilt_when_the_size_changes(
    explorer: MainWidget,
) -> None:
    cfg = load_framing_settings()
    panel = explorer.playback_panel
    _wait(cfg.resize_debounce_ms * 3)
    before = panel._playback._renderer
    explorer.resize(explorer.width() + 1, explorer.height())
    _wait(cfg.resize_debounce_ms * 3)
    assert panel._playback._renderer is before

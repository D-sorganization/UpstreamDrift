"""Offscreen: the explorer plays real engine results on the golfer (#12915).

A run is the committed 4-frame fixture (real MuJoCo fit / Drake rollout
values) paired with the repository's canonical anthro-driver specification,
so the explorer builds the shared-appearance golfer exactly as for a
datastore run. Covers File > Load Results, the Results Playback dock, the
timeline scrub, the plots and Export MP4.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
mujoco = pytest.importorskip("mujoco", reason="golfer playback needs mujoco")
pytest.importorskip("cv2", reason="MP4 export needs opencv")
pytest.importorskip("PyQt6", reason="explorer needs PyQt6")

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from src.shared.python.motion_matching.same_input.bundle import (  # noqa: E402
    InputBundle,
)
from src.tools.model_explorer.golfer_playback import CLUB_HEAD_SPEED  # noqa: E402
from src.tools.model_explorer.gui import MainWidget  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
FIXTURES = REPO_ROOT / "tests/fixtures/results_playback"


def _require_offscreen_gl() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    try:
        mujoco.Renderer(model, 16, 16).close()
    except Exception as exc:  # noqa: BLE001 - any GL backend failure means skip
        pytest.skip(f"no offscreen OpenGL for mujoco.Renderer: {exc}")


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    _require_offscreen_gl()
    inst = QApplication.instance()
    return inst if isinstance(inst, QApplication) else QApplication(["explorer"])


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Fixture q on the canonical spec: a MuJoCo fit and a Drake rollout."""
    root = tmp_path_factory.mktemp("runs")
    spec_bytes = SPEC.read_bytes()
    fit = root / "mujoco_fit"
    shutil.copytree(FIXTURES / "mujoco_fit", fit)
    (fit / "full_body_spec_hipcal_scaled.json").write_bytes(spec_bytes)

    drake = root / "drake_rollout"
    drake.mkdir()
    with np.load(FIXTURES / "drake_rollout" / "closed_loop_drake.npz") as data:
        q = np.asarray(data["q"])
        np.savez(drake / "closed_loop_drake.npz", q=q, time_s=data["time_s"])
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=tuple(json.loads(spec_bytes)["coordinate_order"]),
        dt_s=0.001,
        q0=q[0],
        v0=np.zeros(q.shape[1]),
        efforts=np.zeros((len(q) - 1, q.shape[1])),
        reference_q=q,
        reference_v=np.zeros_like(q),
        reference_engine="mujoco",
    )
    bundle.save(drake / "bundle.npz")
    receipt = {
        "schema": "same-input-closed-loop/v1",
        "engine": "drake",
        "bundle": {"spec_sha256": bundle.spec_sha256},
    }
    (drake / "closed_loop_drake.json").write_text(json.dumps(receipt))
    return {
        "mujoco": fit / "dynamics_record.npz",
        "drake": drake / "closed_loop_drake.npz",
    }


@pytest.fixture
def boxes() -> Generator[list[str], None, None]:
    shown: list[str] = []

    def _record(*args: Any, **_kwargs: Any) -> int:
        shown.append(" ".join(str(a) for a in args[1:]))
        return int(QMessageBox.StandardButton.Ok)

    with (
        patch.object(QMessageBox, "warning", _record),
        patch.object(QMessageBox, "critical", _record),
        patch.object(QMessageBox, "information", _record),
    ):
        yield shown


@pytest.fixture(scope="module")
def explorer(qapp: QApplication) -> Generator[MainWidget, None, None]:
    widget = MainWidget()
    widget.resize(1280, 800)
    yield widget
    widget.cleanup()
    widget.deleteLater()


@pytest.mark.unit
def test_mujoco_fit_opens_playback_dock_on_the_golfer(
    explorer: MainWidget, runs: dict[str, Path], boxes: list[str]
) -> None:
    assert explorer.load_results(runs["mujoco"]) is True, boxes

    panel = explorer.playback_panel
    assert panel is not None
    assert explorer.playback_dock is not None
    assert explorer.playback_dock.windowTitle() == "Results Playback"
    assert panel.title_label.text() == "MuJoCo result"
    assert (panel.slider.minimum(), panel.slider.maximum()) == (0, 3)
    image = panel.current_image()
    assert image.shape[2] == 3 and image.std() > 10.0  # a scene, not a blank
    assert explorer.export_video_action.isEnabled()


@pytest.mark.unit
def test_scrubbing_moves_the_golfer_and_the_plot_cursor(
    explorer: MainWidget, runs: dict[str, Path], boxes: list[str]
) -> None:
    assert explorer.load_results(runs["mujoco"]) is True, boxes
    panel = explorer.playback_panel
    first = panel.current_image()

    panel.slider.setValue(3)

    assert explorer.current_result_frame == 3
    assert "4/4" in panel.time_label.text()
    assert np.abs(panel.current_image().astype(int) - first.astype(int)).mean() > 1.0
    assert CLUB_HEAD_SPEED in panel.series
    assert panel.series[CLUB_HEAD_SPEED].values.shape == (4,)
    panel.select_channel("HipInputZ")
    assert panel.channel_combo.currentText() == "HipInputZ"
    with pytest.raises(ValueError, match="no channel"):
        panel.select_channel("not_a_joint")


@pytest.mark.unit
def test_drake_rollout_is_labelled_playback_of_drake_result(
    explorer: MainWidget, runs: dict[str, Path], boxes: list[str]
) -> None:
    assert explorer.load_results(runs["drake"]) is True, boxes

    title = explorer.playback_panel.title_label.text()
    assert title.startswith("Playback of Drake result")
    assert "MuJoCo golfer model" in title


@pytest.mark.unit
def test_export_mp4_writes_the_golfer_video(
    explorer: MainWidget, runs: dict[str, Path], boxes: list[str], tmp_path: Path
) -> None:
    assert explorer.load_results(runs["mujoco"]) is True, boxes
    target = tmp_path / "swing.mp4"

    with patch.object(
        QFileDialog, "getSaveFileName", return_value=(str(target), "MP4")
    ):
        out = explorer.playback_panel.export_mp4()

    assert out == target
    assert target.is_file() and target.stat().st_size > 0


@pytest.mark.unit
def test_invalid_run_reports_the_missing_field(
    explorer: MainWidget, boxes: list[str]
) -> None:
    broken = FIXTURES / "missing_q" / "dynamics_record.npz"

    assert explorer.load_results(broken) is False

    assert explorer.playback_panel is None
    assert any("'q'" in text for text in boxes)

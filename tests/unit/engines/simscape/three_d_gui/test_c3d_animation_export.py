"""Tests for C3D animation export service."""

import pytest
from pathlib import Path
import os
import stat

from src.apps.services.c3d_animation_export import export_animation
from src.apps.core.models import C3DDataModel, MarkerData
import numpy as np
import sys

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


def test_export_animation_validates_path(tmp_path: Path):
    model = C3DDataModel(filepath="dummy.c3d")
    model.markers = {"M1": MarkerData(name="M1", position=np.zeros((5, 3)))}
    with pytest.raises(ValueError, match="Export path must have an .mp4 suffix"):
        export_animation(model, tmp_path / "out.avi")


def test_export_animation_validates_fps(tmp_path: Path):
    model = C3DDataModel(filepath="dummy.c3d")
    model.markers = {"M1": MarkerData(name="M1", position=np.zeros((5, 3)))}
    with pytest.raises(ValueError, match="fps must be positive"):
        export_animation(model, tmp_path / "out.mp4", fps=0)


def test_export_animation_validates_dimensions(tmp_path: Path):
    model = C3DDataModel(filepath="dummy.c3d")
    model.markers = {"M1": MarkerData(name="M1", position=np.zeros((5, 3)))}
    with pytest.raises(ValueError, match="width and height must be positive"):
        export_animation(model, tmp_path / "out.mp4", width=0)

    with pytest.raises(ValueError, match="width and height must be even"):
        export_animation(model, tmp_path / "out.mp4", width=111)


def test_export_animation_no_filepath(tmp_path: Path):
    model = C3DDataModel(filepath="")
    with pytest.raises(ValueError, match="model has no loaded C3D file path"):
        export_animation(model, tmp_path / "out.mp4")


def test_export_animation_no_markers(tmp_path: Path):
    model = C3DDataModel(filepath="dummy.c3d")
    with pytest.raises(ValueError, match="Model has no markers to export"):
        export_animation(model, tmp_path / "out.mp4")


@pytest.mark.skipif(
    not (Path(__file__).parents[5] / "data" / "C3D_TA_Driver.c3d").exists(),
    reason="Missing driver test data",
)
def test_export_animation_real_data(tmp_path: Path, monkeypatch):
    pytest.importorskip("ezc3d")
    pytest.importorskip("cv2")
    c3d_path = Path(__file__).parents[5] / "data" / "C3D_TA_Driver.c3d"
    model = C3DDataModel(filepath=str(c3d_path))
    model.markers = {"dummy": MarkerData(name="dummy", position=np.zeros((1, 3)))}
    out_path = tmp_path / "test_out.mp4"
    result = export_animation(model, out_path, fps=30, frame_indices=[0, 1])
    assert result.output_path == out_path
    assert result.frame_count == 2
    assert result.fps == 30.0
    assert out_path.exists()
    assert out_path.stat().st_size > 100  # Not empty


def test_viewer_menu_action_wired(qt_app):
    pytest.importorskip("trimesh")
    from src.apps.c3d_viewer import C3DViewerMainWindow

    win = C3DViewerMainWindow()
    assert not win.action_export_animation.isEnabled()
    # It should be enabled after model is populated
    model = C3DDataModel(filepath="dummy.c3d")
    win.model = model
    win._populate_ui_with_model()
    assert win.action_export_animation.isEnabled()

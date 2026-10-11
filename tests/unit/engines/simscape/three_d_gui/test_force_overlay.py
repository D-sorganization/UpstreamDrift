"""Tests for Simscape 3D viewer force and torque overlay (ADR-0052, #11305)."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QCheckBox

from src.apps.core.models import C3DDataModel, MarkerData
from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab
from src.shared.python.force_overlay.contracts import AxialLoadFrame, WrenchKind

pytestmark = pytest.mark.unit


def _write_synthetic_trial_csv(path: Path) -> Path:
    """Create a minimal synthetic Simscape trial CSV with TotalHandForceGlobal."""
    headers = [
        "time",
        "CalculatedSignalsLogs_TotalHandForceGlobal_1",
        "CalculatedSignalsLogs_TotalHandForceGlobal_2",
        "CalculatedSignalsLogs_TotalHandForceGlobal_3",
        "MidpointCalcsLogs_MPGlobalPosition_1",
        "MidpointCalcsLogs_MPGlobalPosition_2",
        "MidpointCalcsLogs_MPGlobalPosition_3",
        "HipLogs_BaseonHipForceGlobal_1",
        "HipLogs_BaseonHipForceGlobal_2",
        "HipLogs_BaseonHipForceGlobal_3",
        "HipLogs_HipGlobalPosition_dim1",
        "HipLogs_HipGlobalPosition_dim2",
        "HipLogs_HipGlobalPosition_dim3",
    ]
    # Frame 0: t=0.0, Hand force 10 N in X, Hip force 50 N in Z
    # Frame 1: t=0.01, Hand force 20 N in X, Hip force 60 N in Z
    # Frame 2: t=0.5 (gap > 2*dt where dt=0.01), Hand force 0 (below floor)
    rows = [
        [0.0, 10.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.8],
        [0.01, 20.0, 0.0, 0.0, 0.5, 0.6, 1.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.8],
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.7, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([repr(v) for v in r])
    return path


def _make_model_for_trial(csv_path: Path, times: list[float]) -> C3DDataModel:
    point_time = np.array(times, dtype=float)
    markers = {
        "MP": MarkerData("MP", np.column_stack([np.zeros_like(point_time)] * 3)),
    }
    return C3DDataModel(
        filepath=str(csv_path),
        markers=markers,
        point_time=point_time,
        point_rate=100.0,
    )


def test_loading_trial_csv_draws_grip_arrow_on_frame_0(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    viewer = Viewer3DTab()
    csv_file = _write_synthetic_trial_csv(tmp_path / "synthetic_trial.csv")

    model = _make_model_for_trial(csv_file, [0.0, 0.01, 0.5])
    viewer.update_from_model(model)
    app.processEvents()

    # Model should hold loaded ForceTorqueSeries
    assert getattr(viewer.model, "force_series", None) is not None
    assert len(viewer.model.force_series) == 3

    # On frame 0, glyphs should be rendered: at least one grip arrow
    assert viewer.force_glyph_count > 0

    # Changing frame updates the arrow
    viewer.set_frame(1)
    app.processEvents()
    assert viewer.force_glyph_count > 0

    # Gap frame beyond max_gap_s (frame 2 at t=0.5 with dt=0.01) removes glyphs
    viewer.set_frame(2)
    app.processEvents()
    assert viewer.force_glyph_count == 0

    viewer.close()


def test_set_segment_axial_loads_called_on_frame_change(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    viewer = Viewer3DTab()
    csv_file = _write_synthetic_trial_csv(tmp_path / "synthetic_trial.csv")
    model = _make_model_for_trial(csv_file, [0.0, 0.01, 0.5])

    with patch.object(
        viewer, "set_segment_axial_loads", wraps=viewer.set_segment_axial_loads
    ) as mock_axial:
        viewer.update_from_model(model)
        app.processEvents()
        assert mock_axial.called
        # First call has frame's axial loads (or None if no reactions)
        call_args = mock_axial.call_args[0]
        loads_arg = call_args[0]
        # In a synthetic trial with only hand/hip forces and no joint reactions,
        # axial_loads is either an AxialLoadFrame with None values or None
        if loads_arg is not None:
            assert isinstance(loads_arg, AxialLoadFrame)
            assert all(v is None for v in loads_arg.values_n.values())

    viewer.close()


def test_toggles_filter_kinds(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    viewer = Viewer3DTab()
    csv_file = _write_synthetic_trial_csv(tmp_path / "synthetic_trial.csv")
    model = _make_model_for_trial(csv_file, [0.0, 0.01, 0.5])
    viewer.update_from_model(model)
    app.processEvents()

    # Verify toggle controls exist
    check_forces = viewer.findChild(QCheckBox, "check_force_glyphs")
    check_torques = viewer.findChild(QCheckBox, "check_torque_glyphs")
    check_grip = viewer.findChild(QCheckBox, "check_grip_glyphs")
    assert check_forces is not None
    assert check_torques is not None
    assert check_grip is not None

    initial_count = viewer.force_glyph_count
    assert initial_count > 0

    # Unchecking grip removes grip glyphs
    check_grip.setChecked(False)
    app.processEvents()
    assert WrenchKind.GRIP not in viewer.active_force_kinds
    # Since only grip and hip base force were present, unchecking grip should reduce count
    assert viewer.force_glyph_count < initial_count

    # Unchecking forces removes all force arrows
    check_forces.setChecked(False)
    app.processEvents()
    assert viewer.force_glyph_count == 0

    # Checking grip back on while forces is off still has 0 force arrows
    check_grip.setChecked(True)
    app.processEvents()
    assert viewer.force_glyph_count == 0

    # Re-enabling forces restores them
    check_forces.setChecked(True)
    app.processEvents()
    assert viewer.force_glyph_count == initial_count

    viewer.close()


def test_real_trial_peak_force_screenshot() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    csv_path = (
        repo_root
        / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/golf_swing_dataset_20250907_bk/trial_001_20251117_114559.csv"
    )
    if not csv_path.exists():
        pytest.skip(f"{csv_path} not found")

    app = QApplication.instance() or QApplication([])
    viewer = Viewer3DTab()

    # Load 31 frames (t=0.0 to t=0.3 at dt=0.01)
    times = [i * 0.01 for i in range(31)]
    model = _make_model_for_trial(csv_path, times)
    viewer.update_from_model(model)
    app.processEvents()

    assert viewer.model is not None
    assert viewer.model.force_series is not None
    assert len(viewer.model.force_series) == 31

    # Frame 30 corresponds to peak force (t=0.30s)
    viewer.set_frame(30)
    app.processEvents()
    assert viewer.force_glyph_count > 0

    evidence_dir = repo_root / "docs/development/evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    out_png = evidence_dir / "simscape_viewer_trial_001_peak_force.png"
    viewer.canvas_3d.fig.canvas.draw()
    viewer.canvas_3d.fig.savefig(out_png, dpi=100)
    assert out_png.is_file()
    assert out_png.stat().st_size > 0

    viewer.close()


def test_missing_or_corrupt_csv_gracefully_degrades(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    viewer = Viewer3DTab()

    # Case 1: missing CSV file
    non_existent = tmp_path / "missing.csv"
    model_missing = _make_model_for_trial(non_existent, [0.0, 0.01])
    viewer.update_from_model(model_missing)
    app.processEvents()

    assert viewer.model is not None
    assert viewer.model.force_series is not None
    assert len(viewer.model.force_series) == 0
    assert viewer.force_glyph_count == 0
    assert "not found" in viewer._force_overlay.status_note.lower()

    # Case 2: corrupt CSV file (non-numeric data in time column)
    corrupt_csv = tmp_path / "corrupt.csv"
    corrupt_csv.write_text(
        "time,CalculatedSignalsLogs_TotalHandForceGlobal_1\nINVALID,1.0\n",
        encoding="utf-8",
    )
    model_corrupt = _make_model_for_trial(corrupt_csv, [0.0])
    viewer.update_from_model(model_corrupt)
    app.processEvents()

    assert viewer.model is not None
    assert viewer.model.force_series is not None
    assert len(viewer.model.force_series) == 0
    assert viewer.force_glyph_count == 0
    assert "unavailable" in viewer._force_overlay.status_note.lower()

    viewer.close()

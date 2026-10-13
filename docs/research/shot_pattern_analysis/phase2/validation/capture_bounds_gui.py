"""Capture the reviewed bounds controls through Qt's offscreen platform only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PyQt6 import QtCore, QtWidgets

from src.tools.shot_pattern_analysis.matched_gui import MatchedStudyWidget
from src.tools.shot_pattern_analysis.matched_matrix import matched_matrix_cells

CELL = "driver_lean5_curve2_rotation2_combined2"
TARGET_X_M = 178.94875782337786


def capture(output: Path) -> dict[str, Any]:
    """Save one exact supplemental config and three 1000×760 review views."""
    if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        raise RuntimeError("GUI evidence capture requires QT_QPA_PLATFORM=offscreen")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = MatchedStudyWidget()
    try:
        _, request = matched_matrix_cells(
            n_shots=32,
            common_targets={"driver": TARGET_X_M},
            profiles=("combined2",),
            selected_cell_names=(CELL,),
            match_face_bounds_deg=(-6.0, 6.0),
            match_speed_factors=(0.75, 1.30),
        )[0]
        config_path = output / "bounds_gui_loaded_config.json"
        widget.export_config(config_path, request)
        widget.load_config(config_path)
        if widget.build_experiment() != request:
            raise RuntimeError("GUI changed an imported supplemental experiment")
        widget.resize(1000, 760)
        widget.show()
        app.processEvents()
        if (widget.width(), widget.height()) != (1000, 760):
            raise RuntimeError("offscreen widget did not fit 1000×760")
        scrollbar = widget.controls_scroll.verticalScrollBar()
        if scrollbar is None:
            raise RuntimeError("verticalScrollBar is None")
        views = (
            (0, "bounds_gui_main_1000x760.png"),
            (120, "bounds_gui_scrolled_1000x760.png"),
            (scrollbar.maximum(), "bounds_gui_correlations_1000x760.png"),
        )
        for position, filename in views:
            scrollbar.setValue(position)
            app.processEvents()
            if not widget.grab().save(str(output / filename)):
                raise RuntimeError(f"could not save offscreen screenshot: {filename}")
        checks = {
            "cell": CELL,
            "loaded_request_equal": widget.build_experiment() == request,
            "face_bounds_enabled": widget.use_absolute_face_bounds.isChecked(),
            "face_bounds_deg": [
                widget.face_bound_min.value(),
                widget.face_bound_max.value(),
            ],
            "speed_factors": [
                widget.speed_factor_min.value(),
                widget.speed_factor_max.value(),
            ],
            "target_x_m": widget.target_distance.value(),
            "scroll_max": scrollbar.maximum(),
            "run_visible": widget.run_button.isVisibleTo(widget),
            "cancel_visible": widget.action_bar.cancel_button.isVisibleTo(widget),
            "status_visible": widget.status_label.isVisibleTo(widget),
            "run_position": [
                widget.run_button.mapTo(widget, QtCore.QPoint(0, 0)).x(),
                widget.run_button.mapTo(widget, QtCore.QPoint(0, 0)).y(),
            ],
            "cancel_position": [
                widget.action_bar.cancel_button.mapTo(widget, QtCore.QPoint(0, 0)).x(),
                widget.action_bar.cancel_button.mapTo(widget, QtCore.QPoint(0, 0)).y(),
            ],
            "status_position": [
                widget.status_label.mapTo(widget, QtCore.QPoint(0, 0)).x(),
                widget.status_label.mapTo(widget, QtCore.QPoint(0, 0)).y(),
            ],
        }
        if not all(
            checks[key] for key in ("run_visible", "cancel_visible", "status_visible")
        ):
            raise RuntimeError("Run, Cancel, or status is inaccessible")
        (output / "bounds_gui_capture_checks.json").write_text(
            json.dumps(checks, indent=2) + "\n"
        )
        return checks
    finally:
        widget.cleanup()
        widget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    capture(args.output)

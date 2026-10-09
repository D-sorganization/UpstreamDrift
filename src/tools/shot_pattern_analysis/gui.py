"""Optional PyQt presenter for shot-pattern simulation and reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, NamedTuple

from PyQt6 import QtCore, QtGui, QtWidgets

from src.shared.python.core.process_safety import managed_popen
from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.async_action import AsyncActionBar, WorkerContext
from src.tools.shot_pattern_analysis.presets import CLUB_PRESETS, get_club_preset

logger = get_logger(__name__)


class _ProcessOutput(NamedTuple):
    returncode: int
    log: str


class MainWidget(QtWidgets.QWidget):
    """Run the bundled headless analysis in a cancellable background process."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._closed = False
        self.setWindowTitle("Shot Pattern Analysis")
        self._build_ui()

    def _build_ui(self) -> None:
        title = QtWidgets.QLabel("Shot Pattern Analysis")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")

        description = QtWidgets.QLabel(self._description_text(1.0, 1.0))
        description.setWordWrap(True)
        self.description_label = description

        self.output_edit = QtWidgets.QLineEdit(
            str(Path.cwd() / "shot_pattern_analysis_results")
        )
        self.output_edit.setObjectName("outputDirectory")
        browse_button = QtWidgets.QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_output)
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(QtWidgets.QLabel("Save Results To"))
        output_row.addWidget(self.output_edit, stretch=1)
        output_row.addWidget(browse_button)

        self.shot_count = QtWidgets.QSpinBox()
        self.shot_count.setRange(2, 100_000)
        self.shot_count.setValue(10_000)
        self.shot_count.setSuffix(" per pattern")
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(20_261_008)
        self.face_sd = QtWidgets.QDoubleSpinBox()
        self.face_sd.setRange(0.0, 5.0)
        self.face_sd.setDecimals(1)
        self.face_sd.setSingleStep(0.5)
        self.face_sd.setValue(1.0)
        self.face_sd.setSuffix("° SD")
        self.curve_scale = QtWidgets.QDoubleSpinBox()
        self.curve_scale.setRange(0.1, 2.0)
        self.curve_scale.setDecimals(1)
        self.curve_scale.setSingleStep(0.1)
        self.curve_scale.setValue(1.0)
        self.curve_scale.setSuffix("×")
        self.face_sd.valueChanged.connect(self._update_description)
        self.curve_scale.valueChanged.connect(self._update_curve_description)
        self.delivery_mode = QtWidgets.QComboBox()
        self.delivery_mode.addItem("Fixed Delivered Loft", "fixed_loft")
        self.delivery_mode.addItem("Shaft Rotation Coupling", "shaft_rotation")
        self.lie = QtWidgets.QDoubleSpinBox()
        self.lie.setRange(45.0, 75.0)
        self.lie.setDecimals(1)
        self.lie.setValue(58.0)
        self.lie.setSuffix("° Lie")
        self.shaft_lean = QtWidgets.QDoubleSpinBox()
        self.shaft_lean.setRange(-30.0, 30.0)
        self.shaft_lean.setDecimals(1)
        self.shaft_lean.setValue(0.0)
        self.shaft_lean.setSuffix("° Shaft Lean")
        self.delivery_mode.currentIndexChanged.connect(self._delivery_mode_changed)
        self.delivery_mode.currentIndexChanged.connect(
            self._update_delivery_description
        )
        self.lie.valueChanged.connect(self._update_delivery_description)
        self.shaft_lean.valueChanged.connect(self._update_delivery_description)
        self.club_preset = QtWidgets.QComboBox()
        self.club_preset.addItem("Custom / Control Baseline", "custom")
        for preset in CLUB_PRESETS.values():
            self.club_preset.addItem(preset.label, preset.id)
        self.club_preset.setCurrentIndex(self.club_preset.findData("driver"))
        self.club_speed = QtWidgets.QDoubleSpinBox()
        self.club_speed.setRange(10.0, 80.0)
        self.club_speed.setDecimals(1)
        self.club_speed.setValue(45.0)
        self.club_speed.setSuffix(" m/s Speed")
        self.loft = QtWidgets.QDoubleSpinBox()
        self.loft.setRange(1.0, 44.9)
        self.loft.setDecimals(1)
        self.loft.setValue(10.9)
        self.loft.setSuffix("° Nominal Loft")
        self.attack_angle = QtWidgets.QDoubleSpinBox()
        self.attack_angle.setRange(-25.0, 25.0)
        self.attack_angle.setDecimals(1)
        self.attack_angle.setValue(0.0)
        self.attack_angle.setSuffix("° AoA")
        self.clubhead_mass = QtWidgets.QDoubleSpinBox()
        self.clubhead_mass.setRange(0.05, 1.0)
        self.clubhead_mass.setDecimals(3)
        self.clubhead_mass.setSingleStep(0.01)
        self.clubhead_mass.setValue(0.2)
        self.clubhead_mass.setSuffix(" kg Head Mass")
        self._applying_preset = False
        self.club_preset.currentIndexChanged.connect(self._apply_club_preset)
        self.preset_assumption = QtWidgets.QLabel()
        self.preset_assumption.setWordWrap(True)
        self._apply_club_preset(self.club_preset.currentIndex())
        for input_widget in (
            self.club_speed,
            self.loft,
            self.attack_angle,
            self.clubhead_mass,
            self.lie,
            self.shaft_lean,
        ):
            input_widget.valueChanged.connect(self._update_delivery_description)
            input_widget.valueChanged.connect(self._mark_custom_preset)
        self.run_button = QtWidgets.QPushButton("Run Analysis")
        self.run_button.setObjectName("runAnalysis")
        self.run_button.clicked.connect(self.run_analysis)
        control_row = QtWidgets.QHBoxLayout()
        control_row.addWidget(QtWidgets.QLabel("Shots"))
        control_row.addWidget(self.shot_count)
        control_row.addWidget(QtWidgets.QLabel("Seed"))
        control_row.addWidget(self.seed)
        control_row.addWidget(QtWidgets.QLabel("Face Variation"))
        control_row.addWidget(self.face_sd)
        control_row.addWidget(QtWidgets.QLabel("Curve Magnitude"))
        control_row.addWidget(self.curve_scale)
        control_row.addWidget(self.run_button)
        delivery_row = QtWidgets.QHBoxLayout()
        delivery_row.addWidget(QtWidgets.QLabel("Face-to-Loft Model"))
        delivery_row.addWidget(self.delivery_mode)
        delivery_row.addWidget(self.lie)
        delivery_row.addWidget(self.shaft_lean)
        delivery_row.addStretch(1)
        club_row = QtWidgets.QHBoxLayout()
        club_row.addWidget(QtWidgets.QLabel("Illustrative Club Preset"))
        club_row.addWidget(self.club_preset)
        club_row.addWidget(self.club_speed)
        club_row.addWidget(self.loft)
        club_row.addWidget(self.attack_angle)
        club_row.addWidget(self.clubhead_mass)

        self.action_bar = AsyncActionBar()
        self.action_bar.set_trigger_buttons(self.run_button)
        self.action_bar.cancelled.connect(self._cancelled)

        self.summary_table = QtWidgets.QTableWidget(0, 7)
        self.summary_table.setObjectName("summaryTable")
        self.summary_table.setHorizontalHeaderLabels(
            [
                "Pattern",
                "Nominal Delivered Loft (°)",
                "Lateral SD",
                "Aimed Hit Rate",
                "Aimed RMSE",
                "Mean Carry",
                "Model SG",
            ]
        )
        horizontal_header = self.summary_table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeMode.Stretch
            )
        vertical_header = self.summary_table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        self.summary_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.scoring_label = QtWidgets.QLabel(
            "Approach SG Uses a Hypothetical Historical Tour Baseline; It Is Not a Player-Score Prediction."
        )
        self.scoring_label.setWordWrap(True)

        self.status_label = QtWidgets.QLabel(
            "Ready. Analysis and figure export run in a managed process that can be cancelled."
        )
        self.status_label.setWordWrap(True)
        self.figures = QtWidgets.QTabWidget()
        self.figure_labels: dict[str, QtWidgets.QLabel] = {}
        for key, label in (
            ("overhead_flight.png", "Overhead Flights"),
            ("dispersion.png", "Shot Dispersion"),
            ("dispersion_equal_range.png", "Equal-Range Dispersion"),
        ):
            image = QtWidgets.QLabel("Run an analysis to view this graphic.")
            image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            image.setMinimumSize(560, 350)
            image.setScaledContents(False)
            self.figure_labels[key] = image
            self.figures.addTab(image, label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(output_row)
        layout.addLayout(control_row)
        layout.addLayout(delivery_row)
        layout.addLayout(club_row)
        layout.addWidget(self.preset_assumption)
        layout.addWidget(self.action_bar)
        layout.addWidget(self.summary_table)
        self.delivery_assumption = QtWidgets.QLabel("")
        self.delivery_assumption.setWordWrap(True)
        self._delivery_mode_changed(self.delivery_mode.currentIndex())
        layout.addWidget(self.delivery_assumption)
        layout.addWidget(self.scoring_label)
        layout.addWidget(self.figures, stretch=1)
        layout.addWidget(self.status_label)

    def _browse_output(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose Results Folder", self.output_edit.text()
        )
        if directory:
            self.output_edit.setText(directory)

    @staticmethod
    def _description_text(face_sd: float, curve_scale: float) -> str:
        return (
            "Compare straight, draw, and fade patterns with paired face variation "
            f"at {face_sd:g}° SD. Draw face/path: "
            f"+{1.5 * curve_scale:g}°/+{3.0 * curve_scale:g}°; fade is mirrored. "
            "Results use Fixed Delivered Loft with independent face/path controls "
            "and are model-conditional carry predictions."
        )

    def _update_description(self, _face_sd: float) -> None:
        self._update_delivery_description()

    def _update_curve_description(self, _curve_scale: float) -> None:
        self._update_delivery_description()

    def _delivery_mode_changed(self, index: int) -> None:
        coupled = self.delivery_mode.itemData(index) == "shaft_rotation"
        self.lie.setEnabled(coupled)
        self.shaft_lean.setEnabled(coupled)
        self._update_delivery_assumption()

    def _update_delivery_assumption(self) -> None:
        if self.delivery_mode.currentData() == "shaft_rotation":
            text = (
                "Face error rotates about the assumed shaft axis around each pattern nominal; "
                "all patterns share nominal delivered loft."
            )
        else:
            text = (
                "Delivered loft is held fixed for every face error; all patterns share "
                "nominal loft."
            )
        self.delivery_assumption.setText(text)

    def _update_delivery_description(self, _value: object = None) -> None:
        face_sd = self.face_sd.value()
        curve_scale = self.curve_scale.value()
        text = self._description_text(face_sd, curve_scale)
        if self.delivery_mode.currentData() == "shaft_rotation":
            text = text.replace(
                "Fixed Delivered Loft with independent face/path controls",
                "Shaft Rotation Coupling at "
                f"{self.lie.value():g}° Lie and {self.shaft_lean.value():g}° Shaft Lean",
            )
        text += (
            f" Club input: {self.club_speed.value():g} m/s, "
            f"{self.loft.value():g}° nominal loft, "
            f"{self.attack_angle.value():g}° AoA, "
            f"{self.clubhead_mass.value():g} kg head mass."
        )
        self.description_label.setText(text)

    def _apply_club_preset(self, index: int) -> None:
        preset_id = str(self.club_preset.itemData(index))
        if preset_id == "custom":
            values = (45.0, 10.9, 0.0, 58.0, 0.0, 0.2)
            assumption = (
                "Custom / Control Baseline uses the original 45 m/s, 10.9° loft, "
                "0° AoA, 58° lie, 0° shaft lean, and 0.200 kg head mass."
            )
        else:
            preset = get_club_preset(preset_id)
            values = (
                preset.club_speed_mps,
                preset.loft_deg,
                preset.attack_angle_deg,
                preset.lie_deg,
                preset.shaft_lean_deg,
                preset.clubhead_mass_kg,
            )
            assumption = preset.source_assumption
        self._applying_preset = True
        for widget, value in zip(
            (
                self.club_speed,
                self.loft,
                self.attack_angle,
                self.lie,
                self.shaft_lean,
                self.clubhead_mass,
            ),
            values,
            strict=True,
        ):
            widget.setValue(value)
        self._applying_preset = False
        self.preset_assumption.setText(
            f"{assumption} These values are illustrative, not measured golfer means."
        )
        self._update_delivery_description()

    def _mark_custom_preset(self, _value: float) -> None:
        if self._applying_preset:
            return
        preset_id = str(self.club_preset.currentData())
        if preset_id == "custom":
            self.preset_assumption.setText(
                "Custom inputs use the control baseline; values are not measured golfer means."
            )
            return
        preset = get_club_preset(preset_id)
        self.preset_assumption.setText(
            f"{preset.label} (Modified). {preset.source_assumption} "
            "Values are illustrative, not measured golfer means."
        )

    def run_analysis(self) -> None:
        output_text = self.output_edit.text().strip()
        if not output_text:
            self.status_label.setText("Choose an output folder first.")
            return
        output_dir = Path(output_text).expanduser()
        shots = self.shot_count.value()
        seed = self.seed.value()
        face_sd = self.face_sd.value()
        curve_scale = self.curve_scale.value()
        delivery_mode = str(self.delivery_mode.currentData())
        lie_deg = self.lie.value()
        shaft_lean_deg = self.shaft_lean.value()
        club_speed_mps = self.club_speed.value()
        loft_deg = self.loft.value()
        attack_angle_deg = self.attack_angle.value()
        clubhead_mass_kg = self.clubhead_mass.value()
        club_id = str(self.club_preset.currentData())
        self.action_bar.start(
            "Shot pattern analysis",
            lambda ctx: self._run_cli(
                output_dir,
                shots,
                seed,
                face_sd,
                curve_scale,
                delivery_mode,
                lie_deg,
                shaft_lean_deg,
                club_speed_mps,
                loft_deg,
                attack_angle_deg,
                clubhead_mass_kg,
                club_id,
                ctx,
            ),
            on_finished=self._completed,
            on_failed=self._failed,
        )

    @staticmethod
    def _run_cli(
        output_dir: Path,
        shots: int,
        seed: int,
        face_sd: float,
        curve_scale: float,
        delivery_mode: str,
        lie_deg: float,
        shaft_lean_deg: float,
        club_speed_mps: float,
        loft_deg: float,
        attack_angle_deg: float,
        clubhead_mass_kg: float,
        club_id: str,
        ctx: WorkerContext,
    ) -> dict[str, Any]:
        root = Path(__file__).resolve().parents[3]
        command = [
            sys.executable,
            "-m",
            "src.tools.shot_pattern_analysis",
            "--output",
            str(output_dir),
            "--shots",
            str(shots),
            "--seed",
            str(seed),
            "--face-sd",
            f"{face_sd:g}",
            "--curve-scale",
            f"{curve_scale:g}",
            "--delivery-mode",
            delivery_mode,
            "--lie-deg",
            f"{lie_deg:g}",
            "--shaft-lean-deg",
            f"{shaft_lean_deg:g}",
            "--club-speed-mps",
            f"{club_speed_mps:g}",
            "--loft-deg",
            f"{loft_deg:g}",
            "--attack-angle-deg",
            f"{attack_angle_deg:g}",
            "--clubhead-mass-kg",
            f"{clubhead_mass_kg:g}",
            "--club-preset",
            club_id,
        ]
        env = {
            **os.environ,
            "MPLBACKEND": "Agg",
            "QT_QPA_PLATFORM": "offscreen",
            "MUJOCO_GL": "egl",
            "SDL_VIDEODRIVER": "dummy",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        output = MainWidget._run_command(command, root, env, ctx)
        if output.returncode != 0:
            raise RuntimeError(
                f"Analysis process exited with status {output.returncode}: "
                f"{output.log[-2000:]}"
            )
        ctx.raise_if_cancelled()
        summary_path = output_dir / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError("Analysis completed without summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {"output_dir": output_dir, "summary": summary}

    @staticmethod
    def _run_command(
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        ctx: WorkerContext,
    ) -> _ProcessOutput:
        """Poll a managed child so cancellation always reaps its process."""
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log_file:
            with managed_popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=log_file,
                text=True,
                kill_timeout=0.5,
            ) as process:
                started = time.monotonic()
                while process.poll() is None:
                    ctx.raise_if_cancelled()
                    elapsed = int(time.monotonic() - started)
                    ctx.report(
                        None,
                        f"Simulating, scoring approaches, and exporting results ({elapsed}s)",
                    )
                    time.sleep(0.2)
            log_file.seek(0)
            output = log_file.read()
        if process.returncode is None:
            raise RuntimeError("managed analysis process was not reaped")
        return _ProcessOutput(process.returncode, output)

    def _completed(self, result: dict[str, Any]) -> None:
        if self._closed:
            return
        output_dir = Path(result["output_dir"])
        patterns = result["summary"]["patterns"]
        config = result["summary"]["config"]
        scoring = result["summary"].get("course_scoring") or result["summary"].get(
            "approach_scoring", {}
        )
        scoring_available = scoring.get("status", "available") == "available"
        self.summary_table.setRowCount(len(patterns))
        columns = (
            ("aimed_lateral_sd_m", "{:.2f} m"),
            ("aimed_target_hit_fraction", "{:.1%}"),
            ("aimed_target_rmse_m", "{:.2f} m"),
            ("mean_carry_m", "{:.1f} m"),
            ("mean_strokes_gained", "{:+.3f}"),
        )
        for row, name in enumerate(("Straight", "Draw", "Fade")):
            self.summary_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.summary_table.setItem(
                row,
                1,
                QtWidgets.QTableWidgetItem(f"{config['loft_deg']:.1f}°"),
            )
            stats = dict(patterns[name])
            if scoring_available:
                stats.update(scoring["patterns"][name])
            for col, (field, template) in enumerate(columns, start=2):
                value = (
                    template.format(stats[field]) if field in stats else "Unavailable"
                )
                self.summary_table.setItem(row, col, QtWidgets.QTableWidgetItem(value))
        if scoring_available:
            paired = scoring["paired_benefit_vs_straight"]
            intervals = "; ".join(
                f"{name}: {values['estimate']:+.3f} "
                f"[{values['lower_95']:+.3f}, {values['upper_95']:+.3f}] strokes"
                for name, values in paired.items()
            )
            scenario = scoring.get(
                "scenario", "Historical Tour Course Scoring Hypothesis"
            )
            scoring_text = (
                f"Model-Conditional Strokes Gained: {scenario}. It Is Not a "
                f"Player-Score Prediction. Paired Difference vs. Straight (95% Monte "
                f"Carlo Interval): {intervals}."
            )
        else:
            scoring_text = (
                "Model-Conditional Strokes Gained Is Unavailable for This Configuration: "
                f"{scoring.get('reason', 'unsupported target geometry')}"
            )
        self.scoring_label.setText(scoring_text)
        for filename, label in self.figure_labels.items():
            path = output_dir / filename
            pixmap = QtGui.QPixmap(str(path))
            if not pixmap.isNull():
                label.setProperty("sourcePixmap", pixmap)
                label.setToolTip(str(path))
        self._scale_previews()
        self.status_label.setText(f"Analysis complete. Results saved in {output_dir}")

    def _failed(self, message: str) -> None:
        if not self._closed:
            self.status_label.setText(f"Analysis failed: {message}")
            logger.error("Shot pattern analysis failed: %s", message)

    def _cancelled(self) -> None:
        if not self._closed:
            self.status_label.setText(
                "Analysis cancelled; the child process was stopped."
            )

    def cleanup(self) -> bool:
        """Cancel and join the active process worker before host teardown."""
        self._closed = True
        return self.action_bar.shutdown()

    def delete_when_idle(self) -> None:
        """Defer QObject deletion until a worker that outlived shutdown ends."""
        self.action_bar.finished.connect(self.deleteLater)
        self.action_bar.failed.connect(self.deleteLater)
        self.action_bar.cancelled.connect(self.deleteLater)

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._scale_previews()

    def _scale_previews(self) -> None:
        for label in self.figure_labels.values():
            source = label.property("sourcePixmap")
            if isinstance(source, QtGui.QPixmap) and not source.isNull():
                label.setPixmap(
                    source.scaled(
                        label.size(),
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                )

    def is_dirty(self) -> bool:
        return False


__all__ = ["MainWidget"]

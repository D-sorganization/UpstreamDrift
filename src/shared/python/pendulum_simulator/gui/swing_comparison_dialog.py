"""
Swing robustness comparison dialog.

Compares Monte Carlo noise-sensitivity across 2–4 swing presets
using the same noise parameters for each, then ranks by coefficient
of variation (CV = std/mean tip speed).

Design by Contract
------------------
- At least 2 presets must be selected before running.
- Noise amplitude >= 0, n_trials >= 2.
- All result statistics are finite (enforced by variability_summary).

DRY
---
- Reuses PerturbationConfig and batch_perturb_and_simulate from
  perturbation_analysis.py.
- Background threading pattern from perturbation_panel.py.

Closes #1285
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..perturbation_analysis import (
    PerturbationConfig,
    perturb_torque_coeffs,
    variability_summary,
)

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

_STYLE = """
QDialog { background: #12121e; }
QGroupBox {
    color: #9090c8; font-size: 11px; font-weight: bold;
    border: 1px solid #303050; border-radius: 4px;
    margin-top: 8px; padding-top: 14px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; }
QLabel { color: #8080b0; font-size: 11px; }
QPushButton {
    background: #262650; color: #b0b0e8; border: 1px solid #404070;
    border-radius: 3px; padding: 4px 12px; font-size: 11px;
}
QPushButton:hover { background: #303068; }
QPushButton:disabled { color: #505060; }
QSpinBox, QDoubleSpinBox {
    background: #1a1a2a; color: #b0b0e8; border: 1px solid #303050;
    border-radius: 2px; font-size: 11px; padding: 2px;
}
QListWidget {
    background: #1a1a2a; color: #b0b0e8; border: 1px solid #303050;
    border-radius: 2px; font-size: 11px;
}
QListWidget::item:selected { background: #303068; }
QProgressBar {
    background: #1a1a2a; color: #b0b0e8; border: 1px solid #303050;
    border-radius: 2px; text-align: center; font-size: 11px;
}
QProgressBar::chunk { background: #3a3a80; }
QTextEdit {
    background: #1a1a2a; color: #b0b0e8; border: 1px solid #303050;
    font-family: monospace; font-size: 11px;
}
"""


# ---------------------------------------------------------------------------
# Background batch worker for multiple presets
# ---------------------------------------------------------------------------


class _ComparisonWorker(QObject):
    """Runs batch simulations for N presets on a background thread.

    Signals
    -------
    preset_progress(str, int) : (preset_name, trial_number)
    preset_done(str, dict) : (preset_name, variability_summary)
    all_done(list) : list of (preset_name, summary) tuples
    error(str)
    """

    preset_progress = pyqtSignal(str, int)
    preset_done = pyqtSignal(str, dict)
    all_done = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        preset_jobs: list[tuple[str, list[list[float]], Callable, Callable]],
        config: PerturbationConfig,
    ) -> None:
        if not (preset_jobs is not None):
            raise ValueError("preset_jobs must be provided")
        super().__init__()
        self._jobs = preset_jobs  # [(name, base_coeffs, simulate_fn, extract_fn)]
        self._config = config
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        results = []
        base_seed = self._config.seed if self._config.seed is not None else 0

        for preset_name, base_coeffs, simulate_fn, extract_fn in self._jobs:
            if self._cancelled:
                break
            trial_results = []
            for i in range(self._config.n_trials):
                if self._cancelled:
                    break  # type: ignore[unreachable]
                trial_seed = base_seed + i
                try:
                    perturbed = perturb_torque_coeffs(
                        base_coeffs,
                        noise_amplitude=self._config.noise_amplitude,
                        noise_type=self._config.noise_type,
                        seed=trial_seed,
                    )
                    sim_result = simulate_fn(perturbed)
                    metrics = extract_fn(sim_result)
                    trial_results.append(metrics)
                except (ValueError, RuntimeError, FloatingPointError, AssertionError):
                    logger.warning(
                        "Trial %d failed for preset %r, skipping",
                        i,
                        preset_name,
                    )
                self.preset_progress.emit(preset_name, i + 1)

            if trial_results:
                summary = variability_summary(trial_results)
                self.preset_done.emit(preset_name, summary)
                results.append((preset_name, summary))
            else:
                logger.warning("No successful trials for preset %r", preset_name)

        self.all_done.emit(results)


# ---------------------------------------------------------------------------
# Comparison dialog
# ---------------------------------------------------------------------------


class SwingComparisonDialog(QDialog):
    """Dialog for comparing swing robustness across multiple presets.

    Parameters
    ----------
    preset_names : list[str]
        Names of available presets.
    get_coeffs_for_preset : callable(name) -> list[list[float]]
        Returns torque polynomial coefficients for the named preset.
    simulate_fn : callable(coeffs) -> result
        Runs one simulation and returns result object.
    extract_fn : callable(result) -> dict
        Extracts tip_speed_final and tip_position_final from a result.
    parent : QWidget, optional

    Design by Contract
    ------------------
    Pre:  len(preset_names) >= 2
    Pre:  get_coeffs_for_preset, simulate_fn, extract_fn are callable
    Post: displayed CVs are finite (enforced by variability_summary)
    """

    def __init__(
        self,
        preset_names: list[str],
        get_coeffs_for_preset: Callable[[str], list[list[float]]],
        simulate_fn: Callable,
        extract_fn: Callable,
        parent: QWidget | None = None,
    ) -> None:
        if not (preset_names is not None):
            raise ValueError("preset_names must be provided")
        super().__init__(parent)
        if not (len(preset_names) >= 2):
            raise ValueError("Need at least 2 presets to compare")
        if not (callable(get_coeffs_for_preset)):
            raise ValueError('DbC Blocked: Precondition failed.')
        if not (callable(simulate_fn)):
            raise ValueError('DbC Blocked: Precondition failed.')
        if not (callable(extract_fn)):
            raise ValueError('DbC Blocked: Precondition failed.')

        self._preset_names = preset_names
        self._get_coeffs = get_coeffs_for_preset
        self._simulate_fn = simulate_fn
        self._extract_fn = extract_fn
        self._thread: QThread | None = None
        self._worker: _ComparisonWorker | None = None
        self._results: list[tuple[str, dict]] = []

        self.setWindowTitle("Swing Robustness Comparison")
        self.setMinimumSize(700, 600)
        self.setStyleSheet(_STYLE)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(self._build_preset_group())
        top.addWidget(self._build_noise_group())
        root.addLayout(top)

        root.addWidget(self._build_run_group())
        if _HAS_MPL:
            root.addWidget(self._build_chart_group())
        root.addWidget(self._build_results_group())

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _build_preset_group(self) -> QGroupBox:
        grp = QGroupBox("Select Presets (2–4)")
        lay = QVBoxLayout(grp)
        self._preset_list = QListWidget()
        self._preset_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for name in self._preset_names:
            item = QListWidgetItem(name)
            self._preset_list.addItem(item)
        # Default: select first two
        for i in range(min(2, self._preset_list.count())):
            default_item = self._preset_list.item(i)
            if default_item is not None:
                default_item.setSelected(True)
        self._preset_list.setFixedHeight(140)
        lay.addWidget(self._preset_list)
        return grp

    def _build_noise_group(self) -> QGroupBox:
        grp = QGroupBox("Noise Settings")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Amplitude:"))
        self._amp_spin = QDoubleSpinBox()
        self._amp_spin.setRange(0.0, 10.0)
        self._amp_spin.setSingleStep(0.01)
        self._amp_spin.setValue(0.1)
        self._amp_spin.setDecimals(3)
        row1.addWidget(self._amp_spin)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Trials per preset:"))
        self._trials_spin = QSpinBox()
        self._trials_spin.setRange(2, 500)
        self._trials_spin.setValue(30)
        row2.addWidget(self._trials_spin)
        lay.addLayout(row2)

        return grp

    def _build_run_group(self) -> QGroupBox:
        grp = QGroupBox("Run")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        btns = QHBoxLayout()
        self._run_btn = QPushButton("Compare Presets")
        self._run_btn.clicked.connect(self._on_run)
        btns.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btns.addWidget(self._cancel_btn)

        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        btns.addWidget(self._export_btn)
        lay.addLayout(btns)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        lay.addWidget(self._progress)

        self._status = QLabel("Select presets and click Compare")
        lay.addWidget(self._status)
        return grp

    def _build_chart_group(self) -> QGroupBox:
        grp = QGroupBox("Mean ± Std Tip Speed by Preset")
        lay = QVBoxLayout(grp)
        fig = Figure(figsize=(6, 2.5), facecolor="#12121e")
        self._canvas = FigureCanvasQTAgg(fig)
        self._ax = fig.add_subplot(111)
        self._ax.set_facecolor("#1a1a2e")
        self._ax.tick_params(colors="#8080b0", labelsize=9)
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#303050")
        lay.addWidget(self._canvas)
        return grp

    def _build_results_group(self) -> QGroupBox:
        grp = QGroupBox("Results (ranked by CV — lower = more robust)")
        lay = QVBoxLayout(grp)
        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setFixedHeight(120)
        lay.addWidget(self._results_text)
        return grp

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        selected = [item.text() for item in self._preset_list.selectedItems()]
        if len(selected) < 2:
            self._status.setText("Select at least 2 presets")
            return
        if len(selected) > 4:
            self._status.setText("Select at most 4 presets")
            return

        config = PerturbationConfig(
            n_trials=self._trials_spin.value(),
            noise_amplitude=self._amp_spin.value(),
            noise_type="white",
        )

        jobs = [
            (name, self._get_coeffs(name), self._simulate_fn, self._extract_fn)
            for name in selected
        ]
        self._total_trials = len(selected) * config.n_trials
        self._completed_trials = 0
        self._results = []

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._export_btn.setEnabled(False)
        self._progress.setValue(0)
        self._status.setText(f"Comparing {len(selected)} presets…")
        self._results_text.clear()

        worker = _ComparisonWorker(jobs, config)
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.preset_progress.connect(self._on_preset_progress)
        worker.preset_done.connect(self._on_preset_done)
        worker.all_done.connect(self._on_all_done)
        worker.all_done.connect(thread.quit)
        worker.error.connect(self._on_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)

        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._status.setText("Cancelling…")

    def _on_preset_progress(self, name: str, trial: int) -> None:
        if not (name is not None):
            raise ValueError("name must be provided")
        self._completed_trials += 1
        pct = int(100 * self._completed_trials / max(self._total_trials, 1))
        self._progress.setValue(pct)
        self._status.setText(f"{name}: trial {trial}")

    def _on_preset_done(self, name: str, summary: dict) -> None:
        self._results.append((name, summary))

    def _on_all_done(self, results: list) -> None:
        if not (results is not None):
            raise ValueError("results must be provided")
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        if not results:
            self._status.setText("No results — all trials failed")
            return
        self._results = results
        self._display_results(results)
        if _HAS_MPL:
            self._update_chart(results)
        self._export_btn.setEnabled(True)
        self._status.setText(f"Done — {len(results)} presets compared")

    def _on_error(self, msg: str) -> None:
        if not (msg is not None):
            raise ValueError("msg must be provided")
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._status.setText(f"Error: {msg}")

    # ------------------------------------------------------------------
    # Results display
    # ------------------------------------------------------------------

    def _display_results(self, results: list[tuple[str, dict]]) -> None:
        if not (results is not None):
            raise ValueError("results must be provided")
        sorted_results = sorted(results, key=lambda x: x[1]["tip_speed_cv"])
        _hdr = f"{'Preset':<35} {'Mean':>8} {'Std':>8} {'CV%':>7} {'Min':>8} {'Max':>8}"
        lines = [_hdr]
        lines.append("-" * 78)
        for i, (name, s) in enumerate(sorted_results):
            marker = " ★" if i == 0 else ""
            short = name[:34]
            lines.append(
                f"{short:<35} {s['tip_speed_mean']:>7.3f}  "
                f"{s['tip_speed_std']:>7.3f}  "
                f"{s['tip_speed_cv'] * 100:>6.2f}%  "
                f"{s['tip_speed_min']:>7.3f}  "
                f"{s['tip_speed_max']:>7.3f}"
                f"{marker}"
            )
        self._results_text.setPlainText("\n".join(lines))

    def _update_chart(self, results: list[tuple[str, dict]]) -> None:
        if not (results is not None):
            raise ValueError("results must be provided")
        self._ax.clear()
        self._ax.set_facecolor("#1a1a2e")
        names = [r[0][:20] for r in results]
        means = [r[1]["tip_speed_mean"] for r in results]
        stds = [r[1]["tip_speed_std"] for r in results]
        x = np.arange(len(names))
        colors = ["#5555b0", "#b05555", "#55b055", "#b0b055"]
        self._ax.bar(
            x,
            means,
            yerr=stds,
            color=colors[: len(names)],
            edgecolor="#303070",
            capsize=5,
            error_kw={"ecolor": "#d0d0d0", "linewidth": 1.5},
        )
        self._ax.set_xticks(x)
        self._ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        self._ax.set_ylabel("Tip speed (m/s)", color="#8080b0", fontsize=9)
        self._ax.tick_params(colors="#8080b0", labelsize=9)
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#303050")
        self._canvas.draw()

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Comparison", "swing_comparison.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Preset",
                    "Mean (m/s)",
                    "Std (m/s)",
                    "CV (%)",
                    "Min (m/s)",
                    "Max (m/s)",
                    "N trials",
                ]
            )
            sorted_results = sorted(self._results, key=lambda x: x[1]["tip_speed_cv"])
            for name, s in sorted_results:
                writer.writerow(
                    [
                        name,
                        f"{s['tip_speed_mean']:.4f}",
                        f"{s['tip_speed_std']:.4f}",
                        f"{s['tip_speed_cv'] * 100:.3f}",
                        f"{s['tip_speed_min']:.4f}",
                        f"{s['tip_speed_max']:.4f}",
                        s["n_trials"],
                    ]
                )
        self._status.setText(f"Exported to {path}")

"""
Monte Carlo noise injection panel for swing consistency analysis.

Provides a PyQt6 widget for configuring and running perturbation analysis:
- Noise type selector (white / pink / brown)
- Amplitude slider + spin box
- Trial count spinner
- Optional seed for reproducibility
- Run / Cancel buttons with progress bar
- Results display (mean ± std, CV, min/max)
- Histogram of tip-speed distribution via embedded matplotlib

Design by Contract
------------------
- run_batch() must have a valid simulate_fn and extract_fn before calling.
- All results displayed must be finite (enforced by variability_summary).

DRY
---
- Background threading reuses the QObject worker pattern from simulation_panel.py.
- Styling constants match optimization_widget.py dark theme.

Closes #1284
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..perturbation_analysis import (
    PerturbationConfig,
    variability_summary,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional matplotlib for histogram
# ---------------------------------------------------------------------------

try:
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_STYLE = """
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
QComboBox {
    background: #1a1a2a; color: #b0b0e8; border: 1px solid #303050;
    border-radius: 2px; font-size: 11px; padding: 2px;
}
QComboBox::drop-down { border: 0; }
QProgressBar {
    background: #1a1a2a; color: #b0b0e8; border: 1px solid #303050;
    border-radius: 2px; text-align: center; font-size: 11px;
}
QProgressBar::chunk { background: #3a3a80; }
"""


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


class _PerturbWorker(QObject):
    """Runs batch_perturb_and_simulate on a background thread.

    Signals
    -------
    progress(int) : trial number completed (1-based)
    finished(list) : batch results list
    error(str) : error message on failure
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        base_coeffs: list[list[float]],
        config: PerturbationConfig,
        simulate_fn: Callable,
        extract_fn: Callable,
    ) -> None:
        assert base_coeffs is not None, "base_coeffs must be provided"
        super().__init__()
        self._base_coeffs = base_coeffs
        self._config = config
        self._simulate_fn = simulate_fn
        self._extract_fn = extract_fn
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        """Run all trials, emitting progress after each."""
        from ..perturbation_analysis import perturb_torque_coeffs

        results = []
        base_seed = self._config.seed if self._config.seed is not None else 0
        for i in range(self._config.n_trials):
            if self._cancelled:
                break
            trial_seed = base_seed + i
            try:
                perturbed = perturb_torque_coeffs(
                    self._base_coeffs,
                    noise_amplitude=self._config.noise_amplitude,
                    noise_type=self._config.noise_type,
                    seed=trial_seed,
                )
                sim_result = self._simulate_fn(perturbed)
                metrics = self._extract_fn(sim_result)
                results.append(metrics)
            except (ValueError, RuntimeError, FloatingPointError, AssertionError):
                logger.warning("Trial %d failed, skipping", i, exc_info=True)
            self.progress.emit(i + 1)

        self.finished.emit(results)


# ---------------------------------------------------------------------------
# Perturbation panel widget
# ---------------------------------------------------------------------------


class PerturbationPanel(QWidget):
    """Widget for Monte Carlo noise injection and swing consistency analysis.

    Parameters
    ----------
    parent : QWidget, optional

    Usage
    -----
    Call :meth:`set_simulation_callbacks` before the user can click "Run Batch".
    The panel then dispatches batch runs using the provided callables.

    Design by Contract
    ------------------
    Pre:  set_simulation_callbacks() must be called with non-None callables
          before run_batch() is invoked.
    Post: displayed statistics are always finite (enforced by variability_summary).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._simulate_fn: Callable | None = None
        self._extract_fn: Callable | None = None
        self._thread: QThread | None = None
        self._worker: _PerturbWorker | None = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_simulation_callbacks(
        self,
        simulate_fn: Callable[[list[list[float]]], Any],
        extract_fn: Callable[[Any], dict],
    ) -> None:
        """Register the simulation and metric-extraction callables.

        Parameters
        ----------
        simulate_fn : callable(coeffs) -> result
            Takes perturbed polynomial coefficients (list of lists) and
            returns a simulation result object.
        extract_fn : callable(result) -> dict
            Extracts {'tip_speed_final': float, 'tip_position_final': array}
            from a simulation result.
        """
        assert simulate_fn is not None, "simulate_fn must not be None"
        assert extract_fn is not None, "extract_fn must not be None"
        self._simulate_fn = simulate_fn
        self._extract_fn = extract_fn
        self._run_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setStyleSheet(_STYLE)
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        root.addWidget(self._build_config_group())
        root.addWidget(self._build_run_group())
        root.addWidget(self._build_results_group())
        if _HAS_MPL:
            root.addWidget(self._build_histogram_group())
        root.addStretch()

    def _build_config_group(self) -> QGroupBox:
        grp = QGroupBox("Noise Configuration")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        # Noise type
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Noise type:"))
        self._noise_combo = QComboBox()
        self._noise_combo.addItems(["white", "pink", "brown"])
        row1.addWidget(self._noise_combo)
        lay.addLayout(row1)

        # Amplitude
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Amplitude:"))
        self._amp_spin = QDoubleSpinBox()
        self._amp_spin.setRange(0.0, 10.0)
        self._amp_spin.setSingleStep(0.01)
        self._amp_spin.setValue(0.1)
        self._amp_spin.setDecimals(3)
        row2.addWidget(self._amp_spin)
        lay.addLayout(row2)

        # Trials
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Trials:"))
        self._trials_spin = QSpinBox()
        self._trials_spin.setRange(2, 2000)
        self._trials_spin.setValue(50)
        row3.addWidget(self._trials_spin)
        lay.addLayout(row3)

        # Seed
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Seed (0=random):"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setValue(0)
        row4.addWidget(self._seed_spin)
        lay.addLayout(row4)

        return grp

    def _build_run_group(self) -> QGroupBox:
        grp = QGroupBox("Run")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Batch")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)
        lay.addLayout(btn_row)

        self._compare_btn = QPushButton("Compare Presets…")
        self._compare_btn.setEnabled(False)
        self._compare_btn.clicked.connect(self._on_compare)
        lay.addWidget(self._compare_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        lay.addWidget(self._progress)

        self._status_label = QLabel("Ready")
        lay.addWidget(self._status_label)

        return grp

    def _build_results_group(self) -> QGroupBox:
        grp = QGroupBox("Results")
        lay = QVBoxLayout(grp)
        lay.setSpacing(2)

        self._result_labels: dict[str, QLabel] = {}
        for key in ("Mean", "Std", "CV", "Min", "Max", "Trials"):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{key}:"))
            lbl = QLabel("—")
            lbl.setStyleSheet("color: #d0d0f0; font-family: monospace;")
            row.addWidget(lbl)
            lay.addLayout(row)
            self._result_labels[key] = lbl

        return grp

    def _build_histogram_group(self) -> QGroupBox:
        grp = QGroupBox("Tip Speed Distribution")
        lay = QVBoxLayout(grp)
        fig = Figure(figsize=(4, 2.5), facecolor="#12121e")
        self._canvas = FigureCanvasQTAgg(fig)
        self._ax = fig.add_subplot(111)
        self._ax.set_facecolor("#1a1a2e")
        self._ax.tick_params(colors="#8080b0", labelsize=9)
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#303050")
        self._ax.set_xlabel("Tip speed (m/s)", color="#8080b0", fontsize=9)
        self._ax.set_ylabel("Count", color="#8080b0", fontsize=9)
        lay.addWidget(self._canvas)
        return grp

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        assert self._simulate_fn is not None, "simulate_fn not set"
        assert self._extract_fn is not None, "extract_fn not set"

        seed_val = self._seed_spin.value()
        config = PerturbationConfig(
            n_trials=self._trials_spin.value(),
            noise_type=self._noise_combo.currentText(),
            noise_amplitude=self._amp_spin.value(),
            seed=seed_val if seed_val > 0 else None,
        )

        # The base_coeffs will be retrieved via a callback if set, else empty list
        # Use a placeholder — caller must wire set_base_coeffs_fn if needed
        base_coeffs: list[list[float]] = []
        if self._get_coeffs_fn is not None:
            base_coeffs = self._get_coeffs_fn()

        if not base_coeffs:
            self._status_label.setText("No coefficients — run a simulation first")
            return

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setValue(0)
        self._status_label.setText("Running…")
        self._clear_results()

        worker = _PerturbWorker(
            base_coeffs,
            config,
            self._simulate_fn,
            self._extract_fn,
        )
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)

        self._thread = thread
        self._worker = worker
        self._n_trials = config.n_trials
        thread.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._status_label.setText("Cancelling…")

    def _on_progress(self, trial: int) -> None:
        assert trial is not None, "trial must be provided"
        n = getattr(self, "_n_trials", 1)
        pct = int(100 * trial / max(n, 1))
        self._progress.setValue(pct)
        self._status_label.setText(f"Trial {trial} / {n}")

    def _on_finished(self, results: list) -> None:
        assert results is not None, "results must be provided"
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        if not results:
            self._status_label.setText("No successful trials")
            return
        summary = variability_summary(results)
        self._display_summary(summary)
        if _HAS_MPL:
            self._update_histogram([r["tip_speed_final"] for r in results])
        self._status_label.setText(f"Done — {summary['n_trials']} trials completed")

    def _on_error(self, msg: str) -> None:
        assert msg is not None, "msg must be provided"
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._status_label.setText(f"Error: {msg}")
        logger.error("Perturbation batch error: %s", msg)

    # ------------------------------------------------------------------
    # Result display helpers
    # ------------------------------------------------------------------

    def _display_summary(self, summary: dict) -> None:
        assert summary is not None, "summary must be provided"
        mean = summary["tip_speed_mean"]
        std = summary["tip_speed_std"]
        cv = summary["tip_speed_cv"]
        mn = summary["tip_speed_min"]
        mx = summary["tip_speed_max"]
        n = summary["n_trials"]
        self._result_labels["Mean"].setText(f"{mean:.3f} m/s")
        self._result_labels["Std"].setText(f"{std:.3f} m/s")
        self._result_labels["CV"].setText(f"{cv * 100:.2f} %")
        self._result_labels["Min"].setText(f"{mn:.3f} m/s")
        self._result_labels["Max"].setText(f"{mx:.3f} m/s")
        self._result_labels["Trials"].setText(str(n))

    def _clear_results(self) -> None:
        for lbl in self._result_labels.values():
            lbl.setText("—")

    def _update_histogram(self, speeds: list[float]) -> None:
        assert speeds is not None, "speeds must be provided"
        self._ax.clear()
        self._ax.set_facecolor("#1a1a2e")
        self._ax.hist(speeds, bins=20, color="#5555b0", edgecolor="#303070")
        self._ax.set_xlabel("Tip speed (m/s)", color="#8080b0", fontsize=9)
        self._ax.set_ylabel("Count", color="#8080b0", fontsize=9)
        self._ax.tick_params(colors="#8080b0", labelsize=9)
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#303050")
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Coefficients source
    # ------------------------------------------------------------------

    #: Optional callable that returns the current torque polynomial coefficients.
    #: Must be set by the parent panel before the user clicks Run Batch.
    _get_coeffs_fn: Callable[[], list[list[float]]] | None = None

    #: Optional callable for preset comparison — returns preset names.
    _get_preset_names_fn: Callable[[], list[str]] | None = None
    #: Optional callable — returns coefficients for a named preset.
    _get_coeffs_for_preset_fn: Callable[[str], list[list[float]]] | None = None

    def set_coeffs_source(self, fn: Callable[[], list[list[float]]]) -> None:
        """Register a callable that returns current torque polynomial coefficients.

        Parameters
        ----------
        fn : callable() -> list[list[float]]
            Returns the per-joint polynomial coefficient lists for the
            currently loaded simulation.
        """
        assert callable(fn), "fn must be callable"
        self._get_coeffs_fn = fn

    def set_preset_source(
        self,
        get_names_fn: Callable[[], list[str]],
        get_coeffs_fn: Callable[[str], list[list[float]]],
    ) -> None:
        """Register preset-related callables for the comparison dialog.

        Parameters
        ----------
        get_names_fn : callable() -> list[str]
            Returns the list of available preset names.
        get_coeffs_fn : callable(name) -> list[list[float]]
            Returns the polynomial coefficient lists for a named preset.
        """
        assert callable(get_names_fn), "get_names_fn must be callable"
        assert callable(get_coeffs_fn), "get_coeffs_fn must be callable"
        self._get_preset_names_fn = get_names_fn
        self._get_coeffs_for_preset_fn = get_coeffs_fn
        self._compare_btn.setEnabled(True)

    def _on_compare(self) -> None:
        """Open the swing robustness comparison dialog."""
        if self._get_preset_names_fn is None or self._get_coeffs_for_preset_fn is None:
            return
        if self._simulate_fn is None or self._extract_fn is None:
            return
        from .swing_comparison_dialog import SwingComparisonDialog

        names = self._get_preset_names_fn()
        if len(names) < 2:
            return
        dlg = SwingComparisonDialog(
            preset_names=names,
            get_coeffs_for_preset=self._get_coeffs_for_preset_fn,
            simulate_fn=self._simulate_fn,
            extract_fn=self._extract_fn,
            parent=self,
        )
        dlg.exec()

"""Cross-Engine Perturbation Comparison Dashboard.

Addresses GH2020: provides a PyQt6 interactive dashboard (and optional CLI)
for configuring and running ``CrossEnginePerturbationRunner`` across multiple
physics engines, then visualising the results as Matplotlib bar charts.

Features
--------
- Engine selection checkboxes (mujoco, drake, pinocchio, pendulum stub)
- Configurable ``CrossEngineSimConfig`` parameters via spin-boxes
- Robustness Score bar chart (1 - CV, per engine, per metric)
- CV bar chart (per metric across engines)
- CLI mode via ``--no-gui`` for headless / CI use

Design by Contract
------------------
- At least one engine must be checked before "Run Comparison" is clicked.
- All heavy imports (PyQt6, Matplotlib, engine modules) are deferred to
  runtime to ensure strict lazy-loading at import time.
- No print() calls; all output uses logging.

DRY
---
- The compute path (engine construction, comparison run, CV summary,
  robustness score) lives in ``src.shared.python.analysis.cross_engine``
  (issue #7455) and is shared with the web API.
- Style constants match existing dark-theme widgets in the codebase.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.analysis.cross_engine import (
    BACKEND_STUB as _BACKEND_STUB,
    ENGINE_NAMES as _ENGINE_NAMES,
    build_engine,
    cv_values as _cv_values,
    per_engine_robustness as _per_engine_robustness,
    robustness_score as _robustness_score,
    run_comparison_with_provenance as _run_with_provenance,
    substituted_engines as _substituted_engines,
    try_build_real_engine as _try_build_real_engine,
)
from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
    CrossEngineSimConfig,
    CrossEngineRunResult,
)
from src.shared.python.plot_style import (
    MarkerShape,
    MarkerStyle,
    MatplotlibMarkerRenderer,
    PaletteColor,
    PlotStyleSet,
    PlotStyleSpec,
    PresetLibrary,
)
from src.shared.python.logging_pkg.logging_config import get_logger

# Re-exported for backwards compatibility: tests and downstream callers
# reference ``cross_engine_dashboard._StubEngine`` (compute moved to the
# shared service in issue #7455).
from src.launchers.cross_engine_dashboard_stubs import (  # noqa: F401 - compat re-export
    StubEngine as _StubEngine,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from PyQt6.QtWidgets import QMainWindow

logger = get_logger(__name__)


def _build_engine(name: str) -> Any:
    """Build an engine, honouring module-level ``_try_build_real_engine``.

    Thin delegate to the shared service (issue #7455). The module-global
    indirection is kept so existing tests can patch
    ``cross_engine_dashboard._try_build_real_engine``.
    """
    return build_engine(name, try_real=_try_build_real_engine)


_DEFAULT_ENGINE_CONVENTION = {
    "velocity": "engine-native",
    "units": "reported SI",
}

_ENGINE_CONVENTIONS: dict[str, dict[str, str]] = {
    "drake": {
        "velocity": "v generalized velocity",
        "units": "SI; joint angles rad",
    },
    "mujoco": {
        "velocity": "qvel tangent-space",
        "units": "SI; joint angles rad",
    },
    "opensim": {
        "velocity": "coordinate speeds",
        "units": "SI; rotational coordinates deg",
    },
    "pendulum_stub": {
        "velocity": "qdot state",
        "units": "SI; angles rad",
    },
    "pinocchio": {
        "velocity": "tangent vector",
        "units": "SI; joint angles rad",
    },
    "simscape": {
        "velocity": "Simscape logged derivatives",
        "units": "SI; angular channels rad",
    },
}


def _engine_convention(name: str) -> dict[str, str]:
    """Return user-facing convention metadata for an engine result."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string; got {name!r}")
    return _ENGINE_CONVENTIONS.get(name, _DEFAULT_ENGINE_CONVENTION)


def _is_substituted_stub(name: str, backend: str | None) -> bool:
    """True when a requested real engine silently ran as the 2-DOF stub."""
    return backend == _BACKEND_STUB and name != "pendulum_stub"


def _format_engine_result_label(name: str, backend: str | None = None) -> str:
    """Return the comparison label with velocity convention and units.

    A stub-backed series is never labeled as the real engine (#8817): when
    ``backend`` reports a stub substitution the label says so and does not
    claim the real engine's conventions.
    """
    if _is_substituted_stub(name, backend):
        return f"{name} (unavailable — stub)\nsynthetic 2-DOF stub output"
    convention = _engine_convention(name)
    return f"{name}\nvelocity: {convention['velocity']}; units: {convention['units']}"


def _format_engine_result_log_label(name: str, backend: str | None = None) -> str:
    """Return a single-line result label for logs and status text."""
    if _is_substituted_stub(name, backend):
        return f"{name} [UNAVAILABLE — ran as synthetic 2-DOF stub]"
    convention = _engine_convention(name)
    return f"{name} [velocity: {convention['velocity']}; units: {convention['units']}]"


# Curated palette indices for trajectory overlays.  ``tab10`` is the
# colour-blind-friendly default; engines listed here get a deterministic
# entry, anything outside the table falls back to a hash-stable index so
# the overlay still renders.
#
# Five core engines are mapped explicitly per the issue acceptance
# criteria; ``pendulum_stub`` is included so the default GUI selection
# (which checks ``pendulum_stub``) also gets a deterministic colour.
_TRAJECTORY_PALETTE_NAME: str = "tab10"

_ENGINE_PALETTE_INDICES: dict[str, int] = {
    "drake": 0,
    "mujoco": 1,
    "pinocchio": 2,
    "opensim": 3,
    "simscape": 4,
    "pendulum_stub": 7,
}

# Distinct shapes per engine so colour-blind users can still discriminate
# overlapping traces.  Sphere is the default fallback.
_ENGINE_SHAPES: dict[str, MarkerShape] = {
    "drake": MarkerShape.SPHERE,
    "mujoco": MarkerShape.CUBE,
    "pinocchio": MarkerShape.DIAMOND,
    "opensim": MarkerShape.STAR,
    "simscape": MarkerShape.PLUS,
    "pendulum_stub": MarkerShape.POINT,
}


def _engine_palette_index(name: str) -> int:
    """Return the ``tab10`` palette index for ``name`` (deterministic)."""
    if name in _ENGINE_PALETTE_INDICES:
        return _ENGINE_PALETTE_INDICES[name]
    # Stable fallback — wraps modulo 10 for any unknown engine.
    return abs(hash(name)) % 10


def _default_marker_style_template() -> MarkerStyle | None:
    """Return the first style from the ``"default"`` preset, if available.

    The packaged preset is loaded lazily and missing-preset / load-error
    conditions degrade gracefully to ``None`` so the dashboard never
    breaks when run in a stripped environment.
    """
    try:
        library = PresetLibrary.default()
    except Exception:  # pragma: no cover - missing presets package  # noqa: BLE001
        logger.debug("PresetLibrary.default() unavailable; using fallback style")
        return None
    if "default" not in library:
        return None
    preset = library["default"]
    if not preset.entries:
        return None
    return preset.entries[0].style


def _build_engine_marker_style(
    name: str,
    *,
    shape_per_engine: bool = True,
    template: MarkerStyle | None = None,
) -> MarkerStyle:
    """Construct a per-engine :class:`MarkerStyle` for trajectory overlays.

    Parameters
    ----------
    name:
        Engine identifier.
    shape_per_engine:
        When ``True`` each engine uses its distinct :class:`MarkerShape`
        (colour-blind aid). When ``False`` every engine uses
        :data:`MarkerShape.SPHERE`.
    template:
        Optional template style (typically loaded from
        ``PresetLibrary.default()["default"]``); its size / edge / opacity
        attributes are reused so the overlay matches the global theme.
        ``None`` falls back to hard-coded sane defaults.

    Returns
    -------
    MarkerStyle
        A new frozen :class:`MarkerStyle` with a :class:`PaletteColor`
        fill keyed off ``tab10``.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string; got {name!r}")
    fill = PaletteColor(
        palette_name=_TRAJECTORY_PALETTE_NAME,
        palette_index=_engine_palette_index(name),
    )
    shape = (
        _ENGINE_SHAPES.get(name, MarkerShape.SPHERE)
        if shape_per_engine
        else (MarkerShape.SPHERE)
    )
    if template is None:
        return MarkerStyle(
            shape=shape,
            size_px=6.0,
            edge_color="#101020",
            edge_width=0.5,
            fill_color=fill,
            opacity=1.0,
        )
    return MarkerStyle(
        shape=shape,
        size_px=float(template.size_px),
        edge_color=str(template.edge_color),
        edge_width=float(template.edge_width),
        fill_color=fill,
        opacity=float(template.opacity),
    )


def _render_trajectory_overlay(
    ax: Axes,
    trajectories: dict[str, np.ndarray],
    renderer: MatplotlibMarkerRenderer,
    *,
    shape_per_engine: bool = True,
    template: MarkerStyle | None = None,
) -> dict[str, str]:
    """Plot one trajectory per engine on ``ax`` using ``renderer``.

    A single :class:`MatplotlibMarkerRenderer` is reused across every
    engine series (DRY — see issue #4810) by binding it to ``ax`` and
    calling :meth:`MatplotlibMarkerRenderer.add_markers` per engine.

    Parameters
    ----------
    ax:
        Matplotlib 2D ``Axes`` already attached to a figure.
    trajectories:
        Mapping ``engine_name -> (T, D) ndarray`` of trajectories where
        ``D`` is at least 2.  The first two columns are plotted.
    renderer:
        :class:`MatplotlibMarkerRenderer` instance bound to ``ax``.  The
        same instance must be reused — ``RuntimeError`` if its default
        axes don't match.
    shape_per_engine:
        Forwarded to :func:`_build_engine_marker_style`.
    template:
        Forwarded to :func:`_build_engine_marker_style`.

    Returns
    -------
    dict[str, str]
        Mapping engine_name -> renderer handle (useful for later
        :meth:`MatplotlibMarkerRenderer.remove`).

    Design by Contract
    ------------------
    Pre:  trajectories is non-empty and every value has shape (T, D >= 2)
    Pre:  renderer's default axes is ``ax`` (DRY enforcement)
    Post: returns one handle per engine
    """
    if not trajectories:
        raise ValueError("trajectories must be non-empty")
    if getattr(renderer, "_default_ax", None) is not ax:
        raise RuntimeError(
            "MatplotlibMarkerRenderer must be bound to the supplied ax "
            "(DRY: a single renderer instance is required across overlays)"
        )
    handles: dict[str, str] = {}
    for engine_name, traj in trajectories.items():
        arr = np.asarray(traj, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                f"trajectory for {engine_name!r} must have shape (T, D>=2); "
                f"got {arr.shape}"
            )
        # Plot first two DOF as 2D scatter overlay.
        positions = arr[:, :2]
        style = _build_engine_marker_style(
            engine_name,
            shape_per_engine=shape_per_engine,
            template=template,
        )
        handle = renderer.add_markers(positions, style, label=engine_name)
        handles[engine_name] = handle
    return handles


def build_dashboard_style_set(
    engine_names: list[str] | tuple[str, ...] = _ENGINE_NAMES,
    *,
    shape_per_engine: bool = True,
) -> PlotStyleSet:
    """Return a :class:`PlotStyleSet` describing the dashboard overlay styles.

    Useful for persisting the styled session (issue #4810 — round-trip
    test). Each engine becomes a :class:`PlotStyleSpec` with target
    ``"trace:<engine>"`` so callers can save / load the configuration via
    :meth:`PlotStyleSet.save` / :meth:`PlotStyleSet.load`.
    """
    template = _default_marker_style_template()
    entries = tuple(
        PlotStyleSpec(
            name=name,
            target=f"trace:{name}",
            style=_build_engine_marker_style(
                name,
                shape_per_engine=shape_per_engine,
                template=template,
            ),
        )
        for name in engine_names
    )
    return PlotStyleSet(entries=entries)


def _run_headless(
    engine_names: list[str],
    config: CrossEngineSimConfig,
) -> dict[str, float]:
    """Run a cross-engine comparison without a GUI and log the results.

    Parameters
    ----------
    engine_names : list of str
        Names of engines to include.
    config : CrossEngineSimConfig
        Simulation configuration.

    Returns
    -------
    dict — CV summary (keys: cv_total_energy_final, cv_end_effector_speed_final,
    cv_peak_end_effector_speed)

    Design by Contract
    ------------------
    Pre:  len(engine_names) > 0
    Pre:  config is a valid CrossEngineSimConfig
    Post: returns a dict with the three CV keys
    """
    results, cv_summary, backends = _run_with_provenance(engine_names, config)

    substituted = _substituted_engines(backends)
    if substituted:
        logger.warning(
            "Engines %s are unavailable — results below are synthetic 2-DOF "
            "stub output, NOT real engine physics",
            substituted,
        )
    logger.info("=== Cross-Engine Perturbation Comparison Results ===")
    for eng_name, result in results.items():
        logger.info(
            "  %s: mean_energy=%.4f, mean_speed=%.4f, mean_peak=%.4f",
            _format_engine_result_log_label(eng_name, backends.get(eng_name)),
            result.mean_total_energy_final,
            result.mean_end_effector_speed_final,
            result.mean_peak_end_effector_speed,
        )
    logger.info("CV Summary:")
    for key, cv in cv_summary.items():
        logger.info("  %s: %.4f", key, cv)

    return cv_summary


_CV_METRIC_LABELS = ("Energy", "Speed", "Peak Speed")


def _load_dashboard_qt_bindings() -> Any:
    """Return Qt classes used by the lazy dashboard window factory."""
    from PyQt6.QtCore import (  # noqa: PLC0415
        QObject,
        QRunnable,
        QThreadPool,
        pyqtSignal,
    )
    from PyQt6.QtWidgets import (  # noqa: PLC0415
        QCheckBox,
        QDoubleSpinBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    return SimpleNamespace(
        QCheckBox=QCheckBox,
        QDoubleSpinBox=QDoubleSpinBox,
        QGroupBox=QGroupBox,
        QHBoxLayout=QHBoxLayout,
        QLabel=QLabel,
        QMainWindow=QMainWindow,
        QObject=QObject,
        QPushButton=QPushButton,
        QRunnable=QRunnable,
        QSpinBox=QSpinBox,
        QThreadPool=QThreadPool,
        QVBoxLayout=QVBoxLayout,
        QWidget=QWidget,
        pyqtSignal=pyqtSignal,
    )


def _load_dashboard_mpl_bindings() -> Any:
    """Return Matplotlib bindings, preserving no-Matplotlib GUI fallback."""
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: PLC0415
        from matplotlib.figure import Figure  # noqa: PLC0415

        Figure(figsize=(1, 1)).add_subplot(111)
    except ImportError:
        return SimpleNamespace(has_mpl=False, FigureCanvasQTAgg=None, Figure=None)
    except (TypeError, ValueError, RuntimeError):
        return SimpleNamespace(has_mpl=False, FigureCanvasQTAgg=None, Figure=None)
    return SimpleNamespace(
        has_mpl=True,
        FigureCanvasQTAgg=FigureCanvasQTAgg,
        Figure=Figure,
    )


def _draw_bar_value_labels(
    ax: Any,
    bars: Any,
    values: list[float],
    *,
    y_offset: float,
    fmt: str,
) -> None:
    """Annotate a bar chart with centered numeric labels."""
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            fmt.format(val),
            ha="center",
            va="bottom",
            color="#d0d0f0",
            fontsize=8,
        )


_AGGREGATE_ROBUSTNESS_LABEL = "aggregate\n(all selected engines)"


def _robustness_chart_series(
    engine_names: list[str],
    cv_summary: dict[str, float],
    backends: dict[str, str] | None = None,
    robustness_per_engine: dict[str, float] | None = None,
) -> tuple[list[str], list[float]]:
    """Return honest (labels, values) for the robustness chart (#8816, #8817).

    Design by Contract
    ------------------
    Post: an aggregate score is never replicated across per-engine bars —
    when genuine per-engine scores are unavailable the chart collapses to
    a single clearly-labeled aggregate bar. Stub-backed engines are
    labeled as substitutions, never as the real engine.
    """
    if robustness_per_engine:
        backends = backends or {}
        labels = [
            _format_engine_result_label(name, backends.get(name))
            for name in engine_names
        ]
        values = [robustness_per_engine.get(name, 0.0) for name in engine_names]
        return labels, values
    return [_AGGREGATE_ROBUSTNESS_LABEL], [_robustness_score(_cv_values(cv_summary))]


def _draw_robustness_chart(
    ax: Any,
    canvas: Any,
    colors: Any,
    bar_labels: list[str],
    robustness_values: list[float],
    style_ax: Any,
) -> None:
    """Draw the robustness score chart on the supplied axes/canvas."""
    ax.clear()
    ax.set_facecolor("#1a1a2e")
    x = np.arange(len(bar_labels))
    bars = ax.bar(
        x,
        robustness_values,
        color="#5555b0",
        edgecolor="#303070",
        width=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Robustness Score", fontsize=9)
    ax.axhline(0.5, color="#ff6060", linewidth=0.8, linestyle="--")
    style_ax(ax, colors)
    _draw_bar_value_labels(ax, bars, robustness_values, y_offset=0.02, fmt="{:.2f}")
    canvas.draw()


def _draw_cv_chart(
    ax: Any,
    canvas: Any,
    colors: Any,
    cv_values: list[float],
    style_ax: Any,
) -> None:
    """Draw the aggregate coefficient-of-variation chart."""
    ax.clear()
    ax.set_facecolor("#1a1a2e")
    x = np.arange(len(_CV_METRIC_LABELS))
    bars = ax.bar(
        x,
        cv_values,
        color="#8040c0",
        edgecolor="#502080",
        width=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(_CV_METRIC_LABELS, fontsize=9)
    ax.set_ylabel("CV", fontsize=9)
    ax.axhline(1.0, color="#ff6060", linewidth=0.8, linestyle="--")
    style_ax(ax, colors)
    _draw_bar_value_labels(ax, bars, cv_values, y_offset=0.01, fmt="{:.3f}")
    canvas.draw()


def _create_comparison_worker_class(qt: Any) -> type:
    """Build the QRunnable worker class against the lazy Qt bindings."""

    class ComparisonWorkerSignals(qt.QObject):
        """Signals for the ComparisonWorker.

        ``finished`` carries (engine_names, cv_summary, trajectories,
        backends, robustness_per_engine) — backends tag each engine
        ``real`` / ``stub_2dof`` so the charts can disclose stub
        substitution (#8817), and robustness_per_engine carries genuine
        per-engine scores (#8816).
        """

        finished = qt.pyqtSignal(list, dict, dict, dict, dict)
        error = qt.pyqtSignal(str)

    class ComparisonWorker(qt.QRunnable):
        """Run cross-engine comparison in a background thread."""

        def __init__(
            self, engine_names: list[str], config: CrossEngineSimConfig
        ) -> None:
            super().__init__()
            self.signals = ComparisonWorkerSignals()
            self.engine_names = engine_names
            self.config = config

        def run(self) -> None:
            """Execute comparison and emit results."""
            try:
                results, cv_summary, backends = _run_with_provenance(
                    self.engine_names, self.config
                )
                trajectories = _trial_zero_trajectories(results)
                robustness = _per_engine_robustness(results)
                self.signals.finished.emit(
                    self.engine_names, cv_summary, trajectories, backends, robustness
                )
            except Exception as e:  # noqa: BLE001
                self.signals.error.emit(str(e))

    return ComparisonWorker


def _trial_zero_trajectories(
    results: dict[str, CrossEngineRunResult],
) -> dict[str, np.ndarray]:
    """Extract first-trial 2D-capable trajectories from run results."""
    trajectories: dict[str, np.ndarray] = {}
    for name, run_result in results.items():
        if not run_result.metrics_per_trial:
            continue
        traj = np.asarray(run_result.metrics_per_trial[0].trajectory_q, dtype=float)
        if traj.ndim == 2 and traj.shape[1] >= 2:
            trajectories[name] = traj
    return trajectories


class _DashboardConfigPanelMixin:
    """Build the dashboard configuration controls."""

    _qt: Any
    _engine_checks: dict[str, Any]
    _trials_spin: Any
    _amp_spin: Any
    _tend_spin: Any
    _dt_spin: Any
    _run_btn: Any
    _status_label: Any

    def _on_run(self) -> None: ...

    def _build_config_panel(self) -> Any:
        qt = self._qt
        panel = qt.QWidget()
        panel.setMinimumWidth(260)
        layout = qt.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._build_engine_group())
        layout.addWidget(self._build_sim_config_group())
        layout.addWidget(self._build_run_group())
        layout.addStretch()
        return panel

    def _build_engine_group(self) -> Any:
        qt = self._qt
        grp = qt.QGroupBox("Engines")
        lay = qt.QVBoxLayout(grp)
        lay.setSpacing(4)
        self._engine_checks = {}
        for name in _ENGINE_NAMES:
            cb = qt.QCheckBox(name)
            cb.setChecked(name == "pendulum_stub")
            self._engine_checks[name] = cb
            lay.addWidget(cb)
        return grp

    def _build_sim_config_group(self) -> Any:
        qt = self._qt
        grp = qt.QGroupBox("Simulation Config")
        lay = qt.QVBoxLayout(grp)
        lay.setSpacing(4)
        self._add_spin_row(lay, "Trials:", "_trials_spin", qt.QSpinBox, (1, 500), 10)
        self._add_spin_row(
            lay, "Amplitude:", "_amp_spin", qt.QDoubleSpinBox, (0.0, 5.0), 0.1, 3, 0.01
        )
        self._add_spin_row(
            lay, "t_end (s):", "_tend_spin", qt.QDoubleSpinBox, (0.1, 10.0), 1.5, 2, 0.1
        )
        self._add_spin_row(
            lay, "dt (s):", "_dt_spin", qt.QDoubleSpinBox, (0.001, 0.1), 0.01, 3, 0.001
        )
        return grp

    def _add_spin_row(
        self,
        layout: Any,
        label: str,
        attr: str,
        spin_cls: Any,
        value_range: tuple[float, float],
        value: float,
        decimals: int | None = None,
        single_step: float | None = None,
    ) -> None:
        qt = self._qt
        row = qt.QHBoxLayout()
        row.addWidget(qt.QLabel(label))
        spin = spin_cls()
        spin.setRange(*value_range)
        spin.setValue(value)
        if single_step is not None:
            spin.setSingleStep(single_step)
        if decimals is not None:
            spin.setDecimals(decimals)
        setattr(self, attr, spin)
        row.addWidget(spin)
        layout.addLayout(row)

    def _build_run_group(self) -> Any:
        qt = self._qt
        grp = qt.QGroupBox("Run")
        lay = qt.QVBoxLayout(grp)
        lay.setSpacing(4)
        self._run_btn = qt.QPushButton("Run Comparison")
        self._run_btn.clicked.connect(self._on_run)
        lay.addWidget(self._run_btn)
        self._status_label = qt.QLabel("Ready")
        lay.addWidget(self._status_label)
        return grp


class _DashboardThemeMixin:
    """Theme helpers shared by chart panel and chart rendering."""

    def _get_theme_colors(self) -> Any:
        """Retrieve the current theme colors mapped as a SimpleNamespace."""
        try:
            from src.shared.python.theme import DARK_THEME, get_theme_manager

            tm = get_theme_manager()
            raw_c = tm.get_current_colors() if tm else None
            if raw_c:
                return SimpleNamespace(
                    bg=raw_c.get("bg", "#1a1d23"),
                    bg_elevated=raw_c.get(
                        "table_header", raw_c.get("group_bg", "#2a2d35")
                    ),
                    text_secondary=raw_c.get(
                        "text_secondary", raw_c.get("label", "#8b949e")
                    ),
                    border_default=raw_c.get("border", "#3a3d45"),
                )
            return DARK_THEME
        except ImportError:
            return SimpleNamespace(
                bg="#12121e",
                bg_elevated="#1a1a2e",
                text_secondary="#8080b0",
                border_default="#303050",
            )

    @staticmethod
    def _style_ax(ax: Any, colors: Any) -> None:
        """Apply theme styling to a Matplotlib axes."""
        ax.tick_params(colors=colors.text_secondary, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors.border_default)
        ax.yaxis.label.set_color(colors.text_secondary)
        ax.xaxis.label.set_color(colors.text_secondary)


class _DashboardChartPanelMixin(_DashboardThemeMixin):
    """Build chart widgets and canvas bindings."""

    _qt: Any
    _mpl: Any
    _ax_tr: Any
    _traj_renderer: MatplotlibMarkerRenderer | None

    def _build_chart_panel(self) -> Any:
        qt = self._qt
        panel = qt.QWidget()
        layout = qt.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        c = self._get_theme_colors()
        self._add_chart_group(layout, "Robustness Score (1 − CV)", "rs", c)
        self._add_chart_group(layout, "Coefficient of Variation per Metric", "cv", c)
        self._add_chart_group(
            layout, "Trajectory Overlay (per-engine, plot_style)", "tr", c
        )
        self._traj_renderer = MatplotlibMarkerRenderer(self._ax_tr)
        return panel

    def _add_chart_group(
        self, layout: Any, title: str, suffix: str, colors: Any
    ) -> None:
        qt = self._qt
        mpl = self._mpl
        group = qt.QGroupBox(title)
        group_layout = qt.QVBoxLayout(group)
        figure = mpl.Figure(figsize=(5, 2.5), facecolor=colors.bg)
        axis = figure.add_axes((0.12, 0.22, 0.82, 0.68))
        canvas = mpl.FigureCanvasQTAgg(figure)
        axis.set_facecolor(colors.bg_elevated)
        self._style_ax(axis, colors)
        setattr(self, f"_canvas_{suffix}", canvas)
        setattr(self, f"_ax_{suffix}", axis)
        group_layout.addWidget(canvas)
        layout.addWidget(group)


class _DashboardRunMixin:
    """Handle dashboard run actions and completion callbacks."""

    _engine_checks: dict[str, Any]
    _status_label: Any
    _tend_spin: Any
    _dt_spin: Any
    _amp_spin: Any
    _trials_spin: Any
    _run_btn: Any
    _comparison_worker_class: Any
    _thread_pool: Any

    def _update_charts(
        self,
        engine_names: list[str],
        cv_summary: dict[str, float],
        backends: dict[str, str] | None = None,
        robustness_per_engine: dict[str, float] | None = None,
    ) -> None: ...
    def _update_trajectory_overlay(
        self, trajectories: dict[str, np.ndarray]
    ) -> None: ...

    def _on_run(self) -> None:
        """Build config, run comparison in background thread."""
        selected = [name for name, cb in self._engine_checks.items() if cb.isChecked()]
        if not selected:
            self._status_label.setText("Select at least one engine")
            logger.warning("No engines selected for comparison")
            return
        try:
            config = CrossEngineSimConfig(
                t_end=self._tend_spin.value(),
                dt=self._dt_spin.value(),
                noise_amplitude=self._amp_spin.value(),
                n_trials=self._trials_spin.value(),
            )
        except ValueError as exc:
            self._status_label.setText(f"Config error: {exc}")
            logger.error("Invalid CrossEngineSimConfig: %s", exc)
            return
        self._run_btn.setEnabled(False)
        self._status_label.setText("Running…")
        worker = self._comparison_worker_class(selected, config)
        worker.signals.finished.connect(self._on_comparison_finished)
        worker.signals.error.connect(self._on_comparison_error)
        if self._thread_pool is not None:
            self._thread_pool.start(worker)

    def _on_comparison_finished(
        self,
        engine_names: list[str],
        cv_summary: dict[str, float],
        trajectories: dict[str, np.ndarray] | None = None,
        backends: dict[str, str] | None = None,
        robustness_per_engine: dict[str, float] | None = None,
    ) -> None:
        """Handle successful comparison completion."""
        self._update_charts(engine_names, cv_summary, backends, robustness_per_engine)
        self._update_trajectory_overlay(trajectories or {})
        substituted = _substituted_engines(backends or {})
        if substituted:
            self._status_label.setText(
                f"Done — {', '.join(substituted)} unavailable (stub used)"
            )
        else:
            self._status_label.setText("Done")
        self._run_btn.setEnabled(True)

    def _on_comparison_error(self, error_msg: str) -> None:
        """Handle comparison failure."""
        self._status_label.setText(f"Error: {error_msg}")
        logger.error("Comparison failed: %s", error_msg)
        self._run_btn.setEnabled(True)


class _DashboardChartUpdateMixin(_DashboardThemeMixin):
    """Update chart canvases from comparison results."""

    _mpl: Any
    _ax_rs: Any
    _canvas_rs: Any
    _ax_cv: Any
    _canvas_cv: Any
    _traj_renderer: MatplotlibMarkerRenderer | None
    _ax_tr: Any
    _traj_handles: dict[str, Any]
    _canvas_tr: Any
    _shape_per_engine: bool
    _style_template: Any

    def _update_charts(
        self,
        engine_names: list[str],
        cv_summary: dict[str, float],
        backends: dict[str, str] | None = None,
        robustness_per_engine: dict[str, float] | None = None,
    ) -> None:
        """Refresh Robustness Score and CV charts from the latest results.

        Charts genuine per-engine robustness when available; otherwise a
        single clearly-labeled aggregate bar (#8816). Stub-substituted
        engines are labeled as such (#8817).
        """
        if not self._mpl.has_mpl or not engine_names:
            return
        colors = self._get_theme_colors()
        values = _cv_values(cv_summary)
        bar_labels, bar_values = _robustness_chart_series(
            engine_names, cv_summary, backends, robustness_per_engine
        )
        _draw_robustness_chart(
            self._ax_rs,
            self._canvas_rs,
            colors,
            bar_labels,
            bar_values,
            self._style_ax,
        )
        _draw_cv_chart(self._ax_cv, self._canvas_cv, colors, values, self._style_ax)

    def _update_trajectory_overlay(
        self,
        trajectories: dict[str, np.ndarray],
    ) -> None:
        """Render per-engine trajectory overlays via plot_style (#4810)."""
        if not self._mpl.has_mpl or self._traj_renderer is None:
            return
        colors = self._get_theme_colors()
        ax = self._ax_tr
        for handle in list(self._traj_handles.values()):
            with contextlib.suppress(KeyError):  # pragma: no cover - defensive
                self._traj_renderer.remove(handle)
        self._traj_handles.clear()
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        self._style_ax(ax, colors)
        if not trajectories:
            self._canvas_tr.draw()
            return
        self._traj_handles = _render_trajectory_overlay(
            ax,
            trajectories,
            self._traj_renderer,
            shape_per_engine=self._shape_per_engine,
            template=self._style_template,
        )
        ax.set_xlabel("q[0]", fontsize=9)
        ax.set_ylabel("q[1]", fontsize=9)
        ax.legend(
            list(self._traj_handles.keys()),
            fontsize=8,
            loc="best",
            facecolor="#1a1a2e",
            edgecolor="#303050",
            labelcolor="#d0d0f0",
        )
        self._canvas_tr.draw()


def CrossEngineDashboardWindow(
    parent: Any | None = None,
    *,
    shape_per_engine: bool = True,
) -> Any:
    """Return the lazily constructed dashboard window instance."""
    cls = _create_dashboard_window_class()
    return cls(parent, shape_per_engine=shape_per_engine)


def _create_dashboard_window_class() -> type:
    """Construct and return the _Window class with deferred Qt/mpl imports.

    Returns
    -------
    type
        A QMainWindow subclass ready to be instantiated.

    Raises
    ------
    ImportError if PyQt6 is not available.
    """
    qt = _load_dashboard_qt_bindings()
    mpl = _load_dashboard_mpl_bindings()
    comparison_worker_class = _create_comparison_worker_class(qt)

    base_cls = QMainWindow if TYPE_CHECKING else qt.QMainWindow

    class _Window(
        base_cls,  # type: ignore[misc, valid-type]
        _DashboardChartUpdateMixin,
        _DashboardRunMixin,
        _DashboardConfigPanelMixin,
        _DashboardChartPanelMixin,
    ):
        """Cross-Engine Perturbation Comparison Dashboard main window."""

        def __init__(
            self,
            parent: Any | None = None,
            *,
            shape_per_engine: bool = True,
        ) -> None:
            super().__init__(parent)
            self._qt = qt
            self._mpl = mpl
            self._comparison_worker_class = comparison_worker_class
            self.setWindowTitle("Cross-Engine Perturbation Comparison Dashboard")
            self.setMinimumSize(900, 620)
            self._apply_dashboard_theme()
            self._shape_per_engine = bool(shape_per_engine)
            self._traj_renderer: MatplotlibMarkerRenderer | None = None
            self._traj_handles: dict[str, str] = {}
            self._style_template: MarkerStyle | None = _default_marker_style_template()
            self._build_window_layout()
            self._thread_pool = qt.QThreadPool.globalInstance()

        def _apply_dashboard_theme(self) -> None:
            """Apply the repository theme when available."""
            try:
                from src.shared.python.theme import apply_theme_to_window
            except ImportError:
                return
            if callable(apply_theme_to_window):
                apply_theme_to_window(self)

        def _build_window_layout(self) -> None:
            """Create the central dashboard layout."""
            central = self._qt.QWidget()
            central.setObjectName("central")
            self.setCentralWidget(central)
            root = self._qt.QHBoxLayout(central)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(8)
            root.addWidget(self._build_config_panel(), stretch=0)
            if self._mpl.has_mpl:
                root.addWidget(self._build_chart_panel(), stretch=1)

    return _Window


def get_dockable_ui() -> object:
    """Return the dashboard window instance for docking in the unified launcher."""
    return _build_qt_window()


def _build_qt_window(*, shape_per_engine: bool = True) -> Any:
    """Build and return the QMainWindow instance (deferred Qt import).

    Parameters
    ----------
    shape_per_engine:
        Forwarded to the window constructor — when ``True`` each engine
        trajectory uses a distinct :class:`MarkerShape` (issue #4810).
    """
    cls = _create_dashboard_window_class()
    return cls(shape_per_engine=shape_per_engine)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the dashboard CLI."""
    parser = argparse.ArgumentParser(
        description="Cross-Engine Perturbation Comparison Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        default=False,
        help="Run headless comparison and print results (no GUI)",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="pendulum_stub",
        help="Comma-separated engine names to compare (e.g. mujoco,pinocchio,pendulum_stub)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of Monte Carlo trials per engine",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.1,
        help="Noise amplitude (standard deviation, N·m)",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=1.5,
        help="Simulation duration in seconds",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Integration timestep in seconds",
    )
    parser.add_argument(
        "--shape-per-engine",
        dest="shape_per_engine",
        action="store_true",
        default=True,
        help="Use a distinct MarkerShape per engine in trajectory overlays "
        "(colour-blind aid; issue #4810)",
    )
    parser.add_argument(
        "--no-shape-per-engine",
        dest="shape_per_engine",
        action="store_false",
        help="Force every engine to share MarkerShape.SPHERE",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the Cross-Engine Perturbation Comparison Dashboard.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to sys.argv[1:]).

    Design by Contract
    ------------------
    Pre:  Python 3.10+
    Post: Runs comparison (headless) or shows GUI window; returns only after
          window is closed (GUI mode) or comparison finishes (headless).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    engine_names = [n.strip() for n in args.engines.split(",") if n.strip()]
    if not engine_names:
        logger.error("No engine names provided via --engines")
        sys.exit(1)

    config = CrossEngineSimConfig(
        t_end=args.t_end,
        dt=args.dt,
        noise_amplitude=args.amplitude,
        n_trials=args.n_trials,
    )

    if args.no_gui:
        _run_headless(engine_names, config)
        return

    # GUI mode
    try:
        from PyQt6.QtWidgets import QApplication  # noqa: PLC0415
    except ImportError:
        logger.warning("PyQt6 not available — falling back to headless mode")
        _run_headless(engine_names, config)
        return

    app = QApplication.instance() or QApplication(sys.argv)
    window = _build_qt_window(shape_per_engine=args.shape_per_engine)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

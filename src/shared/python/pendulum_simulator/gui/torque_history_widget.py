"""
Torque history plot widget — N-DOF aware.

Displays the full time history of driving, friction, and total applied torques
at each joint after the simulation completes. Dynamically creates one sub-plot
per DOF, supporting double (2), triple (3), and golfer (7) models.

Theme integration
-----------------
Background, axis, and text colours are sourced from the shared PlotThemeManager
when available, falling back to hardcoded dark defaults.  The distinctive *trace*
colours are preserved regardless of theme.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

if TYPE_CHECKING:
    pass  # SimulationResult variants handled via duck typing

logger = logging.getLogger(__name__)

_pg: Any | None = None
_pyqtgraph_checked = False


def _load_pyqtgraph() -> Any | None:
    """Return pyqtgraph when available, without importing it during collection."""
    global _pg, _pyqtgraph_checked
    if _pyqtgraph_checked:
        return _pg
    _pyqtgraph_checked = True
    try:
        import pyqtgraph as pg
    except (ImportError, OSError, RuntimeError) as exc:
        logger.debug("pyqtgraph unavailable, using torque plot fallback: %s", exc)
        _pg = None
    else:
        _pg = pg
    return _pg


# ── Try to import shared PlotThemeManager ──────────────────────────────────
_PLOT_THEME_AVAILABLE = False
_get_plot_theme_manager: Any = None
try:
    import sys
    from pathlib import Path

    from plot_theme.manager import (
        get_plot_theme_manager as _shared_get_plot_theme_manager,
    )

    _get_plot_theme_manager = _shared_get_plot_theme_manager
    _PLOT_THEME_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Trace colour palette and joint labels — imported from Qt-free module
# ---------------------------------------------------------------------------
from .torque_history_constants import (  # noqa: E402
    _DRIVE_COLORS,
    _FRICTION_COLORS,
    _TOTAL_COLORS,
    _joint_labels_for_ndof,
)

# Default dark background/text when no theme system is available
_DEFAULT_BG = "#1a1a28"
_DEFAULT_TEXT = "#c0c0d8"
_DEFAULT_GRID = "#303050"


class TorqueHistoryWidget(QWidget):
    """Post-simulation torque breakdown chart — N-DOF aware.

    Dynamically creates one sub-plot per joint. Supports 2-DOF (double),
    3-DOF (triple), and 7-DOF (golfer) models via duck typing on the
    simulation result's torques_at / friction_torques_at / total_torques_at.

    Contract:
        - set_simulation() must be called before set_frame() or clear().
        - clear() resets to an empty state without crashing.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: Any = None
        self._n_joints = 0
        self._bg_color = _DEFAULT_BG
        self._text_color = _DEFAULT_TEXT
        self._grid_color = _DEFAULT_GRID
        self._plots: list[Any] = []  # pg.PlotWidget per joint
        self._curves: list[dict[str, Any]] = []  # per-joint curve dicts
        self._cursors: list[Any] = []  # pg.InfiniteLine per joint
        self._load_theme_colors()
        self._build_ui()

    # ------------------------------------------------------------------
    # Theme integration
    # ------------------------------------------------------------------

    def _load_theme_colors(self) -> None:
        """Load background/text/grid from PlotThemeManager if available."""
        if not _PLOT_THEME_AVAILABLE or _get_plot_theme_manager is None:
            return
        try:
            manager = _get_plot_theme_manager(settings_app="PendulumSimulator")
            theme = manager.current_theme
            self._bg_color = theme.axes_facecolor
            self._text_color = theme.text_color
            self._grid_color = theme.grid_color
            manager.add_theme_change_callback(self._on_plot_theme_changed)
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("PlotThemeManager unavailable, using defaults: %s", exc)

    def _on_plot_theme_changed(self, theme: object) -> None:
        """Update backgrounds when the plot theme changes."""
        if not (theme is not None):
            raise ValueError("theme must be provided")
        if _load_pyqtgraph() is None:
            return
        try:
            self._bg_color = theme.axes_facecolor  # type: ignore[attr-defined]
            self._text_color = theme.text_color  # type: ignore[attr-defined]
            self._grid_color = theme.grid_color  # type: ignore[attr-defined]
            for pw in self._plots:
                pw.setBackground(self._bg_color)
        except (AttributeError, RuntimeError) as exc:
            logger.debug("Could not update torque plot theme: %s", exc)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._outer_layout = QVBoxLayout(self)
        self._outer_layout.setContentsMargins(4, 4, 4, 4)
        self._outer_layout.setSpacing(4)

        title = QLabel("Torque History")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"color: {self._text_color}; font-size: 14px; font-weight: bold;"
            "padding: 4px; border-bottom: 1px solid #505070;"
        )
        self._outer_layout.addWidget(title)

        if _load_pyqtgraph() is None:
            fallback = QLabel(
                "Install pyqtgraph for torque plots:\n  pip install pyqtgraph"
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet("color: #808090; font-size: 11px;")
            self._outer_layout.addWidget(fallback)
            return

        # Scrollable container for N sub-plots
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._plot_container = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_container)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_layout.setSpacing(2)
        self._scroll.setWidget(self._plot_container)
        self._outer_layout.addWidget(self._scroll)

    def _create_plots_for_ndof(self, n_joints: int) -> None:
        """Dynamically create sub-plots for the given joint count.

        Clears any existing plots first.
        """
        if not (n_joints is not None):
            raise ValueError("n_joints must be provided")
        pg = _load_pyqtgraph()
        if pg is None:
            return

        # Clear old plots
        for pw in self._plots:
            self._plot_layout.removeWidget(pw)
            pw.deleteLater()
        self._plots.clear()
        self._curves.clear()
        self._cursors.clear()

        labels = _joint_labels_for_ndof(n_joints)
        pen_kwargs = {"width": 2}
        _axis_style = {"color": self._text_color, "font-size": "11px"}

        for j in range(n_joints):
            pw = pg.PlotWidget(title=f"{labels[j]}")
            pw.setBackground(self._bg_color)
            pw.getPlotItem().setLabel("bottom", "Time (s)", **_axis_style)
            pw.getPlotItem().setLabel("left", "Torque (N·m)", **_axis_style)
            pw.getPlotItem().addLegend(offset=(10, 10))
            pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            pw.setMinimumHeight(120)

            dc = _DRIVE_COLORS[j % len(_DRIVE_COLORS)]
            fc = _FRICTION_COLORS[j % len(_FRICTION_COLORS)]
            tc = _TOTAL_COLORS[j % len(_TOTAL_COLORS)]

            curves = {
                "drive": pw.plot(name="Drive", pen=pg.mkPen(*dc, **pen_kwargs)),
                "friction": pw.plot(
                    name="Friction",
                    pen=pg.mkPen(*fc, style=Qt.PenStyle.DashLine, **pen_kwargs),
                ),
                "total": pw.plot(
                    name="Total",
                    pen=pg.mkPen(*tc, style=Qt.PenStyle.DotLine, **pen_kwargs),
                ),
            }

            cursor = pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pg.mkPen("w", width=1, style=Qt.PenStyle.DashLine),
            )
            pw.addItem(cursor)

            self._plot_layout.addWidget(pw)
            self._plots.append(pw)
            self._curves.append(curves)
            self._cursors.append(cursor)

        self._n_joints = n_joints

    # ------------------------------------------------------------------
    # Public interface (mirrors matrix_widget.py contract)
    # ------------------------------------------------------------------

    def set_simulation(self, result: Any) -> None:
        """Load simulation result and compute full torque history arrays.

        Dynamically detects the number of joints from the torque data and
        creates the appropriate number of sub-plots.

        Preconditions:
            - result has n_steps >= 2.
            - result has torques_at(), friction_torques_at(), total_torques_at().
        """
        if not (result.n_steps >= 2):
            raise ValueError("Result must have at least 2 time steps")
        self._result = result

        pg = _load_pyqtgraph()
        if pg is None:
            return

        t = result.t
        n = result.n_steps

        # Detect DOF count from first sample
        sample = result.torques_at(0)
        if isinstance(sample, tuple | list):
            n_joints = len(sample)
        elif isinstance(sample, np.ndarray):
            n_joints = sample.shape[0]
        else:
            n_joints = 2  # fallback

        # Recreate plots if joint count changed
        if n_joints != self._n_joints:
            self._create_plots_for_ndof(n_joints)

        # Pre-allocate arrays for all torque components
        drive = np.empty((n, n_joints))
        friction = np.empty((n, n_joints))
        total = np.empty((n, n_joints))

        for i in range(n):
            d = result.torques_at(i)
            f = result.friction_torques_at(i)
            tot = result.total_torques_at(i)
            drive[i] = np.asarray(d).ravel()[:n_joints]
            friction[i] = np.asarray(f).ravel()[:n_joints]
            total[i] = np.asarray(tot).ravel()[:n_joints]

        # Populate each joint's curves
        for j in range(n_joints):
            self._curves[j]["drive"].setData(t, drive[:, j])
            self._curves[j]["friction"].setData(t, friction[:, j])
            self._curves[j]["total"].setData(t, total[:, j])
            self._cursors[j].setValue(t[0])

    def set_frame(self, idx: int) -> None:
        """Move the time cursor to the frame at index idx.

        Called every animation tick — must be O(1).

        Preconditions:
            - set_simulation() has been called.
            - 0 <= idx < result.n_steps.
        """
        if self._result is None or _load_pyqtgraph() is None:
            return
        if not (0 <= idx < self._result.n_steps):
            raise ValueError("DbC Blocked: Precondition failed.")
        t_now = self._result.t[idx]
        for cursor in self._cursors:
            cursor.setValue(t_now)

    def clear(self) -> None:
        """Reset to an empty state (called on simulation reset)."""
        self._result = None
        if _load_pyqtgraph() is None:
            return
        for j_curves in self._curves:
            for curve in j_curves.values():
                curve.setData([], [])
        for cursor in self._cursors:
            cursor.setValue(0.0)

# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""ExerciseTab -- individual tab logic for movement visualizations."""

from __future__ import annotations

import logging
from typing import Any

from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (  # type: ignore[attr-defined]  # matplotlib stubs omit NavigationToolbar2QT from backend_qtagg stub
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..constants import (
    PLOT_GRID_COLS,
    PLOT_GRID_HEIGHT_RATIOS,
    PLOT_GRID_HSPACE,
    PLOT_GRID_MARGINS,
    PLOT_GRID_ROWS,
    PLOT_GRID_WSPACE,
)
from ..models import BodyModel
from ..rendering import Palette, style_axis
from ..trajectory import OptimizationResult
from . import anim_renderer, plot_renderer

logger = logging.getLogger(__name__)


class ExerciseTab(QWidget):
    """Single exercise tab with animation and analysis plots."""

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.name = name
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(11, 9), facecolor=Palette.BG)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self._create_axes()

    def _build_grid_axes(self, gs: GridSpec) -> dict:
        """Allocate all subplot axes from a pre-built GridSpec.

        Args:
            gs: 3-row x 4-col GridSpec already attached to ``self.fig``.

        Returns:
            Dict mapping axis names to Matplotlib Axes objects.
        """
        axes = {
            "anim": self.fig.add_subplot(gs[0, 0:3]),
            "com_path": self.fig.add_subplot(gs[0, 3]),
            "angles": self.fig.add_subplot(gs[1, 0]),
            "torques": self.fig.add_subplot(gs[1, 1]),
            "power": self.fig.add_subplot(gs[1, 2]),
            "com_time": self.fig.add_subplot(gs[1, 3]),
            "spine_comp": self.fig.add_subplot(gs[2, 0:2]),
            "spine_shear": self.fig.add_subplot(gs[2, 2:4]),
        }
        for ax in axes.values():
            style_axis(ax)
        return axes

    def _configure_anim_axis(self, ax: Axes) -> None:
        """Set aspect, limits, and placeholder text on the animation axis.

        Args:
            ax: The animation Matplotlib Axes to configure.
        """
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.15, 1.8)
        ax.text(
            0.5,
            0.5,
            "Click Optimize to begin",
            ha="center",
            va="center",
            fontsize=13,
            color=Palette.FG_DIM,
            style="italic",
            transform=ax.transAxes,
        )

    def _create_axes(self) -> None:
        gs = GridSpec(
            PLOT_GRID_ROWS,
            PLOT_GRID_COLS,
            figure=self.fig,
            height_ratios=list(PLOT_GRID_HEIGHT_RATIOS),
            hspace=PLOT_GRID_HSPACE,
            wspace=PLOT_GRID_WSPACE,
            **PLOT_GRID_MARGINS,
        )
        self.axes = self._build_grid_axes(gs)
        self._configure_anim_axis(self.axes["anim"])
        self.fig.suptitle(
            f"{self.name} Optimization",
            color=Palette.FG,
            fontsize=13,
            fontweight="bold",
        )
        self.canvas.draw()

    def _render_analysis_plots(
        self,
        result: OptimizationResult,
        body: BodyModel,
        bar_mass: float,
        labels: tuple[str, ...],
    ) -> None:
        """Delegate all per-panel plot calls to plot_renderer."""
        plot_renderer.plot_angles(self.axes["angles"], result, labels)
        plot_renderer.plot_torques(self.axes["torques"], result, labels)
        plot_renderer.plot_power(self.axes["power"], result, labels)
        plot_renderer.plot_com_path(self.axes["com_path"], result, body)
        plot_renderer.plot_com_balance(self.axes["com_time"], result, body)
        plot_renderer.plot_spine_loads(
            self.axes["spine_comp"],
            self.axes["spine_shear"],
            result,
            body,
            bar_mass,
            self.name,
        )

    def draw_all_plots(
        self,
        result: OptimizationResult,
        body: BodyModel,
        bar_mass: float,
        exercise_type: str = "squat",
    ) -> None:
        for k in self.axes:
            if k != "anim":
                self.axes[k].clear()
                style_axis(self.axes[k])
        labels = Palette.BENCH_LABELS if exercise_type == "bench_press" else Palette.SEG_LABELS
        self._render_analysis_plots(result, body, bar_mass, labels)
        self.fig.suptitle(
            f"{self.name}  |  {body.body_mass:.0f} kg body, {bar_mass:.0f} kg barbell",
            color=Palette.FG,
            fontsize=12,
            fontweight="bold",
        )
        self.canvas.draw()

    def draw_anim_frame(
        self,
        fi: int,
        result: OptimizationResult,
        dynamics: Any,
        body: BodyModel,
        exercise_type: str,
    ) -> None:
        anim_renderer.draw_anim_frame(
            self.axes["anim"], fi, result, dynamics, body, self.name, exercise_type
        )
        self.canvas.draw()


# ==============================================================
# Playback Controls
# ==============================================================


# ==============================================================
# Comparison Dialog
# ==============================================================

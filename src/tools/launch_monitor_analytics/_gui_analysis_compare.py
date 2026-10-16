"""Comparison, dispersion and trend analysis handlers."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.dates import date2num

from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.async_action import WorkerContext
from src.tools.launch_monitor_analytics._gui_params import (
    _ELLIPSE_POINTS,
    _ComparisonParams,
    _DispersionData,
    _DispersionParams,
    _DispersionPlotData,
    _TrendParams,
)
from src.tools.launch_monitor_model import (
    DispersionResult,
    MonitorComparisonResult,
    TemporalTrendResult,
    analyze_dispersion,
    analyze_trend,
    compare_monitors,
)

logger = get_logger(__name__)


class _CompareAnalysisMixin:
    """Monitor-comparison, dispersion and trend handlers for ``MainWidget`` (#9470)."""

    if TYPE_CHECKING:
        # Host attributes/methods live on ``MainWidget`` (composed mixins).
        def __getattr__(self, name: str) -> Any: ...

    def _read_comparison_params(self) -> _ComparisonParams:
        """Read the comparison controls on the GUI thread (#9470)."""
        match = self.comparison_match.currentText()
        return _ComparisonParams(
            metric=self.comparison_metric.currentText(),
            match_column=None if match == "(unmatched)" else match,
            reference_monitor=self.comparison_reference.currentText() or None,
        )

    def _compute_comparison(
        self,
        params: _ComparisonParams,
        ctx: WorkerContext | None = None,
    ) -> MonitorComparisonResult:
        """Compare monitors as plain data (#9470).

        One opaque call; Cancel takes effect when the comparison returns.
        """
        if ctx is not None:
            ctx.report(
                None,
                "comparing monitors; cancel takes effect when the step ends",
            )
        return compare_monitors(
            self.analysis_frame,
            metric=params.metric,
            match_column=params.match_column,
            reference_monitor=params.reference_monitor,
        )

    def _present_comparison(self, result: MonitorComparisonResult) -> None:
        """Render the comparison table and error bars. GUI-thread only."""
        self.comparison_table.set_frame(
            pd.DataFrame(asdict(item) for item in result.pairwise)
        )
        axes = self.comparison_plot.reset_axes()
        labels = [item.monitor for item in result.summaries]
        means = [item.mean for item in result.summaries]
        errors = [item.standard_deviation for item in result.summaries]
        axes.errorbar(labels, means, yerr=errors, fmt="o", capsize=5)
        axes.set_title(
            self._title_with_project(f"{result.metric}: Mean and Standard Deviation")
        )
        axes.grid(True, axis="y", alpha=0.25)
        self.comparison_plot.draw_idle()
        warning = next((item.warning for item in result.pairwise if item.warning), None)
        self.status_label.setText(
            warning or "Matched monitor agreement analysis complete."
        )

    def _run_comparison_ui(self) -> None:
        """Synchronous monitor comparison (kept pre-migration path)."""
        try:
            result = self._compute_comparison(self._read_comparison_params())
        except ValueError as exc:
            self._show_error("Monitor Comparison Failed", exc)
            return
        self._present_comparison(result)

    def _run_comparison_async(self) -> None:
        """Run the monitor comparison off the GUI thread (#9470)."""
        params = self._read_comparison_params()
        self._start_action(
            "Monitor comparison",
            lambda ctx: self._compute_comparison(params, ctx),
            self._present_comparison,
            "Monitor Comparison Failed",
        )

    # ---- dispersion ---------------------------------------------------------

    def _read_dispersion_params(self) -> _DispersionParams:
        """Read the dispersion controls on the GUI thread (#9470)."""
        return _DispersionParams(
            forward=self.dispersion_forward.currentText(),
            lateral=self.dispersion_lateral.currentText(),
            group_column=self.dispersion_group.currentText(),
        )

    def _compute_dispersion(
        self,
        frame: pd.DataFrame,
        params: _DispersionParams,
        ctx: WorkerContext | None = None,
    ) -> _DispersionData:
        """Compute per-group dispersion as plain data (#9470).

        Loops once per group; every group boundary is a cancellation
        checkpoint and a progress report. Plot geometry is returned as
        plain arrays so the worker never touches widgets.
        """
        groups: list[tuple[object, pd.DataFrame]] = [("All Shots", frame)]
        if params.group_column != "(all shots)" and params.group_column in frame:
            groups = [
                (name, group)
                for name, group in frame.groupby(params.group_column, dropna=False)
            ]
        total = len(groups)
        rows: list[dict[str, object]] = []
        series: list[_DispersionPlotData] = []
        for index, (name, group) in enumerate(groups):
            if ctx is not None:
                ctx.raise_if_cancelled()
                ctx.report(index / total, f"dispersion group {index + 1}/{total}")
            result = analyze_dispersion(
                group, forward=params.forward, lateral=params.lateral
            )
            rows.append({"group": str(name), **asdict(result)})
            series.append(self._dispersion_plot_data(str(name), group, params, result))
        return _DispersionData(rows=rows, series=tuple(series))

    @staticmethod
    def _dispersion_plot_data(
        name: str,
        group: pd.DataFrame,
        params: _DispersionParams,
        result: DispersionResult,
    ) -> _DispersionPlotData:
        """Extract plain per-group plot geometry (worker-safe, no widgets)."""
        angle = np.linspace(0.0, 2.0 * np.pi, _ELLIPSE_POINTS)
        ellipse = np.column_stack(
            [
                result.ellipse_major / 2 * np.cos(angle),
                result.ellipse_minor / 2 * np.sin(angle),
            ]
        )
        rotation = np.array(
            [
                [np.cos(result.ellipse_angle_rad), -np.sin(result.ellipse_angle_rad)],
                [np.sin(result.ellipse_angle_rad), np.cos(result.ellipse_angle_rad)],
            ]
        )
        rotated = ellipse @ rotation.T
        return _DispersionPlotData(
            name=name,
            forward_values=group[params.forward].to_numpy(dtype=float),
            lateral_values=group[params.lateral].to_numpy(dtype=float),
            ellipse_forward=rotated[:, 0] + result.mean_forward,
            ellipse_lateral=rotated[:, 1] + result.mean_lateral,
        )

    def _present_dispersion(
        self,
        data: _DispersionData,
        params: _DispersionParams,
    ) -> None:
        """Render dispersion rows, ellipses, and labels. GUI-thread only."""
        self.dispersion_table.set_frame(pd.DataFrame(data.rows))
        axes = self.dispersion_plot.reset_axes()
        for item in data.series:
            axes.scatter(
                item.forward_values,
                item.lateral_values,
                alpha=0.55,
                label=item.name,
            )
            axes.plot(item.ellipse_forward, item.ellipse_lateral)
        axes.set_xlabel(params.forward)
        axes.set_ylabel(params.lateral)
        axes.set_title(self._title_with_project("95% Dispersion Ellipses"))
        axes.grid(True, alpha=0.25)
        if len(data.series) > 1:
            axes.legend(fontsize="small")
        self.dispersion_plot.draw_idle()
        self.status_label.setText(
            f"Dispersion complete for {len(data.series)} group(s)."
        )

    def _run_dispersion_ui(self) -> None:
        """Synchronous dispersion analysis (kept pre-migration path)."""
        params = self._read_dispersion_params()
        try:
            data = self._compute_dispersion(self.analysis_frame, params)
        except ValueError as exc:
            self._show_error("Dispersion Analysis Failed", exc)
            return
        self._present_dispersion(data, params)

    def _run_dispersion_async(self) -> None:
        """Run the dispersion analysis off the GUI thread (#9470)."""
        params = self._read_dispersion_params()
        frame = self.analysis_frame
        self._start_action(
            "Dispersion analysis",
            lambda ctx: self._compute_dispersion(frame, params, ctx),
            lambda data: self._present_dispersion(data, params),
            "Dispersion Analysis Failed",
        )

    # ---- longitudinal trend ---------------------------------------------------

    def _read_trend_params(self) -> _TrendParams:
        """Read the trend controls on the GUI thread (#9470)."""
        return _TrendParams(
            metric=self.trend_metric.currentText(),
            time_column=self.trend_time.currentText(),
            rolling_window=int(self.trend_window.value()),
        )

    def _compute_trend(
        self,
        params: _TrendParams,
        ctx: WorkerContext | None = None,
    ) -> TemporalTrendResult:
        """Analyze the longitudinal trend as plain data (#9470).

        One opaque call; Cancel takes effect when the analysis returns.
        """
        if ctx is not None:
            ctx.report(
                None,
                "analysing trend; cancel takes effect when the step ends",
            )
        return analyze_trend(
            self.analysis_frame,
            metric=params.metric,
            time_column=params.time_column,
            rolling_window=params.rolling_window,
        )

    def _present_trend(self, result: TemporalTrendResult) -> None:
        """Render the trend table, rolling series, and change points."""
        candidates = pd.DataFrame(asdict(item) for item in result.change_candidates)
        self.trend_table.set_frame(candidates)
        axes = self.trend_plot.reset_axes()
        time_column = self.trend_time.currentText()
        axes.scatter(
            result.rolling[time_column],
            result.rolling["value"],
            s=16,
            alpha=0.4,
            label="Shots",
        )
        axes.plot(
            result.rolling[time_column],
            result.rolling["rolling_mean"],
            label="Rolling Mean",
        )
        axes.plot(result.rolling[time_column], result.rolling["ewma"], label="EWMA")
        for candidate in result.change_candidates:
            axes.axvline(
                date2num(candidate.captured_at.to_pydatetime()),
                linestyle="--",
                alpha=0.35,
            )
        axes.set_title(
            self._title_with_project(f"{result.metric}: Longitudinal Change")
        )
        axes.legend(fontsize="small")
        axes.grid(True, alpha=0.25)
        self.trend_plot.draw_idle()
        self.status_label.setText(
            f"Trend slope={result.slope_per_day:.4g}/day; "
            f"{len(result.change_candidates)} candidate change point(s)."
        )

    def _run_trend_ui(self) -> None:
        """Synchronous trend analysis (kept pre-migration path)."""
        try:
            result = self._compute_trend(self._read_trend_params())
        except ValueError as exc:
            self._show_error("Trend Analysis Failed", exc)
            return
        self._present_trend(result)

    def _run_trend_async(self) -> None:
        """Run the trend analysis off the GUI thread (#9470)."""
        params = self._read_trend_params()
        self._start_action(
            "Trend analysis",
            lambda ctx: self._compute_trend(params, ctx),
            self._present_trend,
            "Trend Analysis Failed",
        )

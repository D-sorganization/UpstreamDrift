"""Characterization tests pinning the public surface of the split ``gui`` module (#12840)."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


@pytest.mark.unit
def test_gui_reexports_public_and_param_names() -> None:
    from src.tools.launch_monitor_analytics import gui

    for name in (
        "MainWidget",
        "LaunchMonitorAnalyticsWindow",
        "PlotCanvas",
        "main",
        "_DispersionData",
        "_DispersionParams",
        "_DispersionPlotData",
        "_ModelParams",
        "_RelationshipParams",
        "_TrendParams",
        "_ComparisonParams",
        "_ELLIPSE_POINTS",
    ):
        assert hasattr(gui, name), name
    assert gui._ELLIPSE_POINTS == 200


@pytest.mark.unit
def test_main_widget_keeps_every_handler_after_split() -> None:
    from src.tools.launch_monitor_analytics.gui import MainWidget

    for name in (
        "_build_ui",
        "_wire_actions",
        "run_relationship_analysis",
        "_run_trend_async",
        "_run_dispersion_async",
        "_run_comparison_async",
        "_run_model_async",
        "_run_treatment_async",
        "_run_multivariate_async",
        "export_data",
        "export_manifest",
        "_filter_rules",
        "_refresh_all",
        "save_project",
        "load_project",
    ):
        assert callable(getattr(MainWidget, name)), name

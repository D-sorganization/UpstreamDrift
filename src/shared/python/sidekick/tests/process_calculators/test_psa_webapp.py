# ruff: noqa: E501
"""Tests for psa_webapp.py."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

sys.modules["streamlit"] = MagicMock()
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.express"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()
sys.modules["pandas"] = MagicMock()

import numpy as np
import pytest
from upstream_drift_tools.process_calculators.psa_package.psa_webapp import (
    DEFAULT_COMPONENTS,
    PSAResults,
    StreamCompositions,
    StreamFlows,
    _render_data_tables_tab,
    _render_key_metrics,
    _render_o2_safety_tab,
    _render_results_tab,
    _render_sensitivity_tab,
    _render_sidebar,
    _resolve_plot_mode,
    get_flammability_status,
    main,
)


@pytest.fixture
def mock_streamlit():
    """Mock the streamlit module."""
    with patch(
        "upstream_drift_tools.process_calculators.psa_package.psa_webapp.st"
    ) as mock_st:
        yield mock_st


@pytest.fixture
def sample_results():
    """Create a sample PSAResults object."""
    flows = StreamFlows(
        fresh_feed=np.array([10.0, 5.0]),
        s2_tail_recycle=np.array([1.0, 0.5]),
        product_recycle=np.array([2.0, 1.0]),
        mixed_feed=np.array([13.0, 6.5]),
        exhaust=np.array([2.0, 1.0]),
        s2_tail_vent=np.array([0.5, 0.25]),
        interstage=np.array([11.0, 5.5]),
        gross_product=np.array([10.0, 5.0]),
        s2_tail=np.array([1.0, 0.5]),
        net_product=np.array([8.0, 4.0]),
    )
    comps = StreamCompositions(
        fresh_feed=np.array([66.6, 33.3]),
        s2_tail_recycle=np.array([66.6, 33.3]),
        product_recycle=np.array([66.6, 33.3]),
        mixed_feed=np.array([66.6, 33.3]),
        exhaust=np.array([66.6, 33.3]),
        s2_tail_vent=np.array([66.6, 33.3]),
        interstage=np.array([66.6, 33.3]),
        gross_product=np.array([66.6, 33.3]),
        s2_tail=np.array([66.6, 33.3]),
        net_product=np.array([66.6, 33.3]),
    )
    return PSAResults(
        component_names=["H2", "O2"],
        flows=flows,
        compositions=comps,
        h2_recovery_pct=80.0,
        h2_purity_pct=99.9,
        total_feed_scfm=15.0,
        total_net_product_scfm=12.0,
        total_exhaust_scfm=3.0,
        total_s2_tail_vent_scfm=0.75,
        mass_balance_error=0.001,
        s2_tail_h2_pct=5.0,
        s2_tail_o2_pct=1.0,
    )


def test_resolve_plot_mode():
    assert _resolve_plot_mode(True, True) == "lines+markers"
    assert _resolve_plot_mode(False, True) == "markers"
    assert _resolve_plot_mode(True, False) == "lines"
    assert _resolve_plot_mode(False, False) == "lines"


def test_get_flammability_status():
    status, color = get_flammability_status(5.0, 0.05)
    assert status == "Safe-Low O2"
    assert color == "green"

    status, color = get_flammability_status(5.0, 3.0)
    assert status == "CRITICAL"
    assert color == "red"

    status, color = get_flammability_status(3.0, 3.0)
    assert status == "Safe-Below LFL"
    assert color == "green"

    status, color = get_flammability_status(80.0, 1.0)
    assert status == "Caution-Rich"
    assert color == "orange"

    status, color = get_flammability_status(50.0, 1.5)
    assert status == "FLAMMABLE"
    assert color == "red"


@patch("upstream_drift_tools.process_calculators.psa_package.psa_webapp.px.imshow")
@patch("upstream_drift_tools.process_calculators.psa_package.psa_webapp.go.Figure")
@patch(
    "upstream_drift_tools.process_calculators.psa_package.psa_webapp._render_data_tables_tab"
)
@patch(
    "upstream_drift_tools.process_calculators.psa_package.psa_webapp._render_o2_safety_tab"
)
@patch(
    "upstream_drift_tools.process_calculators.psa_package.psa_webapp._render_sensitivity_tab"
)
@patch(
    "upstream_drift_tools.process_calculators.psa_package.psa_webapp._render_results_tab"
)
@patch(
    "upstream_drift_tools.process_calculators.psa_package.psa_webapp._render_key_metrics"
)
@patch(
    "upstream_drift_tools.process_calculators.psa_package.psa_webapp._render_sidebar"
)
def test_webapp_main(
    mock_sidebar,
    mock_key,
    mock_res,
    mock_sens,
    mock_o2,
    mock_data,
    mock_fig,
    mock_imshow,
    mock_streamlit,
):
    mock_sidebar.return_value = (
        1100.0,
        100,
        0,
        [
            {
                "name": "H2",
                "feed_pct": 32.08,
                "stage1_removal_pct": 18.0,
                "stage2_removal_pct": 15.0,
            },
            {
                "name": "O2",
                "feed_pct": 0.50,
                "stage1_removal_pct": 81.0,
                "stage2_removal_pct": 99.99,
            },
        ],
    )

    # Mock tabs
    tab1, tab2, tab3, tab4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    mock_streamlit.tabs.return_value = (tab1, tab2, tab3, tab4)

    main()

    mock_sidebar.assert_called_once()
    mock_key.assert_called_once()
    mock_streamlit.title.assert_called()
    mock_streamlit.tabs.assert_called()


def test_render_sidebar(mock_streamlit):
    mock_streamlit.sidebar.slider.side_effect = [1100, 100, 0, 18, 81]
    mock_streamlit.sidebar.number_input.side_effect = [32.08, 0.50]
    total_feed, s2_recycle, prod_recycle, components = _render_sidebar(
        DEFAULT_COMPONENTS
    )
    assert total_feed == 1100.0
    assert s2_recycle == 100
    assert prod_recycle == 0
    assert len(components) == 7


def test_render_key_metrics(mock_streamlit, sample_results):
    mock_streamlit.columns.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    sample_results.s2_tail_h2_pct = 2.0
    sample_results.s2_tail_o2_pct = 1.0
    _render_key_metrics(sample_results)
    assert mock_streamlit.columns.called
    assert mock_streamlit.metric.called
    assert mock_streamlit.success.called


def test_render_key_metrics_flammable(mock_streamlit, sample_results):
    mock_streamlit.columns.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    sample_results.s2_tail_h2_pct = 50.0
    sample_results.s2_tail_o2_pct = 5.0
    _render_key_metrics(sample_results)
    assert mock_streamlit.error.called


def test_render_key_metrics_caution(mock_streamlit, sample_results):
    mock_streamlit.columns.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    sample_results.s2_tail_h2_pct = 80.0
    sample_results.s2_tail_o2_pct = 1.0
    _render_key_metrics(sample_results)
    assert mock_streamlit.warning.called


@patch("upstream_drift_tools.process_calculators.psa_package.psa_webapp.go.Figure")
def test_render_results_tab(mock_figure, mock_streamlit, sample_results):
    mock_streamlit.columns.return_value = (MagicMock(), MagicMock())
    _render_results_tab(sample_results)
    assert mock_streamlit.subheader.called
    assert mock_streamlit.columns.called
    assert mock_streamlit.dataframe.called
    assert mock_streamlit.plotly_chart.called


@patch("upstream_drift_tools.process_calculators.psa_package.psa_webapp.go.Figure")
def test_render_sensitivity_tab(mock_figure, mock_streamlit):
    mock_streamlit.columns.side_effect = [
        (MagicMock(), MagicMock(), MagicMock()),
        (MagicMock(), MagicMock()),
    ]
    mock_streamlit.checkbox.side_effect = [True, False]
    mock_streamlit.slider.return_value = 2
    _render_sensitivity_tab(1100.0, 100, 0, DEFAULT_COMPONENTS)
    assert mock_streamlit.plotly_chart.called


@patch("upstream_drift_tools.process_calculators.psa_package.psa_webapp.px.imshow")
@patch("upstream_drift_tools.process_calculators.psa_package.psa_webapp.go.Figure")
def test_render_o2_safety_tab(mock_figure, mock_imshow, mock_streamlit):
    mock_streamlit.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_streamlit.slider.return_value = 2
    _render_o2_safety_tab(1100.0, DEFAULT_COMPONENTS)
    assert mock_streamlit.plotly_chart.called


def test_render_data_tables_tab(mock_streamlit, sample_results):
    _render_data_tables_tab(sample_results)
    assert mock_streamlit.dataframe.called
    assert mock_streamlit.markdown.called

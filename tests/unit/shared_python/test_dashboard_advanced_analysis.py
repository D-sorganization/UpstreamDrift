"""Tests for Advanced Analysis Dialog."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from src.shared.python.dashboard.advanced_analysis import (
    AdvancedAnalysisDialog,
    CoherenceTab,
    PhasePlaneTab,
    SpectrogramTab,
)


class MockRecorder:
    """Mock recorder providing synthetic time series data."""

    def get_time_series(self, key: str) -> tuple:
        """Return synthetic time series for the given key."""
        t = np.linspace(0, 1, 100)
        data = np.sin(2 * np.pi * 10 * t)
        if key == "joint_positions":
            data = data.reshape(-1, 1)
        elif key == "joint_velocities":
            data = np.cos(2 * np.pi * 10 * t).reshape(-1, 1)
        elif key == "joint_torques":
            data = (data * 0.1).reshape(-1, 1)
        return t, data


@pytest.fixture
def app(qapp) -> QApplication:
    """Provide the QApplication instance."""
    return qapp


@pytest.fixture
def recorder() -> MockRecorder:
    """Create a MockRecorder instance."""
    return MockRecorder()


def test_spectrogram_tab(recorder, qtbot) -> None:
    """Test SpectrogramTab initialization and metric switching."""
    tab = SpectrogramTab(recorder)
    qtbot.addWidget(tab)

    # Check initial plot
    assert len(tab.ax.collections) > 0  # pcolormesh creates a collection

    # Change metric
    tab.combo_metric.setCurrentText("Joint Torques")
    assert tab.current_key == "joint_torques"
    # Should re-plot
    assert len(tab.ax.collections) > 0


def test_phase_plane_tab(recorder, qtbot) -> None:
    """Test PhasePlaneTab initialization and dimension change."""
    tab = PhasePlaneTab(recorder)
    qtbot.addWidget(tab)

    # Check initial plot
    assert len(tab.ax.lines) >= 1  # The line + start/end markers

    # Change dim
    tab.spin_dim.setValue(0)
    # Should re-plot
    assert len(tab.ax.lines) >= 1


def test_coherence_tab(recorder, qtbot) -> None:
    """Test CoherenceTab initialization and signal switching."""
    tab = CoherenceTab(recorder)
    qtbot.addWidget(tab)

    # Check initial plot
    assert len(tab.ax.lines) > 0

    # Change signal 2
    tab.combo2.setCurrentText("Joint Velocities")
    assert len(tab.ax.lines) > 0


def test_advanced_analysis_dialog(recorder, qtbot) -> None:
    """Test AdvancedAnalysisDialog contains all expected tabs."""
    dlg = AdvancedAnalysisDialog(None, recorder)
    qtbot.addWidget(dlg)

    assert dlg.tabs.count() == 6
    assert isinstance(dlg.tab_spectrogram, SpectrogramTab)
    assert isinstance(dlg.tab_phase, PhasePlaneTab)
    assert isinstance(dlg.tab_coherence, CoherenceTab)

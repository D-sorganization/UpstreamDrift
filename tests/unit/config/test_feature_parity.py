"""Unit tests for feature parity configuration and registry integrity.

Tests the promotion of render.club_head_and_ball to parity following the
integration of the decorative ball at address in GolferModel.tsx (#11719,
closing #11717, part of #9545).
"""

from __future__ import annotations

import pytest

from src.config.feature_parity_loader import FeatureParityRegistry

pytestmark = [pytest.mark.unit, pytest.mark.parity]


def test_registry_loads_successfully() -> None:
    """The committed feature parity registry loads without schema errors."""
    registry = FeatureParityRegistry.load()
    assert registry.version == "1.0.0"
    assert len(registry.entries) > 0


def test_club_head_and_ball_render_parity() -> None:
    """render.club_head_and_ball must have status 'parity' and no open gap issue.

    With GolferModel.tsx rendering <BallModel position={ADDRESS_BALL} /> using
    ballPositionAtAddress(ADDRESS_FACE_CENTRE, [1, 0, 0], 0) alongside ClubHead,
    the web visual layer matches the Python mesh placement contract.
    """
    registry = FeatureParityRegistry.load()
    entry = registry.get("render.club_head_and_ball")
    assert entry is not None, "render.club_head_and_ball entry missing from registry"

    assert entry.status == "parity"
    assert entry.issue is None, "Parity entry must not retain an open gap issue"
    assert entry.pyqt == "src/shared/python/model_appearance/club_head_mesh.py"
    assert entry.web == "ui/src/components/visualization/ClubHead.tsx"
    assert "GolferModel.tsx" in entry.notes
    assert "ADDRESS_BALL" in entry.notes
    assert "ballPositionAtAddress" in entry.notes


def test_launch_monitor_analytics_parity() -> None:
    """tools.launch_monitor_analytics must have status 'parity' and no open gap issue."""
    registry = FeatureParityRegistry.load()
    entry = registry.get("tools.launch_monitor_analytics")
    assert entry is not None, (
        "tools.launch_monitor_analytics entry missing from registry"
    )

    assert entry.status == "parity"
    assert entry.issue is None, "Parity entry must not retain an open gap issue"
    assert entry.pyqt == "src/tools/launch_monitor_analytics/gui.py"
    assert entry.web == "ui/src/pages/LaunchMonitorAnalytics.tsx"
    assert "LaunchMonitorSessionsPanel.tsx" in entry.notes
    assert "LaunchMonitorReportsPanel.tsx" in entry.notes


def test_all_gap_entries_have_valid_issues() -> None:
    """Every entry with status 'gap' must cite an open issue number."""
    registry = FeatureParityRegistry.load()
    for entry in registry.by_status("gap"):
        assert isinstance(entry.issue, int) and entry.issue > 0, (
            f"Gap entry {entry.feature_id} must have a positive integer issue number."
        )

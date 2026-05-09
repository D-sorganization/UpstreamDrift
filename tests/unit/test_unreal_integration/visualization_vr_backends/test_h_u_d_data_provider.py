"""Unit tests for visualization, VR interaction, and viewer backends.

TDD tests for the remaining Unreal Engine integration components.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.unreal_integration.data_models import (
    ForceVector,
    Quaternion,
    SwingMetrics,
    TrajectoryPoint,
    Vector3,
)
from src.unreal_integration.mesh_loader import (
    LoadedMesh,
    MeshFace,
    MeshVertex,
)
from src.unreal_integration.viewer_backends import (
    BackendType,
    CameraState,
    MockBackend,
    ViewerConfig,
    create_viewer,
)
from src.unreal_integration.visualization import (
    ForceVectorRenderer,
    HUDDataProvider,
    RenderData,
    TrajectoryRenderer,
    VisualizationConfig,
    VisualizationType,
)
from src.unreal_integration.vr_interaction import (
    VRControllerHand,
    VRControllerState,
    VRHeadsetState,
    VRInteractionManager,
    VRLocomotionMode,
)

# ============================================================================
# Visualization Tests
# ============================================================================


class TestHUDDataProvider:
    """Tests for HUDDataProvider."""

    def test_create_provider_metric(self) -> None:
        """Test provider with metric units."""
        provider = HUDDataProvider(units="metric")
        assert provider.units == "metric"

    def test_create_provider_imperial(self) -> None:
        """Test provider with imperial units."""
        provider = HUDDataProvider(units="imperial")
        assert provider.units == "imperial"

    def test_get_hud_data_with_metrics(self) -> None:
        """Test HUD data with swing metrics."""
        provider = HUDDataProvider()
        metrics = SwingMetrics(
            club_head_speed=45.0,
            x_factor=52.0,
            smash_factor=1.48,
        )
        hud = provider.get_hud_data(metrics=metrics, timestamp=0.5, frame_number=30)

        assert hud["timestamp"] == 0.5
        assert hud["frame"] == 30
        assert "panels" in hud
        assert "club_head_speed" in hud["panels"]

    def test_format_value(self) -> None:
        """Test value formatting."""
        provider = HUDDataProvider()
        panel = {"value": 45.234, "unit": "m/s", "format": "{:.1f}"}
        formatted = provider.format_value(panel)
        assert formatted == "45.2 m/s"

    def test_get_compact_hud(self) -> None:
        """Test compact HUD output."""
        provider = HUDDataProvider()
        metrics = SwingMetrics(club_head_speed=45.0, x_factor=52.0)
        compact = provider.get_compact_hud(metrics)
        assert "Club Head Speed" in compact

    def test_unit_conversion_imperial(self) -> None:
        """Test unit conversion to imperial."""
        provider = HUDDataProvider(units="imperial")
        metrics = SwingMetrics(club_head_speed=44.7)  # ~100 mph
        hud = provider.get_hud_data(metrics=metrics)
        # Should be converted to mph
        speed_panel = hud["panels"]["club_head_speed"]
        assert speed_panel["unit"] == "mph"


# ============================================================================
# VR Interaction Tests
# ============================================================================


# ============================================================================
# Viewer Backend Tests
# ============================================================================


# Should not raise

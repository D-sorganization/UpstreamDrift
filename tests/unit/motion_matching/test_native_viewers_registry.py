"""Tests for native viewer registry and fail-closed dependency handlers (MV-05 #10481)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.shared.python.motion_matching.native_viewers import (
    NativeViewerAdapter,
    NativeViewerBackend,
    ViewerLaunchConfig,
    ViewerLaunchResult,
    ViewerUnavailableError,
    get_backend_adapter,
    get_supported_backends,
    open_in_native_viewer,
)
from src.shared.python.motion_matching.visualization.simulation_viewer import (
    SimulationData,
)
import numpy as np

pytestmark = pytest.mark.unit


@pytest.fixture
def dummy_sim_data() -> SimulationData:
    time_s = np.linspace(0, 1.0, 10)
    q = np.zeros((10, 6))
    return SimulationData(time_s=time_s, q=q)


def test_supported_backends_inventory():
    backends = get_supported_backends()
    assert "mujoco" in backends
    assert "meshcat" in backends
    assert "gepetto" in backends
    assert "opensim" in backends
    assert "matlab" in backends


def test_get_backend_adapter_returns_adapter():
    for name in ["mujoco", "meshcat", "gepetto", "opensim", "matlab"]:
        adapter = get_backend_adapter(name)
        assert isinstance(adapter, NativeViewerAdapter)
        assert adapter.name == name
        assert len(adapter.install_hint) > 0


def test_invalid_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported viewer backend"):
        get_backend_adapter("unreal_engine")


def test_open_in_native_viewer_fails_closed_when_unavailable(dummy_sim_data):
    with patch.object(
        get_backend_adapter("opensim"), "is_available", return_value=False
    ):
        with pytest.raises(ViewerUnavailableError) as excinfo:
            open_in_native_viewer(dummy_sim_data, "opensim")
        assert "conda install -c opensim-org opensim" in str(excinfo.value)


def test_open_in_native_viewer_dispatches_when_available(dummy_sim_data):
    adapter = get_backend_adapter("meshcat")
    with (
        patch.object(adapter, "is_available", return_value=True),
        patch.object(adapter, "launch") as mock_launch,
    ):
        mock_launch.return_value = ViewerLaunchResult(
            success=True,
            backend="meshcat",
            url="http://127.0.0.1:7000/static/",
        )
        with patch(
            "src.shared.python.motion_matching.native_viewers.get_backend_adapter",
            return_value=adapter,
        ):
            res = open_in_native_viewer(
                dummy_sim_data,
                "meshcat",
                config=ViewerLaunchConfig(view_mode="fitted", speed=0.5),
            )
            assert res.success
            assert res.backend == "meshcat"
            assert res.url == "http://127.0.0.1:7000/static/"
            mock_launch.assert_called_once()


def test_viewer_launch_config_validation():
    config = ViewerLaunchConfig(view_mode="static")
    assert config.view_mode == "static"

    with pytest.raises(ValueError, match="Invalid view_mode"):
        ViewerLaunchConfig(view_mode="virtual_reality")

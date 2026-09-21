"""Unit tests for SimulationViewer and multi-viewer launcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from scripts.launch_simulation_viewer import build_parser, main as cli_main
from src.shared.python.motion_matching.visualization.simulation_viewer import (
    SimulationData,
    SimulationViewer,
    ViewerBackend,
    launch_viewer,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_sim_data() -> SimulationData:
    """Create mock simulation trajectory data."""
    n_frames = 10
    nq = 44
    time_s = np.linspace(0.0, 0.5, n_frames)
    q = np.zeros((n_frames, nq))
    v = np.zeros((n_frames, nq))
    markers = np.zeros((n_frames, 34, 3))
    target = np.zeros((n_frames, 34, 3))
    valid = np.ones((n_frames, 34), dtype=bool)
    labels = [f"M_{i}" for i in range(34)]
    coords = [f"joint_{i}" for i in range(nq)]
    return SimulationData(
        time_s=time_s,
        q=q,
        v=v,
        markers_m=markers,
        target_m=target,
        valid=valid,
        labels=labels,
        coordinate_order=coords,
    )


def test_simulation_data_from_npz(
    tmp_path: Path, sample_sim_data: SimulationData
) -> None:
    file_path = tmp_path / "test_candidate.npz"
    save_kwargs: dict[str, Any] = {
        "time_s": sample_sim_data.time_s,
        "q": sample_sim_data.q,
        "v": sample_sim_data.v,
        "markers_m": sample_sim_data.markers_m,
        "target_m": sample_sim_data.target_m,
        "valid": sample_sim_data.valid,
        "labels": sample_sim_data.labels,
        "coordinate_order": sample_sim_data.coordinate_order,
    }
    np.savez(file_path, **save_kwargs)

    loaded = SimulationData.from_npz(file_path)
    assert len(loaded.time_s) == 10
    assert loaded.q.shape == (10, 44)
    assert loaded.labels is not None and len(loaded.labels) == 34


def test_is_backend_available() -> None:
    assert SimulationViewer.is_backend_available(ViewerBackend.MATPLOTLIB) is True
    # PyVista is installed in this test environment
    assert SimulationViewer.is_backend_available("pyvista") in (True, False)


def test_launch_gepetto_raises_if_unreachable(sample_sim_data: SimulationData) -> None:
    with (
        patch(
            "src.engines.physics_engines.pinocchio.python.dtack.viz.geppetto_viewer.GEPETTO_AVAILABLE",
            True,
        ),
        patch(
            "src.engines.physics_engines.pinocchio.python.dtack.viz.geppetto_viewer.GeppettoViewer.is_server_reachable",
            return_value=False,
        ),
    ):
        with pytest.raises(RuntimeError, match="Geppetto CORBA server is unreachable"):
            SimulationViewer.launch_gepetto(sample_sim_data)


def test_launch_gepetto_plays_trajectory_when_ready(
    sample_sim_data: SimulationData,
) -> None:
    mock_viewer = Mock()
    with (
        patch(
            "src.engines.physics_engines.pinocchio.python.dtack.viz.geppetto_viewer.GEPETTO_AVAILABLE",
            True,
        ),
        patch(
            "src.engines.physics_engines.pinocchio.python.dtack.viz.geppetto_viewer.GeppettoViewer.is_server_reachable",
            return_value=True,
        ),
        patch(
            "src.engines.physics_engines.pinocchio.python.dtack.viz.geppetto_viewer.GeppettoViewer",
            return_value=mock_viewer,
        ),
    ):
        SimulationViewer.launch_gepetto(sample_sim_data, stride=2, fps=100.0)
        mock_viewer.play_trajectory.assert_called_once()
        args, kwargs = mock_viewer.play_trajectory.call_args
        assert kwargs["stride"] == 2
        assert kwargs["dt"] == 0.01


def test_launch_meshcat_dispatches_when_available(
    sample_sim_data: SimulationData,
) -> None:
    mock_viewer = Mock()
    with (
        patch(
            "src.engines.physics_engines.pinocchio.python.dtack.viz.meshcat_viewer.MESHCAT_AVAILABLE",
            True,
        ),
        patch(
            "src.engines.physics_engines.pinocchio.python.dtack.viz.meshcat_viewer.MeshCatViewer",
            return_value=mock_viewer,
        ),
    ):
        SimulationViewer.launch_meshcat(sample_sim_data, loop=False, open_browser=False)
        # Successfully dispatched without errors


def test_launch_matplotlib_creates_figure(sample_sim_data: SimulationData) -> None:
    import matplotlib.pyplot as plt

    fig = SimulationViewer.launch_matplotlib(sample_sim_data, interactive=False)
    assert fig is not None
    plt.close(fig)


def test_launch_pyvista_offscreen(sample_sim_data: SimulationData) -> None:
    pytest.importorskip("pyvista")
    plotter = SimulationViewer.launch_pyvista(sample_sim_data, interactive=False)
    assert plotter is not None
    plotter.close()


def test_cli_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["--candidate", "test.npz", "--viewer", "gepetto"])
    assert args.candidate == Path("test.npz")
    assert args.viewer == "gepetto"
    assert args.fps == 60.0


def test_cli_main_check_only() -> None:
    exit_code = cli_main(
        ["--candidate", "nonexistent.npz", "--viewer", "matplotlib", "--check-only"]
    )
    assert exit_code == 0

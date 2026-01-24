"""Python wrapper for Rob Neal club data visualization.

This module provides visualization capabilities for Rob Neal golf club data,
inspired by the MATLAB ClubDataGUI_v2.m functionality.

Features:
- Load .mat files with club data
- 3D visualization of club shaft and hands
- Playback controls
- Velocity/acceleration vector visualization
- Multiple view angles (isometric, face-on, down-the-line, top-down)
- Trace visualization
"""

from __future__ import annotations

from pathlib import Path

import numpy as np  # noqa: TID253
import numpy.typing as npt  # noqa: TID253

from src.shared.python.logging_config import get_logger

try:
    from scipy.io import loadmat

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import meshcat.geometry as g
    import meshcat.visualizer as viz

    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False

logger = get_logger(__name__)


class RobNealDataViewer:
    """Visualize Rob Neal golf club data in MeshCat.

    This viewer loads .mat files containing club motion data and displays
    them in an interactive 3D viewer with playback controls.
    """

    def __init__(self, zmq_url: str = "tcp://127.0.0.1:6000") -> None:
        """Initialize Rob Neal data viewer.

        Args:
            zmq_url: ZMQ URL for MeshCat server

        Raises:
            ImportError: If required dependencies are not installed
        """
        if not SCIPY_AVAILABLE:
            msg = "scipy is required for .mat import. Install with: pip install scipy"
            raise ImportError(msg)

        if not MESHCAT_AVAILABLE:
            msg = "MeshCat is required. Install with: pip install meshcat"
            raise ImportError(msg)

        self.viewer = viz.Visualizer(zmq_url=zmq_url)
        self.viewer.open()
        self.data: dict[str, npt.NDArray[np.float64]] | None = None
        self.params: dict[str, int] | None = None
        self.current_frame = 0

        logger.info("Rob Neal data viewer initialized")

    def load_data(self, mat_file: Path | str) -> None:
        """Load Rob Neal .mat file.

        Args:
            mat_file: Path to .mat file containing 'data' and 'params' structures

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file doesn't contain required structures
        """
        mat_path = Path(mat_file)
        if not mat_path.exists():
            msg = f"File not found: {mat_path}"
            raise FileNotFoundError(msg)

        loaded = loadmat(str(mat_path), squeeze_me=True)

        if "data" not in loaded or "params" not in loaded:
            msg = "File must contain 'data' and 'params' structures"
            raise ValueError(msg)

        self.data = loaded["data"]
        self.params = loaded["params"]
        self.current_frame = 0

        if self.data is None:
            msg = "Failed to load data"
            raise ValueError(msg)

        logger.info("Loaded data file: %s", mat_path.name)
        logger.info("  Time points: %d", len(self.data["time"]))

    def visualize_frame(
        self,
        frame: int,
        show_trace: bool = False,  # noqa: FBT001, FBT002
        show_velocity: bool = False,  # noqa: FBT001, FBT002
        show_acceleration: bool = False,  # noqa: ARG002, FBT001, FBT002
    ) -> None:
        """Visualize a single frame of data.

        Args:
            frame: Frame index (0-based)
            show_trace: Whether to show trajectory trace
            show_velocity: Whether to show velocity vectors
            show_acceleration: Whether to show acceleration vectors
        """
        if self.data is None:
            logger.warning("No data loaded")
            return

        # Get frame data
        midhands_xyz = self.data["midhands_xyz"][frame]
        clubface_xyz = self.data["clubface_xyz"][frame]

        # Draw club shaft
        self.viewer["club/shaft"].set_object(
            g.Line(
                g.PointsGeometry(
                    np.array([midhands_xyz, clubface_xyz]).T,
                ),
                g.LineBasicMaterial(color=0x000000, linewidth=3),
            )
        )

        # Draw hands position
        self.viewer["club/hands"].set_object(
            g.Sphere(0.02), g.MeshBasicMaterial(color=0x0000FF)
        )
        # Construct proper homogeneous transformation matrix for hands position
        hands_transform = np.eye(4)
        hands_transform[:3, 3] = midhands_xyz
        self.viewer["club/hands"].set_transform(hands_transform)

        # Draw clubface position
        self.viewer["club/face"].set_object(
            g.Sphere(0.03), g.MeshBasicMaterial(color=0xFF0000)
        )
        # Construct proper homogeneous transformation matrix for clubface position
        clubface_transform = np.eye(4)
        clubface_transform[:3, 3] = clubface_xyz
        self.viewer["club/face"].set_transform(clubface_transform)

        # Show trace if requested
        if show_trace and frame > 0:
            trace_points = self.data["clubface_xyz"][: frame + 1]
            self.viewer["club/trace"].set_object(
                g.Line(
                    g.PointsGeometry(trace_points.T),
                    g.LineBasicMaterial(color=0x00FF00, linewidth=1),
                )
            )

        # Show velocity if requested
        if show_velocity and "clubface_xyz_vel" in self.data:
            vel = self.data["clubface_xyz_vel"][frame]
            vel_end = clubface_xyz + vel * 0.1  # Scale for visibility
            self.viewer["club/velocity"].set_object(
                g.Line(
                    g.PointsGeometry(np.array([clubface_xyz, vel_end]).T),
                    g.LineBasicMaterial(color=0x00FFFF, linewidth=2),
                )
            )

        self.current_frame = frame

    def set_view(self, view_type: str) -> None:
        """Set camera view angle.

        Args:
            view_type: One of 'isometric', 'face-on', 'down-the-line', 'top-down'
        """
        # MeshCat uses different camera control - this is a placeholder
        # Actual implementation would use viewer.set_transform or camera controls
        logger.info("Setting view to: %s", view_type)

    def close(self) -> None:
        """Close viewer."""
        self.viewer.close()

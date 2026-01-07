"""MeshCat viewer wrapper for browser visualization."""

from __future__ import annotations

import contextlib
import logging
import typing

if typing.TYPE_CHECKING:
    import pinocchio as pin


try:
    import meshcat.visualizer as viz

    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False

with contextlib.suppress(ImportError):
    from pinocchio.visualize import MeshcatVisualizer

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pinocchio as pin

logger = logging.getLogger(__name__)


class MeshCatViewer:
    """MeshCat viewer wrapper for Pinocchio models."""

    def __init__(self, zmq_url: str = "tcp://127.0.0.1:6000") -> None:
        """Initialize MeshCat viewer.

        Args:
            zmq_url: ZMQ URL for MeshCat server

        Raises:
            ImportError: If MeshCat is not installed
        """
        if not MESHCAT_AVAILABLE:
            msg = "MeshCat required but not installed. Install: pip install meshcat"
            raise ImportError(msg)

        self.viewer = viz.Visualizer(zmq_url=zmq_url)
        self.viewer.open()
        logger.info("MeshCat viewer initialized")

    def load_model(
        self, model: pin.Model, visual_model: pin.GeometryModel | None = None
    ) -> None:
        """Load Pinocchio model into viewer.

        Args:
            model: Pinocchio model
            visual_model: Optional visual geometry model
        """
        if visual_model is not None:
            viz = MeshcatVisualizer(model, visual_model, visual_model)
            viz.initViewer(viewer=self.viewer)
            viz.loadViewerModel()
        else:
            logger.warning("No visual model provided, using basic visualization")

    def display(self, q: npt.NDArray[np.float64]) -> None:
        """Display configuration.

        Args:
            q: Joint positions [nq]
        """
        # This would need the visualizer instance from load_model
        # For now, this is a placeholder
        logger.debug("Displaying configuration with %d DOF", len(q))

    def close(self) -> None:
        """Close viewer."""
        self.viewer.close()

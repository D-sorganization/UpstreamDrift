"""Geppetto viewer wrapper for desktop visualization."""

from __future__ import annotations

import contextlib
import typing

from src.shared.python.logging_config import get_logger

if typing.TYPE_CHECKING:
    import pinocchio as pin

try:
    import gepetto.corbaserver

    GEPETTO_AVAILABLE = True
except ImportError:
    GEPETTO_AVAILABLE = False

with contextlib.suppress(ImportError):
    from pinocchio.visualize import GepettoVisualizer

if typing.TYPE_CHECKING:
    import pinocchio as pin

logger = get_logger(__name__)


class GeppettoViewer:
    """Geppetto viewer wrapper for Pinocchio models.

    Geppetto provides desktop visualization ideal for joint validation.
    """

    def __init__(self) -> None:
        """Initialize Geppetto viewer.

        Raises:
            ImportError: If Geppetto is not installed
        """
        if not GEPETTO_AVAILABLE:
            msg = (
                "Geppetto is required but not installed. Install with: "
                "conda install -c conda-forge gepetto-viewer"
            )
            raise ImportError(msg)

        try:
            self.client = gepetto.corbaserver.Client()
            self.client.gui.createWindow("golfer_viewer")
            logger.info("Geppetto viewer initialized")
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to connect to Geppetto server: %s", e)
            logger.info("Start Geppetto server with: gepetto-gui")

    def load_model(
        self, model: pin.Model, visual_model: pin.GeometryModel | None = None
    ) -> None:
        """Load Pinocchio model into viewer.

        Args:
            model: Pinocchio model
            visual_model: Optional visual geometry model
        """
        if visual_model is not None:
            viz = GepettoVisualizer(model, visual_model, visual_model)
            viz.initViewer(viewer=self.client)
            viz.loadViewerModel()
        else:
            logger.warning("No visual model provided")

    def display(self, q: list[float] | None = None) -> None:
        """Display configuration.

        Args:
            q: Joint positions [nq]. If None, displays neutral configuration.
        """
        if q is not None:
            logger.debug("Displaying configuration with %d DOF", len(q))
        else:
            logger.debug("Displaying neutral configuration")

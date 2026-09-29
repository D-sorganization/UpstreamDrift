"""MeshCat viewer wrapper for browser visualization."""

from __future__ import annotations

import typing
import uuid

from src.shared.python.logging_pkg.logging_config import get_logger

from ._validation import validated_configuration

viz: typing.Any = None
try:
    import meshcat.visualizer as _meshcat_viz

    viz = _meshcat_viz
    MESHCAT_AVAILABLE = True
except (ImportError, OSError):
    MESHCAT_AVAILABLE = False

pin: typing.Any = None
try:
    import pinocchio as _pinocchio

    pin = _pinocchio
except (ImportError, OSError):
    pass

MeshcatVisualizer: typing.Any = None
try:
    from pinocchio.visualize import MeshcatVisualizer as _MeshcatVisualizer

    MeshcatVisualizer = _MeshcatVisualizer
except (ImportError, OSError):
    pass

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pinocchio as pin

logger = get_logger(__name__)
_SERVER_WAIT_TIMEOUT_SECONDS = 5.0


class MeshCatViewer:
    """MeshCat viewer wrapper for Pinocchio models."""

    def __init__(
        self,
        zmq_url: str | None = "tcp://127.0.0.1:6000",
        *,
        open_browser: bool = True,
    ) -> None:
        """Initialize MeshCat viewer.

        Args:
            zmq_url: ZMQ URL for MeshCat server. ``None`` starts a
                self-managed server.
            open_browser: Whether to open the viewer in a browser.

        Raises:
            ImportError: If MeshCat is not installed
        """
        if not MESHCAT_AVAILABLE or viz is None or MeshcatVisualizer is None:
            msg = "MeshCat required but not installed. Install: pip install meshcat"
            raise ImportError(msg)

        self.viewer = viz.Visualizer(zmq_url=zmq_url)
        self._owns_server = zmq_url is None
        self._root_node_name = f"pinocchio_{uuid.uuid4().hex}"
        self._visualizer: typing.Any = None
        self._model: typing.Any = None
        self._closed = False
        if open_browser:
            try:
                self.viewer.open()
            except Exception:
                if self._owns_server:
                    self._stop_owned_server()
                raise
        logger.info("MeshCat viewer initialized")

    def load_model(
        self,
        model: pin.Model,
        visual_model: pin.GeometryModel | None = None,
        *,
        collision_model: pin.GeometryModel | None = None,
    ) -> None:
        """Load Pinocchio model into viewer.

        Args:
            model: Pinocchio model
            visual_model: Optional visual geometry model
            collision_model: Optional collision geometry model
        """
        self._require_open()
        if MeshcatVisualizer is None:
            raise ImportError("Pinocchio MeshcatVisualizer is unavailable")
        self._clean_native(self._visualizer)
        self._visualizer = None
        self._model = None
        native = MeshcatVisualizer(model, collision_model, visual_model)
        try:
            native.initViewer(viewer=self.viewer)
            native.loadViewerModel(rootNodeName=self._root_node_name)
        except Exception:
            self._clean_native(native)
            raise
        self._visualizer = native
        self._model = model

    def display(self, q: npt.NDArray[np.float64] | None = None) -> None:
        """Display configuration.

        Args:
            q: Joint positions [nq]
        """
        self._require_loaded()
        if pin is None:
            raise ImportError("Pinocchio is required to resolve neutral configuration")
        configuration = validated_configuration(self._model, q, pin.neutral)
        self._visualizer.display(configuration)

    def close(self) -> None:
        """Clean the scene and close only a server owned by this adapter."""
        if self._closed:
            return
        self._clean_native(self._visualizer)
        self._visualizer = None
        self._model = None
        if self._owns_server:
            self._stop_owned_server()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("MeshCat viewer is closed")

    def _require_loaded(self) -> None:
        self._require_open()
        if self._visualizer is None or self._model is None:
            raise RuntimeError("No model is loaded in the MeshCat viewer")

    @staticmethod
    def _clean_native(native: typing.Any) -> None:
        if native is None:
            return
        root = getattr(native, "viewerRootNodeName", None)
        backend_viewer = getattr(native, "viewer", None)
        if root and backend_viewer is not None:
            try:
                delete = getattr(backend_viewer[root], "delete", None)
                if callable(delete):
                    delete()
                    return
            except Exception as exc:
                logger.warning("MeshCat scene cleanup was incomplete: %s", exc)
                return
        logger.warning("MeshCat scene root was unavailable during cleanup")

    def _stop_owned_server(self) -> None:
        window = getattr(self.viewer, "window", None)
        server_proc = getattr(window, "server_proc", None)
        poll = getattr(server_proc, "poll", None)
        if callable(poll):
            try:
                if poll() is None:
                    kill = getattr(server_proc, "kill", None)
                    wait = getattr(server_proc, "wait", None)
                    if callable(kill):
                        kill()
                    if callable(wait):
                        wait(timeout=_SERVER_WAIT_TIMEOUT_SECONDS)
                return
            except Exception as exc:
                logger.warning("MeshCat server cleanup was incomplete: %s", exc)
                return
        logger.warning("MeshCat server process was unavailable during cleanup")

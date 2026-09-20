"""Geppetto viewer wrapper for desktop visualization."""

from __future__ import annotations

import typing
import uuid

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

from ._validation import validated_configuration

_corbaserver: typing.Any = None
try:
    import gepetto.corbaserver as _gepetto_corbaserver

    _corbaserver = _gepetto_corbaserver
except (ImportError, OSError):
    pass

pin: typing.Any = None
try:
    import pinocchio as _pinocchio

    pin = _pinocchio
except (ImportError, OSError):
    pass

GepettoVisualizer: typing.Any = None
try:
    from pinocchio.visualize import GepettoVisualizer as _GepettoVisualizer

    GepettoVisualizer = _GepettoVisualizer
except (ImportError, OSError):
    pass

if typing.TYPE_CHECKING:
    import pinocchio as pin

GEPETTO_AVAILABLE = _corbaserver is not None and GepettoVisualizer is not None

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
        if not GEPETTO_AVAILABLE or _corbaserver is None or GepettoVisualizer is None:
            msg = (
                "Geppetto CORBA and Pinocchio visualizer are required. Install "
                "with: conda install -c conda-forge gepetto-viewer-corba pinocchio"
            )
            raise ImportError(msg)

        try:
            self.client = _corbaserver.Client()
        except (RuntimeError, ValueError, OSError) as exc:
            raise RuntimeError(
                "Failed to connect to Geppetto; start gepetto-gui first"
            ) from exc
        self._visualizer: typing.Any = None
        self._model: typing.Any = None
        self._root_node_name = f"pinocchio_{uuid.uuid4().hex}"
        self._closed = False
        logger.info("Geppetto viewer initialized")

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
        if GepettoVisualizer is None:
            raise ImportError("Pinocchio GepettoVisualizer is unavailable")
        self._clean_native(self._visualizer)
        self._visualizer = None
        self._model = None
        native = GepettoVisualizer(model, collision_model, visual_model)
        try:
            native.initViewer(
                viewer=self.client, windowName="python-pinocchio", sceneName="world"
            )
            native.loadViewerModel(rootNodeName=self._root_node_name)
        except Exception:
            self._clean_native(native)
            raise
        self._visualizer = native
        self._model = model

    def display(self, q: list[float] | None = None) -> None:
        """Display configuration.

        Args:
            q: Joint positions [nq]. If None, displays neutral configuration.
        """
        self._require_loaded()
        if pin is None:
            raise ImportError("Pinocchio is required to resolve neutral configuration")
        configuration = validated_configuration(self._model, q, pin.neutral)
        self._visualizer.display(configuration)

    @staticmethod
    def is_server_reachable() -> bool:
        """Check whether Gepetto CORBA server is currently reachable."""
        if not GEPETTO_AVAILABLE or _corbaserver is None:
            return False
        try:
            client = _corbaserver.Client()
            return client is not None
        except Exception:
            return False

    def play_trajectory(
        self,
        q_trajectory: typing.Any,
        dt: float = 1.0 / 60.0,
        *,
        loop: bool = False,
        stride: int = 1,
    ) -> None:
        """Play back a trajectory in the Geppetto viewer.

        Args:
            q_trajectory: Array of configurations of shape (N, nq).
            dt: Time step between frames in seconds.
            loop: Whether to loop playback continuously.
            stride: Stride step between frames (default 1).
        """
        import time

        self._require_loaded()
        q_arr = np.asarray(q_trajectory, dtype=float)
        n_frames = len(q_arr)
        if n_frames == 0:
            return

        while True:
            for k in range(0, n_frames, max(1, stride)):
                t_start = time.perf_counter()
                self.display(q_arr[k].tolist())
                elapsed = time.perf_counter() - t_start
                sleep_time = (dt * max(1, stride)) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if not loop:
                break

    def close(self) -> None:
        """Clean the scene without shutting down the shared Geppetto server."""
        if self._closed:
            return
        self._clean_native(self._visualizer)
        self._visualizer = None
        self._model = None
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Geppetto viewer is closed")

    def _require_loaded(self) -> None:
        self._require_open()
        if self._visualizer is None or self._model is None:
            raise RuntimeError("No model is loaded in the Geppetto viewer")

    @staticmethod
    def _clean_native(native: typing.Any) -> None:
        if native is None:
            return
        root = getattr(native, "viewerRootNodeName", None)
        backend_viewer = getattr(native, "viewer", None)
        gui = getattr(backend_viewer, "gui", None)
        delete = getattr(gui, "deleteNode", None)
        if root and callable(delete):
            try:
                delete(root, True)
                return
            except Exception as exc:
                logger.warning("Geppetto scene cleanup was incomplete: %s", exc)
                return
        logger.warning("Geppetto scene root was unavailable during cleanup")

from __future__ import annotations

import asyncio
import logging
import threading
import time
from queue import Empty, Queue

import numpy as np

from src.unreal_integration.data_models import (
    BallState,
    ClubState,
    JointState,
    Quaternion,
    UnrealDataFrame,
    Vector3,
)
from src.unreal_integration.mesh_loader import LoadedMesh
from src.unreal_integration.streaming import StreamingConfig, UnrealStreamingServer

from ._viewer_base import ViewerBackend, ViewerConfig

logger = logging.getLogger(__name__)


class UnrealBridgeBackend(ViewerBackend):
    """Unreal Engine bridge viewer backend.

    Streams simulation data to Unreal Engine via WebSocket.
    Runs the streaming server in a background thread to maintain
    responsiveness of the main simulation loop.

    Example:
        >>> backend = UnrealBridgeBackend()
        >>> with backend:
        ...     backend.add_mesh(mesh, name="club")
        ...     backend.render()
    """

    def __init__(self, config: ViewerConfig | None = None) -> None:
        """Initialize Unreal Bridge backend.

        Args:
            config: Viewer configuration.
        """
        super().__init__(config)
        self._server: UnrealStreamingServer | None = None
        self._server_thread: threading.Thread | None = None
        self._frame_queue: Queue[UnrealDataFrame] = Queue()
        self._stop_event = threading.Event()
        self._server_ready_event = threading.Event()
        self._server_start_error: BaseException | None = None
        self._frame_counter = 0
        self._object_counter = 0
        self._start_time = 0.0

    def initialize(self) -> None:
        """Initialize streaming server in background thread.

        Raises:
            RuntimeError: If the streaming server fails to bind its socket or
                does not become ready within the startup timeout.
        """
        if self._is_initialized:
            return

        self._start_time = time.time()
        self._server_start_error = None

        # Configure streaming
        streaming_config = StreamingConfig(
            host=self.config.server_host,
            port=self.config.server_port,
        )
        self._server = UnrealStreamingServer(config=streaming_config)

        # Start background thread
        self._stop_event.clear()
        self._server_ready_event.clear()
        self._server_thread = threading.Thread(
            target=self._run_server_loop, daemon=True, name="UnrealBridgeThread"
        )
        self._server_thread.start()

        # Block until the server loop signals readiness (or failure).
        if not self._server_ready_event.wait(timeout=5.0):
            self._stop_event.set()
            raise RuntimeError(
                f"Unreal streaming server did not start within 5 s "
                f"(port {self.config.server_port})"
            )

        # Re-raise any error captured by the background thread.
        if self._server_start_error is not None:
            err = self._server_start_error
            self._server_start_error = None
            raise RuntimeError(
                f"Unreal streaming server failed to start: {err}"
            ) from err

        self._is_initialized = True
        logger.info(
            f"Unreal Bridge backend initialized on port {self.config.server_port}"
        )

    def _run_server_loop(self) -> None:
        """Run the asyncio event loop for the server."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run() -> None:
            if self._server is None:
                return

            # Start server — capture failures and always signal the ready event
            # so initialize() does not hang on a timeout.
            try:
                await self._server.start()
            except (OSError, RuntimeError) as exc:
                self._server_start_error = exc
                self._server_ready_event.set()
                return

            self._server_ready_event.set()

            # Process frames
            while not self._stop_event.is_set():
                try:
                    # Non-blocking check for new frames
                    # We check frequently but sleep briefly to avoid busy loop
                    try:
                        frame = self._frame_queue.get_nowait()
                        await self._server.broadcast(frame)
                        self._frame_queue.task_done()
                    except Empty:
                        await asyncio.sleep(0.001)

                except (OSError, RuntimeError) as e:
                    logger.error(f"Error in streaming loop: {e}")
                    await asyncio.sleep(1.0)  # Backoff on error

            # Stop server
            await self._server.stop()

        try:
            loop.run_until_complete(run())
        finally:
            loop.close()

    def shutdown(self) -> None:
        """Shutdown streaming server."""
        self._stop_event.set()
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            if self._server_thread.is_alive():
                logger.warning("Unreal Bridge thread did not stop cleanly")
            self._server_thread = None

        self._server = None
        self._is_initialized = False
        self._objects.clear()
        logger.info("Unreal Bridge backend shutdown")

    def add_mesh(
        self,
        mesh: LoadedMesh,
        name: str | None = None,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: float = 1.0,
    ) -> str:
        """Add mesh to tracked objects."""
        if mesh is None:
            raise ValueError("mesh must be provided")
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        if name is None:
            name = f"mesh_{self._object_counter}"
            self._object_counter += 1

        self._objects[name] = {
            "mesh": mesh,
            "position": position or Vector3.zero(),
            "rotation": rotation or Quaternion.identity(),
            "scale": scale,
        }
        return name

    def update_transform(
        self,
        name: str,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: float | None = None,
    ) -> None:
        """Update object transform."""
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        if name not in self._objects:
            logger.warning(f"Object not found: {name}")
            return

        obj = self._objects[name]
        if position is not None:
            obj["position"] = position
        if rotation is not None:
            obj["rotation"] = rotation
        if scale is not None:
            obj["scale"] = scale

    def remove_object(self, name: str) -> bool:
        """Remove object."""
        if name is None:
            raise ValueError("name must be provided")
        if not self._is_initialized:
            return False

        if name in self._objects:
            del self._objects[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all objects."""
        self._objects.clear()

    def render(self) -> np.ndarray | None:
        """Queue current frame for streaming."""
        if not self._is_initialized:
            return None

        # Construct UnrealDataFrame
        timestamp = time.time() - self._start_time

        joints: dict[str, JointState] = {}
        club: ClubState | None = None
        ball: BallState | None = None

        for name, obj in self._objects.items():
            pos = obj["position"]
            rot = obj["rotation"]

            # Map known names to specific fields
            if name == "club":
                club = ClubState(
                    head_position=pos,
                    # We don't track velocity/acceleration in ViewerBackend currently
                    head_velocity=Vector3.zero(),
                )
            elif name == "ball":
                ball = BallState(
                    position=pos,
                    velocity=Vector3.zero(),
                )
            else:
                # Map everything else to joints
                joints[name] = JointState(
                    name=name,
                    position=pos,
                    rotation=rot,
                )

        frame = UnrealDataFrame(
            timestamp=timestamp,
            frame_number=self._frame_counter,
            joints=joints,
            club=club,
            ball=ball,
        )
        self._frame_counter += 1

        # Queue for sending
        self._frame_queue.put(frame)

        # No image return for streaming backend
        return None

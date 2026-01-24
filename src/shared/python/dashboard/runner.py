"""Simulation runner thread for the Unified Dashboard.

Handles the simulation loop, timing, and recording updates.
"""

from __future__ import annotations

import time

from PyQt6 import QtCore

from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.interfaces import PhysicsEngine
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


class SimulationRunner(QtCore.QThread):
    """Runs the physics simulation in a separate thread."""

    frame_ready = QtCore.pyqtSignal()  # Signal to update GUI
    simulation_finished = QtCore.pyqtSignal()
    status_message = QtCore.pyqtSignal(str)

    def __init__(
        self,
        engine: PhysicsEngine,
        recorder: GenericPhysicsRecorder,
        target_fps: float = 60.0,
    ) -> None:
        """Initialize runner.

        Args:
            engine: Physics engine instance
            recorder: Recorder instance
            target_fps: Target frames per second
        """
        super().__init__()
        self.engine = engine
        self.recorder = recorder
        self.target_fps = target_fps
        self.running = False
        self.paused = False
        self.step_count = 0
        self.max_steps = 10000  # Default limit to prevent infinite run

    def run(self) -> None:
        """Main simulation loop."""
        self.running = True
        self.recorder.reset()
        self.recorder.start()
        self.step_count = 0

        target_dt = 1.0 / self.target_fps

        # Determine simulation timestep from engine if possible, else use default
        # Assuming engine handles its own timestep in step() if dt=None
        # But for synchronization we sleep.

        logger.info("Simulation started.")
        self.status_message.emit("Simulation running...")

        while self.running and self.step_count < self.max_steps:
            start_time = time.time()

            if not self.paused:
                try:
                    # Step physics
                    self.engine.step()

                    # Record data
                    self.recorder.record_step()

                    self.step_count += 1

                    # Emit update signal periodically
                    # (Emit every frame for smooth plot, but might overwhelm if too fast)
                    self.frame_ready.emit()

                except Exception as e:
                    logger.error("Simulation error: %s", e)
                    self.status_message.emit(f"Error: {e}")
                    break

            # Sleep to maintain FPS
            elapsed = time.time() - start_time
            sleep_time = max(0.0, target_dt - elapsed)

            if sleep_time > 0:
                self.msleep(int(sleep_time * 1000))

        self.recorder.stop()
        self.running = False
        self.simulation_finished.emit()
        self.status_message.emit("Simulation finished.")
        logger.info("Simulation finished.")

    def stop(self) -> None:
        """Stop the simulation."""
        self.running = False
        self.wait()

    def toggle_pause(self) -> None:
        """Toggle pause state."""
        self.paused = not self.paused
        state = "Paused" if self.paused else "Running"
        self.status_message.emit(f"Simulation {state}")

"""Interfaces for robot communication protocols."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from src.deployment.realtime.state import ControlCommand, ControlMode, RobotState

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.deployment.realtime.controller import RobotConfig

logger = logging.getLogger(__name__)


class RobotInterface(ABC):
    """Abstract base class for robot communication interfaces."""

    def __init__(self, config: RobotConfig) -> None:
        """Initialize the interface.

        Args:
            config: Robot configuration.
        """
        self.config = config
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to robot."""
        return self._is_connected

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the robot."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot."""

    @abstractmethod
    def read_state(self) -> RobotState:
        """Read the current robot state."""

    @abstractmethod
    def send_command(self, command: ControlCommand) -> None:
        """Send a control command to the robot."""


class SimulationInterface(RobotInterface):
    """Interface for simulated robot (returns zeros)."""

    def __init__(self, config: RobotConfig) -> None:
        super().__init__(config)
        self._start_time = 0.0

    def connect(self) -> bool:
        self._start_time = time.perf_counter()
        self._is_connected = True
        return True

    def disconnect(self) -> None:
        self._is_connected = False

    def read_state(self) -> RobotState:
        timestamp = time.perf_counter() - self._start_time
        n_joints = self.config.n_joints
        return RobotState(
            timestamp=timestamp,
            joint_positions=np.zeros(n_joints),
            joint_velocities=np.zeros(n_joints),
            joint_torques=np.zeros(n_joints),
        )

    def send_command(self, command: ControlCommand) -> None:
        # Simulation just accepts commands
        pass


class LoopbackInterface(RobotInterface):
    """Interface for loopback simulation (double integrator physics)."""

    def __init__(self, config: RobotConfig, dt: float) -> None:
        super().__init__(config)
        self.dt = dt
        self._start_time = 0.0
        self._sim_state: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None
        self._lock = threading.Lock()

    def connect(self) -> bool:
        self._start_time = time.perf_counter()
        n_joints = self.config.n_joints
        self._sim_state = (np.zeros(n_joints), np.zeros(n_joints))
        self._is_connected = True
        return True

    def disconnect(self) -> None:
        self._is_connected = False
        self._sim_state = None

    def read_state(self) -> RobotState:
        timestamp = time.perf_counter() - self._start_time
        n_joints = self.config.n_joints

        with self._lock:
            if self._sim_state is None:
                self._sim_state = (np.zeros(n_joints), np.zeros(n_joints))
            q, qd = self._sim_state

        return RobotState(
            timestamp=timestamp,
            joint_positions=q.copy(),
            joint_velocities=qd.copy(),
            joint_torques=np.zeros(n_joints),
        )

    def send_command(self, command: ControlCommand) -> None:
        with self._lock:
            if self._sim_state is None:
                n_joints = self.config.n_joints
                self._sim_state = (np.zeros(n_joints), np.zeros(n_joints))

            q, qd = self._sim_state

            if command.mode == ControlMode.TORQUE:
                if command.torque_commands is not None:
                    # Simple double integrator: acc = torque (assuming unit mass)
                    # Add damping to prevent instability
                    damping = 0.1
                    qdd = command.torque_commands - damping * qd
                    qd = qd + qdd * self.dt
                    q = q + qd * self.dt

            elif command.mode == ControlMode.POSITION:
                if command.position_targets is not None:
                    # Instantaneous position control (infinite gain)
                    q = command.position_targets
                    # Reset velocity or leave it? Let's zero it to be safe
                    qd = np.zeros_like(q)

            elif command.mode == ControlMode.VELOCITY:
                if command.velocity_targets is not None:
                    qd = command.velocity_targets
                    q = q + qd * self.dt

            elif (
                command.mode == ControlMode.IMPEDANCE
                and command.position_targets is not None
                and command.stiffness is not None
                and command.damping is not None
            ):
                # Impedance control: tau = K(q_d - q) + D(0 - qd)
                # acc = tau (unit mass)
                q_err = command.position_targets - q
                tau = command.stiffness * q_err - command.damping * qd
                if command.feedforward_torque is not None:
                    tau += command.feedforward_torque

                qdd = tau
                qd = qd + qdd * self.dt
                q = q + qd * self.dt

            self._sim_state = (q, qd)


class UDPInterface(RobotInterface):
    """Interface for UDP communication."""

    def __init__(self, config: RobotConfig) -> None:
        super().__init__(config)
        self._socket: socket.socket | None = None
        self._address = (config.ip_address, config.port)
        self._start_time = 0.0

    def connect(self) -> bool:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.settimeout(0.1)  # Non-blocking with timeout
            # Send a handshake or initial packet to verify connectivity?
            # For UDP, "connect" is virtual. We can just check if we can bind/resolve.
            self._start_time = time.perf_counter()
            self._is_connected = True
            logger.info(f"UDP interface initialized for {self._address}")
            return True
        except (OSError, ValueError) as e:
            logger.error(f"Failed to initialize UDP socket: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        if self._socket:
            self._socket.close()
            self._socket = None
        self._is_connected = False

    def read_state(self) -> RobotState:
        if not self._socket:
            raise RuntimeError("UDP socket not connected")

        timestamp = time.perf_counter() - self._start_time
        n_joints = self.config.n_joints

        try:
            data, _ = self._socket.recvfrom(4096)
            # Simple JSON protocol for now
            state_dict = json.loads(data.decode("utf-8"))

            return RobotState(
                timestamp=state_dict.get("timestamp", timestamp),
                joint_positions=np.array(state_dict["joint_positions"]),
                joint_velocities=np.array(state_dict["joint_velocities"]),
                joint_torques=np.array(state_dict["joint_torques"]),
            )

        except TimeoutError:
            # Return last state or zeros if timeout?
            # For strict real-time, missing a packet is bad, but we can't block forever.
            logger.warning("UDP receive timeout")
            return RobotState(
                timestamp=timestamp,
                joint_positions=np.zeros(n_joints),
                joint_velocities=np.zeros(n_joints),
                joint_torques=np.zeros(n_joints),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to decode UDP packet: {e}")
            return RobotState(
                timestamp=timestamp,
                joint_positions=np.zeros(n_joints),
                joint_velocities=np.zeros(n_joints),
                joint_torques=np.zeros(n_joints),
            )

    def send_command(self, command: ControlCommand) -> None:
        if not self._socket:
            raise RuntimeError("UDP socket not connected")

        try:
            # Serialize command to JSON
            cmd_dict = {
                "timestamp": command.timestamp,
                "mode": command.mode.value,
            }

            if command.position_targets is not None:
                cmd_dict["position_targets"] = command.position_targets.tolist()
            if command.velocity_targets is not None:
                cmd_dict["velocity_targets"] = command.velocity_targets.tolist()
            if command.torque_commands is not None:
                cmd_dict["torque_commands"] = command.torque_commands.tolist()
            if command.stiffness is not None:
                cmd_dict["stiffness"] = command.stiffness.tolist()
            if command.damping is not None:
                cmd_dict["damping"] = command.damping.tolist()
            if command.feedforward_torque is not None:
                cmd_dict["feedforward_torque"] = command.feedforward_torque.tolist()

            data = json.dumps(cmd_dict).encode("utf-8")
            self._socket.sendto(data, self._address)

        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to send UDP command: {e}")


class ROS2Interface(RobotInterface):
    """Interface for ROS2 communication."""

    def connect(self) -> bool:
        # Placeholder for ROS2 node initialization
        raise NotImplementedError("ROS2 interface not yet implemented")

    def disconnect(self) -> None:
        pass

    def read_state(self) -> RobotState:
        raise NotImplementedError("ROS2 interface not yet implemented")

    def send_command(self, command: ControlCommand) -> None:
        raise NotImplementedError("ROS2 interface not yet implemented")


class EtherCATInterface(RobotInterface):
    """Interface for EtherCAT communication."""

    def connect(self) -> bool:
        # Placeholder for EtherCAT master initialization
        raise NotImplementedError("EtherCAT interface not yet implemented")

    def disconnect(self) -> None:
        pass

    def read_state(self) -> RobotState:
        raise NotImplementedError("EtherCAT interface not yet implemented")

    def send_command(self, command: ControlCommand) -> None:
        raise NotImplementedError("EtherCAT interface not yet implemented")

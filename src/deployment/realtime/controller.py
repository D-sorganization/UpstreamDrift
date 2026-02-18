"""Real-time control loop manager for robot hardware."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from src.deployment.realtime.state import ControlCommand, ControlMode, RobotState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CommunicationType(Enum):
    """Communication protocol for robot interface."""

    ETHERCAT = "ethercat"
    ROS2 = "ros2"
    UDP = "udp"
    SIMULATION = "simulation"
    LOOPBACK = "loopback"


@dataclass
class RobotConfig:
    """Configuration for robot hardware interface.

    Attributes:
        name: Robot identifier.
        n_joints: Number of joints.
        joint_names: Names of each joint.
        joint_limits_lower: Lower joint position limits (rad).
        joint_limits_upper: Upper joint position limits (rad).
        velocity_limits: Maximum joint velocities (rad/s).
        torque_limits: Maximum joint torques (Nm).
        communication_type: Communication protocol.
        ip_address: Robot IP address (if networked).
        port: Communication port.
    """

    name: str
    n_joints: int
    joint_names: list[str] = field(default_factory=list)
    joint_limits_lower: NDArray[np.floating] | None = None
    joint_limits_upper: NDArray[np.floating] | None = None
    velocity_limits: NDArray[np.floating] | None = None
    torque_limits: NDArray[np.floating] | None = None
    communication_type: CommunicationType = CommunicationType.SIMULATION
    ip_address: str = "127.0.0.1"
    port: int = 5000

    def __post_init__(self) -> None:
        """Set default joint names if not provided."""
        if not self.joint_names:
            self.joint_names = [f"joint_{i}" for i in range(self.n_joints)]


@dataclass
class TimingStatistics:
    """Statistics for control loop timing.

    Attributes:
        mean_cycle_time: Mean cycle time (seconds).
        max_cycle_time: Maximum cycle time (seconds).
        min_cycle_time: Minimum cycle time (seconds).
        std_cycle_time: Standard deviation of cycle time.
        jitter: Maximum deviation from target period.
        overruns: Number of cycle time overruns.
        total_cycles: Total number of control cycles.
        uptime: Total running time (seconds).
    """

    mean_cycle_time: float = 0.0
    max_cycle_time: float = 0.0
    min_cycle_time: float = float("inf")
    std_cycle_time: float = 0.0
    jitter: float = 0.0
    overruns: int = 0
    total_cycles: int = 0
    uptime: float = 0.0


class RealTimeController:
    """Real-time control loop manager for robot hardware.

    Manages the high-frequency control loop that interfaces
    with robot hardware, providing:
    - Consistent timing at specified frequency
    - State reading and command writing
    - Timing statistics and jitter monitoring
    - Safe shutdown procedures

    Attributes:
        control_frequency: Control loop frequency (Hz).
        comm_type: Communication protocol type.
        config: Robot configuration.
        is_connected: Whether connected to robot.
        is_running: Whether control loop is active.
    """

    def __init__(
        self,
        control_frequency: float = 1000.0,
        communication_type: str = "simulation",
    ) -> None:
        """Initialize real-time controller.

        Args:
            control_frequency: Control loop frequency in Hz.
            communication_type: Communication protocol.
        """
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.comm_type = CommunicationType(communication_type)

        self._config: RobotConfig | None = None
        self._control_callback: Callable[[RobotState], ControlCommand] | None = None
        self._is_connected = False
        self._is_running = False
        self._should_stop = False

        # Control loop thread
        self._control_thread: threading.Thread | None = None

        # Timing statistics
        self._cycle_times: list[float] = []
        self._start_time: float = 0.0
        self._overruns = 0

        # State buffers
        self._last_state: RobotState | None = None
        self._last_command: ControlCommand | None = None

        # Loopback physics state
        self._sim_state: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None

        # Locks
        self._state_lock = threading.Lock()
        self._command_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        """Check if connected to robot."""
        return self._is_connected

    @property
    def is_running(self) -> bool:
        """Check if control loop is running."""
        return self._is_running

    def connect(self, robot_config: RobotConfig) -> bool:
        """Connect to real robot.

        Args:
            robot_config: Robot configuration.

        Returns:
            True if connection successful.
        """
        self._config = robot_config

        try:
            if self.comm_type in (
                CommunicationType.SIMULATION,
                CommunicationType.LOOPBACK,
            ):
                # Simulated connection always succeeds
                self._is_connected = True
                # Reset simulation state on connect to ensure correct sizing
                self._sim_state = None
            elif self.comm_type == CommunicationType.ROS2:
                self._connect_ros2()
            elif self.comm_type == CommunicationType.UDP:
                self._connect_udp()
            elif self.comm_type == CommunicationType.ETHERCAT:
                self._connect_ethercat()

            self._is_connected = True
            return True

        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Failed to connect: %s", e)
            self._is_connected = False
            return False

    def _connect_ros2(self) -> None:
        """Connect via ROS2."""
        # ROS2 connection would be implemented here
        # For now, placeholder for integration

    def _connect_udp(self) -> None:
        """Connect via UDP socket."""
        # UDP socket connection would be implemented here

    def _connect_ethercat(self) -> None:
        """Connect via EtherCAT."""
        # EtherCAT connection would be implemented here

    def disconnect(self) -> None:
        """Safely disconnect from robot."""
        if self._is_running:
            self.stop()

        self._is_connected = False
        self._config = None

    def set_control_callback(
        self,
        callback: Callable[[RobotState], ControlCommand],
    ) -> None:
        """Set the control callback function.

        The callback is called at control_frequency with the
        current robot state and must return a control command.

        Args:
            callback: Function mapping RobotState to ControlCommand.
        """
        self._control_callback = callback

    def start(self) -> None:
        """Start real-time control loop.

        Raises:
            RuntimeError: If not connected or no callback set.
        """
        if not self._is_connected:
            raise RuntimeError("Must connect to robot before starting")
        if self._control_callback is None:
            raise RuntimeError("Must set control callback before starting")
        if self._is_running:
            return

        self._should_stop = False
        self._is_running = True
        self._start_time = time.perf_counter()
        self._cycle_times = []
        self._overruns = 0

        self._control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True,
        )
        self._control_thread.start()

    def stop(self) -> None:
        """Stop control loop safely."""
        self._should_stop = True

        if self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None

        self._is_running = False

        # Send zero command
        if self._config is not None:
            zero_command = ControlCommand(
                timestamp=time.perf_counter(),
                mode=ControlMode.TORQUE,
                torque_commands=np.zeros(self._config.n_joints),
            )
            self._send_command(zero_command)

    def _control_loop(self) -> None:
        """Main real-time control loop."""
        next_cycle_time = time.perf_counter()

        while not self._should_stop:
            cycle_start = time.perf_counter()

            try:
                # Read state
                state = self._read_state()

                with self._state_lock:
                    self._last_state = state

                # Compute control
                if self._control_callback is not None:
                    command = self._control_callback(state)

                    # Validate command
                    if self._config is not None:
                        command.validate(self._config.n_joints)

                    # Send command
                    self._send_command(command)

                    with self._command_lock:
                        self._last_command = command

            except (RuntimeError, ValueError, OSError) as e:
                logger.error("Control loop error: %s", e)

            # Record timing
            cycle_end = time.perf_counter()
            cycle_time = cycle_end - cycle_start
            self._cycle_times.append(cycle_time)

            # Check for overrun
            if cycle_time > self.dt:
                self._overruns += 1

            # Sleep until next cycle
            next_cycle_time += self.dt
            sleep_time = next_cycle_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Missed deadline, reset timing
                next_cycle_time = time.perf_counter()

        self._is_running = False

    def _read_state(self) -> RobotState:
        """Read current robot state.

        Returns:
            Current robot state.
        """
        timestamp = time.perf_counter() - self._start_time

        if self.comm_type == CommunicationType.SIMULATION:
            # Return simulated state
            n_joints = self._config.n_joints if self._config else 7
            return RobotState(
                timestamp=timestamp,
                joint_positions=np.zeros(n_joints),
                joint_velocities=np.zeros(n_joints),
                joint_torques=np.zeros(n_joints),
            )

        if self.comm_type == CommunicationType.LOOPBACK:
            n_joints = self._config.n_joints if self._config else 7
            if self._sim_state is None:
                self._sim_state = (np.zeros(n_joints), np.zeros(n_joints))

            # Atomic read of state tuple
            current_sim_state = self._sim_state
            return RobotState(
                timestamp=timestamp,
                joint_positions=current_sim_state[0],
                joint_velocities=current_sim_state[1],
                joint_torques=np.zeros(n_joints),
            )

        # Real hardware reading would be implemented per protocol
        raise NotImplementedError(
            f"State reading for communication type '{self.comm_type.value}' is not yet "
            f"supported. Currently only CommunicationType.SIMULATION is implemented."
        )

    def _send_command(self, command: ControlCommand) -> None:
        """Send control command to robot.

        Args:
            command: Control command to send.
        """
        if self.comm_type == CommunicationType.SIMULATION:
            # Simulated: command is "sent"
            return

        if self.comm_type == CommunicationType.LOOPBACK:
            if self._sim_state is None:
                # Initialize state if not present (should be handled in _read_state, but safety check)
                n_joints = self._config.n_joints if self._config else 7
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
                    # Reset velocity or leave it? Let's zero it to be safe as position jump implies infinite velocity
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
            return

        # Real hardware command sending would be implemented per protocol
        raise NotImplementedError(
            f"Command sending for communication type '{self.comm_type.value}' is not yet "
            f"supported. Currently only CommunicationType.SIMULATION is implemented."
        )

    def get_timing_stats(self) -> TimingStatistics:
        """Get control loop timing statistics.

        Returns:
            Timing statistics.
        """
        if not self._cycle_times:
            return TimingStatistics()

        cycle_times = np.array(self._cycle_times)
        target_period = self.dt

        return TimingStatistics(
            mean_cycle_time=float(np.mean(cycle_times)),
            max_cycle_time=float(np.max(cycle_times)),
            min_cycle_time=float(np.min(cycle_times)),
            std_cycle_time=float(np.std(cycle_times)),
            jitter=float(np.max(np.abs(cycle_times - target_period))),
            overruns=self._overruns,
            total_cycles=len(self._cycle_times),
            uptime=time.perf_counter() - self._start_time if self._start_time else 0,
        )

    def get_last_state(self) -> RobotState | None:
        """Get most recent robot state.

        Returns:
            Last received state or None.
        """
        with self._state_lock:
            return self._last_state

    def get_last_command(self) -> ControlCommand | None:
        """Get most recent command.

        Returns:
            Last sent command or None.
        """
        with self._command_lock:
            return self._last_command

    def wait_for_state(self, timeout: float = 1.0) -> RobotState | None:
        """Wait for a new state reading.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            Robot state or None if timeout.
        """
        start = time.perf_counter()
        initial_state = self._last_state

        while time.perf_counter() - start < timeout:
            with self._state_lock:
                if self._last_state is not initial_state:
                    return self._last_state
            time.sleep(0.001)

        return None

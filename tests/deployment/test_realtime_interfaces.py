"""Tests for real-time robot interfaces."""

import json
import socket
import threading
import time
from typing import Any

import numpy as np
import pytest

from src.deployment.realtime.controller import CommunicationType, RobotConfig
from src.deployment.realtime.interfaces import (
    EtherCATInterface,
    LoopbackInterface,
    ROS2Interface,
    SimulationInterface,
    UDPInterface,
)
from src.deployment.realtime.state import ControlCommand, ControlMode


class MockUDPServer:
    """Mock UDP server for testing."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.1)
        self.running = False
        self.thread: threading.Thread | None = None
        self.received_commands: list[dict[str, Any]] = []
        self.state_to_send: dict[str, Any] | None = None

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join()
        self.sock.close()

    def _loop(self) -> None:
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                cmd = json.loads(data.decode("utf-8"))
                self.received_commands.append(cmd)

                # Send response if state is set
                if self.state_to_send:
                    resp = json.dumps(self.state_to_send).encode("utf-8")
                    self.sock.sendto(resp, addr)

            except TimeoutError:
                continue
            except Exception as e:
                print(f"Mock server error: {e}")


class TestUDPInterface:
    """Tests for UDPInterface."""

    @pytest.fixture
    def udp_config(self) -> RobotConfig:
        return RobotConfig(
            name="udp_robot",
            n_joints=7,
            communication_type=CommunicationType.UDP,
            ip_address="127.0.0.1",
            port=5005,
        )

    @pytest.fixture
    def mock_server(self) -> Any:
        server = MockUDPServer(5005)
        server.start()
        yield server
        server.stop()

    def test_udp_connect(self, udp_config: RobotConfig) -> None:
        interface = UDPInterface(udp_config)
        assert interface.connect()
        assert interface.is_connected
        interface.disconnect()
        assert not interface.is_connected

    def test_udp_send_receive(
        self, udp_config: RobotConfig, mock_server: MockUDPServer
    ) -> None:
        interface = UDPInterface(udp_config)
        interface.connect()

        # Prepare server response
        state_data = {
            "timestamp": 1.0,
            "joint_positions": [0.1] * 7,
            "joint_velocities": [0.01] * 7,
            "joint_torques": [0.0] * 7,
        }
        mock_server.state_to_send = state_data

        # Send command
        cmd = ControlCommand(
            timestamp=1.0,
            mode=ControlMode.TORQUE,
            torque_commands=np.ones(7),
        )
        interface.send_command(cmd)

        # Allow time for transmission
        time.sleep(0.1)

        # Verify command received by server
        assert len(mock_server.received_commands) > 0
        last_cmd = mock_server.received_commands[-1]
        assert last_cmd["mode"] == "torque"
        assert last_cmd["torque_commands"] == [1.0] * 7

        # Read state
        state = interface.read_state()
        np.testing.assert_allclose(state.joint_positions, np.array([0.1] * 7))

        interface.disconnect()


class TestLoopbackInterface:
    """Tests for LoopbackInterface."""

    def test_loopback_physics(self) -> None:
        config = RobotConfig(name="loopback", n_joints=1)
        interface = LoopbackInterface(config, dt=0.01)
        interface.connect()

        # Initial state should be zero
        state = interface.read_state()
        assert state.joint_positions[0] == 0.0

        # Apply torque
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.TORQUE,
            torque_commands=np.array([1.0]),
        )
        interface.send_command(cmd)

        # Physics check: qdd = tau - damping*qd
        # Step 1: qd += (1 - 0) * 0.01 = 0.01, q += 0.01 * 0.01 = 0.0001
        state = interface.read_state()
        assert state.joint_velocities[0] > 0
        assert state.joint_positions[0] > 0

        interface.disconnect()


class TestOtherInterfaces:
    """Tests for other interfaces."""

    def test_simulation_interface(self) -> None:
        config = RobotConfig(name="sim", n_joints=7)
        interface = SimulationInterface(config)
        interface.connect()
        state = interface.read_state()
        assert np.all(state.joint_positions == 0)
        interface.disconnect()

    def test_ros2_interface(self) -> None:
        config = RobotConfig(name="ros2", n_joints=7)
        interface = ROS2Interface(config)
        with pytest.raises(NotImplementedError):
            interface.connect()

    def test_ethercat_interface(self) -> None:
        config = RobotConfig(name="ethercat", n_joints=7)
        interface = EtherCATInterface(config)
        with pytest.raises(NotImplementedError):
            interface.connect()

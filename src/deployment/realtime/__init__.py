"""Real-Time Control Interface for robot hardware.

This module provides the interface layer between simulation
and real robot hardware with support for:
- High-frequency control loops (up to 1kHz)
- Multiple communication protocols (EtherCAT, ROS2, UDP)
- Timing statistics and jitter monitoring
"""

from __future__ import annotations

from src.deployment.realtime.controller import (
    RealTimeController,
    RobotConfig,
    TimingStatistics,
)
from src.deployment.realtime.state import (
    ControlCommand,
    ControlMode,
    IMUReading,
    RobotState,
)

__all__ = [
    "RealTimeController",
    "RobotConfig",
    "RobotState",
    "ControlCommand",
    "ControlMode",
    "IMUReading",
    "TimingStatistics",
]

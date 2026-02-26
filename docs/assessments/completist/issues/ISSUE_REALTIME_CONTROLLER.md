# GitHub Issue Draft: Critical Incomplete Implementation in RealTimeController

**Title:** [CRITICAL] Implement missing RealTimeController connectivity and I/O methods
**Labels:** incomplete-implementation, critical

## Description
The `RealTimeController` class in `src/deployment/realtime/controller.py` contains placeholder methods for critical hardware connectivity and I/O features. These methods raise `NotImplementedError` or are empty pass-throughs when called for non-simulation modes, effectively blocking any integration with robotic hardware via ROS2, UDP, or EtherCAT.

## Impact
- **Blocking:** Users cannot connect to, read from, or control physical robots using standard protocols.
- **Critical:** This functionality is core to the deployment module; currently only simulation mode works.

## Technical Details
The following methods need implementation:
1.  **Connectivity**:
    - `_connect_ros2(self)`: Should initialize `rclpy` node and publishers/subscribers.
    - `_connect_udp(self)`: Should create a UDP socket and bind to the configured port.
    - `_connect_ethercat(self)`: Should initialize the EtherCAT master (e.g., using `pysoem`).

2.  **Input/Output**:
    - `_read_state(self)`: Raises `NotImplementedError` for ROS2, UDP, and EtherCAT. Needs to implement hardware-specific state reading logic.
    - `_send_command(self)`: Raises `NotImplementedError` for ROS2, UDP, and EtherCAT. Needs to implement hardware-specific command sending logic.

## Acceptance Criteria
- [ ] `_connect_*` methods successfully initialize communication for their respective protocols.
- [ ] `_read_state` returns valid `RobotState` objects from hardware streams.
- [ ] `_send_command` correctly transmits `ControlCommand` data to hardware.
- [ ] Unit tests are added to verify logic (mocking external libraries where necessary).
- [ ] `NotImplementedError` is removed from these methods.

## References
- File: `src/deployment/realtime/controller.py`
- Completist Report: `docs/assessments/completist/COMPLETIST_LATEST.md`

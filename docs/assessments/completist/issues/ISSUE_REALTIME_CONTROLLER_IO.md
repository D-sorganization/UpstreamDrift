# GitHub Issue Draft: Critical Missing IO in RealTimeController

**Title:** [CRITICAL] Implement RealTimeController I/O methods (_read_state, _send_command)
**Labels:** incomplete-implementation, critical

## Description
The `RealTimeController` class in `src/deployment/realtime/controller.py` implements the main control loop but lacks the actual I/O logic for communicating with physical hardware.

Specifically, the methods `_read_state` and `_send_command` raise `NotImplementedError` for any `CommunicationType` other than `SIMULATION` or `LOOPBACK`.

## Impact
- **Blocking**: The control loop cannot function with real robots (ROS2, UDP, EtherCAT). The controller is currently simulation-only.
- **Critical**: This renders the deployment module unusable for its intended purpose (hardware control).

## Technical Details
The following methods require implementation for supported protocols (ROS2, UDP, EtherCAT):

1.  **`_read_state(self) -> RobotState`**:
    *   Must read joint positions, velocities, and torques from the hardware interface.
    *   Must handle timeouts and communication errors gracefully.
    *   Must map hardware units to internal units (radians, Nm).

2.  **`_send_command(self, command: ControlCommand) -> None`**:
    *   Must transmit torque, position, or velocity commands to the hardware.
    *   Must respect safety limits before sending.

## Acceptance Criteria
- [ ] `_read_state` implemented for at least one hardware protocol (e.g., UDP or ROS2) to verify the pattern.
- [ ] `_send_command` implemented for the same protocol.
- [ ] Integration tests using a mock hardware interface (e.g., mock UDP server).
- [ ] `NotImplementedError` removed for the implemented protocol.

## References
- File: `src/deployment/realtime/controller.py`
- Related Issue: `ISSUE_REALTIME_CONTROLLER.md` (Covers connectivity `_connect_*`, this issue covers per-cycle I/O).

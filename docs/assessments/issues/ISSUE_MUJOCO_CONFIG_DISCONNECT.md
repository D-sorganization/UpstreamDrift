# Critical: Live Analysis Configuration Not Propagated in MuJoCo

**ID:** ISSUE_MUJOCO_CONFIG_DISCONNECT
**Severity:** Critical
**Status:** Open
**Date:** 2025-01-22

## Description
The MuJoCo Humanoid Launcher's `RemoteRecorder` class (`engines/physics_engines/mujoco/python/humanoid_launcher.py`) has an empty implementation of `set_analysis_config`. This method is called by the `LivePlotWidget` when users adjust analysis settings in the UI. Because `RemoteRecorder` sits in the GUI process and the simulation runs in a separate process (`ProcessWorker`), these settings are never sent to the simulation engine.

## Impact
*   **User Experience:** Users changing analysis parameters (e.g., filtering, metrics) in the "Live Analysis" tab see no effect on the results.
*   **Functionality:** "Live" analysis control is broken.

## Technical Details
*   **File:** `engines/physics_engines/mujoco/python/humanoid_launcher.py`
*   **Method:** `RemoteRecorder.set_analysis_config`
*   **Current Code:** `pass  # Config driven by simulation loop`
*   **Root Cause:** Lack of IPC mechanism to send configuration updates from the GUI process to the simulation process.

## Recommended Fix
1.  Implement a command channel in `ProcessWorker` (e.g., using ZMQ or stdin) to accept configuration updates.
2.  Update `RemoteRecorder.set_analysis_config` to forward the config dict to `HumanoidLauncher.simulation_thread` (the `ProcessWorker`).
3.  Update the simulation loop (`humanoid_golf/sim.py`) to poll for and apply configuration updates.

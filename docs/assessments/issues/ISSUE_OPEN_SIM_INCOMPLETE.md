# Critical: Incomplete OpenSim Simulation Logic

**ID:** ISSUE_OPEN_SIM_INCOMPLETE
**Severity:** Critical
**Status:** Open
**Date:** 2025-01-22

## Description
The OpenSim physics engine GUI (`engines/physics_engines/opensim/python/opensim_gui.py`) contains a `run_simulation` method that attempts to call `self.model.run_simulation()`. This call raises `NotImplementedError`, which is caught and displayed to the user as "OpenSim simulation is not yet fully implemented".

## Impact
*   **User Experience:** Users expecting to run OpenSim simulations are met with an error dialog.
*   **Functionality:** The core purpose of the OpenSim engine integration is unfulfilled.

## Technical Details
*   **File:** `engines/physics_engines/opensim/python/opensim_gui.py`
*   **Line:** ~267 (exception handler)
*   **Root Cause:** The backend method `GolfSwingModel.run_simulation` (in `opensim_golf.core`) is either missing or raises `NotImplementedError`.

## Recommended Fix
1.  Implement `run_simulation` in `engines/physics_engines/opensim/python/opensim_golf/core.py`.
2.  Ensure it integrates with the OpenSim API to perform forward dynamics or kinematic analysis.
3.  If implementation is delayed, disable the "Run Simulation" button in the GUI or label it "Coming Soon".

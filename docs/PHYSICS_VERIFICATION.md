# Physics Verification Guide

This document outlines how to verify the accuracy of the physics engines (MuJoCo, Drake, Pinocchio) used in the Golf Modeling Suite.

## Overview

We use an automated verification suite that compares simulation results against analytical baselines to ensure:
1.  **Energy Conservation**: Systems stay within physical energy bounds.
2.  **Accuracy**: Trajectories match closed-form solutions (e.g., pendulums, ballistics).
3.  **Stability**: Long-running simulations do not diverge.

## Running Verification

### 1. Local Execution (Fastest)

If you have the physics engines installed locally (via `pip` or system install), run:

```bash
python scripts/verify_physics.py
```

Arguments:
- None required.

Output:
- Console summary of Engine Status and Test Results.
- Detailed Markdown report at `output/PHYSICS_VERIFICATION_REPORT.md`.

### 2. Docker Execution (Recommended)

To run tests in a guaranteed clean environment with all dependencies (especially for Drake/MuJoCo binaries), use the Docker runner:

```bash
python scripts/run_tests_in_docker.py
```

This script will:
1.  Build the `golf-suite-dev` image using `engines/physics_engines/mujoco/Dockerfile`.
2.  Mount your current codebase into the container.
3.  Run the verification suite.
4.  Save reports to your local `output/` directory (thanks to the volume mount).

## Understanding the Report

The generated `PHYSICS_VERIFICATION_REPORT.md` contains three sections:

1.  **Engine Status**:
    -   **Available**: Engine is ready for simulation.
    -   **Missing Binary/Assets**: Engine is importable but lacks required DLLs or model XMLs.
    -   **Not Installed**: Python package is missing.

2.  **Validation Test Results**:
    -   **PASSED**: Simulation matches analytical baseline within tolerance (<0.1% energy error).r).
    -   **SKIPPED**: Engine not available.
    -   **FAILED**: Physics violation detected. **Critical:** Do not merge code if this occurs.

3.  **Recommendations**:
    -   Actionable steps to fix missing engines or failing tests.

## Adding New Tests

Add new validation logic to `tests/physics_validation/`:
-   `analytical.py`: Add exact math solutions here.
-   `test_*.py`: Add pytest files. Use `Analytical*` classes to verify simulation data.

## Continuous Integration

These tests are automatically run on PRs via GitHub Actions.

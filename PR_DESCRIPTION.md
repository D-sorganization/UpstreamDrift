# Advanced Physics Verification Suite

## Summary
This PR introduces a comprehensive Physics Verification Suite to ensure the accuracy, stability, and energy conservation of the integrated physics engines (MuJoCo, Drake, Pinocchio). It resolves previous failures in pendulum accuracy tests and provides a robust, Docker-based verification workflow.

## Key Changes
- **New Scripts**:
    - `scripts/verify_physics.py`: Automated runner that probes engines, runs validation tests, and generates a markdown report.
    - `scripts/run_tests_in_docker.py`: One-click script to build the dev container and run verification in a clean environment.
- **Test Fixes**:
    - Fixed `test_mujoco_pendulum_accuracy` by correctly accounting for the rotational inertia of the sphere geometry ($I \approx 1.004$) vs point mass ($I=1.0$).
    - Updated `AnalyticalPendulum` to support custom inertia.
- **Engine Probes**:
    - Enhanced `MuJoCoProbe` to detect assets in `myo_sim`.
    - Improved `DrakeProbe` to fail gracefully if `pydrake.multibody` is missing.
- **Documentation**:
    - Added `docs/PHYSICS_VERIFICATION.md` with usage instructions.

## automated Verification
Run the suite locally or via Docker:
```bash
python scripts/run_tests_in_docker.py
```

## Report Output
A detailed report is generated at `output/PHYSICS_VERIFICATION_REPORT.md`.

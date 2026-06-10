# Rust Parity And Wheel Gates

This table is the required map from Python facade to Rust crate, parity test,
and CI wheel import smoke. New Rust-backed facades must add a row here and keep
the CI ratchet in `scripts/ci/check_rust_parity_wheel_gates.py` green.

| Python facade                                    | Rust crate                         | Parity test                                                     | Wheel module             |
| ------------------------------------------------ | ---------------------------------- | --------------------------------------------------------------- | ------------------------ |
| src/shared/python/physics/rust_kernel.py         | rust_core/upstream-physics         | rust_core/upstream-physics/tests/parity_physics.rs              | upstream_physics         |
| src/shared/python/physics/ball_flight_physics.py | rust_core/upstream-physics         | rust_core/upstream-physics/tests/parity_physics.rs              | upstream_physics         |
| src/shared/python/motion_pipeline/preprocessing  | rust_core/upstream-mocap-preproc   | tests/unit/motion_pipeline/preprocessing/test_rust_parity.py    | upstream_mocap_preproc   |
| src/shared/python/motion_pipeline/sources        | rust_core/upstream-mocap-io        | tests/unit/motion_pipeline/sources/test_mocap_io_rust_parity.py | upstream_mocap_io        |
| src/shared/python/biomechanics/rust_muscle.py    | rust_core/upstream-muscle          | rust_core/upstream-muscle/tests/parity_full.rs                  | upstream_muscle          |
| src/shared/python/motion_pipeline/matching       | rust_core/upstream-motion-matching | rust_core/upstream-motion-matching/tests/parity_finite_diff.rs  | upstream_motion_matching |

The Rust quality gate builds the wheels for these crates with Maturin, installs
all built artifacts into the smoke virtual environment, and imports every module
listed above. That catches missing PyO3 module exports and packaging mistakes
before a PR can report a passing Rust backend.

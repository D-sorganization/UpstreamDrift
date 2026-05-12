# ADR-0015: Rust Outer-Loop Driver with Python Callbacks

- Status: Accepted
- Date: 2026-05-12
- Decision Makers: Dieter Olson
- Related Issues/PRs: #5218, #5254 (Slices 1-6)

## Context

The motion matching solvers (Pinocchio Inverse Dynamics, Computed Muscle Control, and MuJoCo Torque) require executing physics engine evaluations over high-frame-rate trajectories (e.g., $N=1000$ frames). Profiling identified the Python interpreter overhead in the outer driver loop—specifically finite-difference calculations, list comprehension staging, per-frame array boundary checking, and GIL context switching—as a major performance bottleneck.

To accelerate the process, we sought to move the loop execution to Rust via PyO3 without imposing a heavy C++ dependencies (like Pinocchio or MuJoCo dev headers) on our Rust build matrix.

## Decision

We introduce an engine-agnostic Rust crate (`upstream_pinocchio_id` / `upstream-motion-matching`) that acts as a **Rust-driven outer loop**. 

The architecture pattern is:
1. Python calls a Rust facade function, passing contiguous `(N, D)` NumPy buffers (`q_all`, `times`) and a `Py<PyAny>` Python callback representing the physics engine's per-frame logic (e.g., `pin.rnea` or `mujoco.mj_inverse`).
2. Rust precomputes `qdot` and `qddot` buffers natively using fast array iteration (when not overridden).
3. Rust loops over the frames, yielding the GIL to invoke the Python callback once per frame using `py.allow_threads` and safely marshalling `(q_row, v_row, a_row)` views.
4. Rust aggregates the scalar/vector callback responses into a contiguous `(N, M)` matrix and returns it to Python.

## Alternatives Considered

1. **Native Rust bindings via `pin-sys` / C++ FFI:** 
   - *Pros*: Completely removes the GIL and Python interpreter from the loop. Maximum performance.
   - *Cons*: Explodes the Rust workspace dependency matrix. Requires compiling large C++ libraries natively for all platforms, severely harming CI/CD reliability and local build times.
2. **Vectorized Python / NumPy without Rust:** 
   - *Pros*: Pure Python implementation.
   - *Cons*: Physics engines (Pinocchio, MuJoCo) do not generally support vectorized `(N, D)` inputs to their inverse dynamics routines. An interpreted loop is unavoidable on the Python side, limiting speedups.

## Consequences

- **Positive:**
  - Amortizes finite-difference calculation and array allocation overhead.
  - Substantial end-to-end performance gains (≥ 3× speedup for N=1000 trajectory vs Python alone).
  - Clean separation of concerns: The Rust crate handles buffer orchestration, while Python handles engine-specific API calls.
  - Zero heavy C++ bindings added to the Rust build system.
- **Negative:**
  - The Python interpreter and GIL must still be acquired per frame to invoke the callback, setting a hard upper bound on performance scalability compared to a pure native solution.
- **Follow-ups:**
  - Adopt this callback-into-Python pattern for future per-frame loops (e.g., Forward Kinematics, scaling loops in `motion_pipeline/preprocessing/`) if they face similar overhead.

## Validation

- Heavy-integration benchmarks (`test_bench_full_pinocchio_rust_driver.py`) are strictly gated in CI.
- Parity tests assert that the Rust-driven solver produces tau output numerically identical (<1e-9 RMSE) to the pure-Python solver.

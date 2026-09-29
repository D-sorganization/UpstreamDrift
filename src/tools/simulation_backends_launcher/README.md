# Simulation Backends

A polished PyQt6 launcher tile for the GPU-ready simulation layer
(`src.shared.python.simulation_backends`). It lets you drive the golf
double-pendulum model through every available physics backend from one
window — no scripting required.

## Why

The simulation backends package exposes three interchangeable physics
engines behind one frozen interface:

- **ode** — the analytical RK4 reference. Always available, CPU only,
  and the ground truth for cross-validation.
- **mujoco** — the MuJoCo CPU backend. Also exposes dynamics primitives
  (mass matrix, bias forces) for an independent derivation of the
  equations of motion.
- **mjwarp** — the MuJoCo Warp GPU backend for massively parallel
  batched rollouts. Gracefully unavailable without CUDA.

This tile turns that library into an interactive comparison bench: edit
the model, run a swing, sweep a parameter, prove two backends agree, and
save the trajectory.

## Run

```bash
python -m src.tools.simulation_backends_launcher
```

Requires the `gui-tools` extra (`pip install upstream-drift[gui-tools]`):
PyQt6 plus the matplotlib QtAgg backend. If those are missing the entry
point writes an install hint to stderr and exits non-zero.

The tile also appears in the unified launcher as **Simulation
Backends** (manifest id `simulation_backends`); the launcher embeds the
`MainWidget` directly.

## Layout

| Group                | What it does                                                                                                                                                                                                            |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Physics backend**  | Pick `ode` / `mujoco` / `mjwarp`. Unavailable backends are annotated (e.g. `mjwarp (GPU not available)`). A read-only capabilities line updates on change.                                                              |
| **Model parameters** | Spin boxes (with SI units) for upper-segment mass [kg], clubhead mass [kg], wrist damping [N·m·s/rad], and swing-plane inclination [deg], plus a gravity on/off checkbox. Initialised from `GolfModelParams.default()`. |
| **Run controls**     | Horizon [steps] and time step [s], and the four action buttons below.                                                                                                                                                   |
| **Output**           | A matplotlib canvas (trajectory / sweep plots), a read-only report pane, and a status line.                                                                                                                             |

## The four actions

1. **Run Rollout** — builds the selected backend from the current
   parameters, integrates a passive (zero-torque) swing from a raised
   initial pose, plots `theta1` / `theta2` versus time, and stores the
   result for export.
2. **Run Parameter Sweep** — varies the clubhead mass across 24 samples
   around the current value, rolls each out passively on a CPU backend,
   and plots a clubhead-speed proxy (`norm(final joint velocity)`)
   versus clubhead mass. A summary is written to the report pane.
3. **Cross-validate vs ODE** — builds the ODE and MuJoCo backends from
   the current parameters and compares their mass matrices and
   integrated trajectories with
   `simulation_backends.validation`. Each `ValidationReport` (max
   absolute error, tolerances, pass/fail) is rendered into the report
   pane. When MuJoCo is not installed an explanatory note is shown
   instead.
4. **Export HDF5…** — writes the last rollout to a versioned HDF5 trace
   via `simulation_backends.trace_io.write_trace`. The button opens a
   save dialog; the underlying `export_trace_to(path)` method raises
   `ValueError` if no rollout has been run yet.

## Design notes

- The action methods (`run_rollout`, `run_sweep`,
  `run_cross_validation`, `export_trace_to`) are synchronous and free of
  modal dialogs so headless tests can call them directly. Rollouts of a
  2-DoF model are sub-millisecond, so no worker thread is needed.
- `_embed_adapter.py` is **PyQt6-free**: it imports neither a Qt binding
  nor `gui.py` at module top level, so the launcher bootstrap can import
  it even where PyQt6 is absent. The widget is constructed lazily inside
  `create_main_widget`.
- The app theme is applied best-effort; a missing theme package is never
  fatal.

## Tests

Headless smoke and contract tests live in
`tests/ui/tools/simulation_backends/`. They run under
`QT_QPA_PLATFORM=offscreen` with the Agg matplotlib backend and skip
cleanly when PyQt6 is unavailable.

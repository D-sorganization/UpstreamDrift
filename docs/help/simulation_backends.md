---
title: Simulation Backends
tile_id: simulation_backends
status: complete
---

# Simulation Backends

## Purpose

This tile is a comparison bench for the suite's interchangeable physics
backends. You edit a small golf double-pendulum model, roll it out on the
backend of your choice, sweep the clubhead mass, prove two backends agree on the
same model, and export the trajectory as HDF5 - all without writing a script.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| Physics backend | `ode` / `mujoco` / `mjwarp` | Unavailable backends are annotated in the combo box (for example `mjwarp (GPU not available)`). `ode` is preselected because it is always available. |
| Upper-segment mass | kg | Spin box, suffix ` kg`, initialised from `GolfModelParams.default()`. |
| Clubhead mass | kg | Spin box, suffix ` kg`, same default source. |
| Wrist damping | N*m*s/rad | Spin box, range 0.0 to 5.0, step 0.01, suffix ` N*m*s/rad`. |
| Swing-plane inclination | deg | Spin box, range -90.0 to +90.0 deg, step 1.0 deg. Tilt from vertical; scales effective gravity. |
| Gravity enabled | boolean | Unchecking gives a conservative, free swing. |
| Horizon | steps | Spin box, range 10 to 5000 steps, default 300 steps. |
| Time step | s | Spin box, range 0.0001 s to 0.1 s, step 0.001 s, default 0.005 s. |
| Initial pose | rad | Not editable. Fixed at `_INITIAL_Q = (1.2, -0.6)` rad with zero initial velocity, so gravity drives a visible swing. |

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| Rollout plot | rad vs s | `theta1 (shoulder)` and `theta2 (wrist)` versus time. |
| Sweep plot | clubhead-speed proxy vs kg | X axis labelled `clubhead mass [kg]`; Y is `norm(final joint velocity)`, a proxy rather than a calibrated clubhead speed. |
| Sweep summary | kg | Report pane names the mass range swept and the mass at which the proxy peaks. |
| Cross-validation report | max absolute error, tolerance, pass/fail | One `ValidationReport` per compared quantity (mass matrices and integrated trajectories). |
| HDF5 trace | file | Versioned trace of the last rollout. |
| Backend capabilities line | read-only text | Updates when the backend selection changes. |
| Status line | text | Progress / error messages. |

## Method

The tile is a front end over `src.shared.python.simulation_backends`, which
exposes three interchangeable engines behind one frozen interface:

- **ode** - the analytical RK4 reference. CPU only, always available, and the
  ground truth for cross-validation.
- **mujoco** - the MuJoCo CPU backend, which also exposes mass matrix and bias
  forces for an independent derivation of the equations of motion.
- **mjwarp** - the MuJoCo Warp GPU backend for batched rollouts; gracefully
  unavailable without CUDA. See
  [ADR-0023: MuJoCo Warp backend](../adr/0023-mujoco-warp-backend.md).

Rollouts and sweeps integrate a passive (zero-torque) swing from the fixed
raised pose. The sweep evaluates 24 clubhead masses (`_SWEEP_SAMPLES`) centred
on the current value, on a CPU backend. Cross-validation calls
`simulation_backends.validation`; HDF5 export calls
`simulation_backends.trace_io.write_trace`.

The GUI lives in `src/tools/simulation_backends_launcher/gui.py`; the entry
point `src/tools/simulation_backends_launcher/__main__.py` only imports it and
prints an install hint on `ImportError`. The tile's own reference document is
`src/tools/simulation_backends_launcher/README.md`.

## Limitations

- **The model is a 2 degree-of-freedom double pendulum, not a golfer.** Only
  four parameters are editable: two masses, wrist damping and plane
  inclination.
- Rollouts are **passive**. There is no torque input, controller or muscle
  model, so every swing is gravity-driven from the same fixed initial pose,
  which itself is not editable from the UI.
- The sweep's Y axis is `norm(final joint velocity)` - a proxy. It is not a
  measured or calibrated clubhead speed in m/s.
- The sweep varies clubhead mass only. No other parameter can be swept, and the
  sample count is fixed at 24.
- Cross-validation compares ODE against MuJoCo. Without MuJoCo installed you
  get an explanatory note rather than a comparison.
- `mjwarp` needs CUDA. Without it the backend appears but is annotated
  unavailable.
- `Export HDF5...` raises `ValueError` if no rollout has been run yet.
- Requires the `gui-tools` extra (`pip install upstream-drift[gui-tools]`):
  PyQt6 plus the matplotlib QtAgg backend. Without them the module entry point
  writes an install hint to stderr and exits non-zero.

## See Also
- [Simulation backends overview](../simulation_backends/README.md)
- [Simulation backends user guide](../simulation_backends/USER_GUIDE.md)
- [Cross-engine comparison reports](../simulation_backends/cross_engine_comparison.md)
- [ADR-0023: MuJoCo Warp backend](../adr/0023-mujoco-warp-backend.md)
- [Cross-Engine Dashboard tile](cross_engine_dashboard.md)
- [Simulation controls](simulation_controls.md)
- [Analysis tools](analysis_tools.md)

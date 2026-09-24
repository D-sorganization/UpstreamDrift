---
title: Cross-Engine Dashboard
tile_id: cross_engine_dashboard
status: complete
---

# Cross-Engine Dashboard

## Purpose

This tile answers one question: which physics engine gives you the most
repeatable answer when the torques are noisy? It runs the same Monte Carlo
perturbation experiment on each engine you tick, then charts a robustness score
(1 - coefficient of variation) per engine and per metric. It is a
reproducibility comparison, not a swing-analysis tool.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| Engines | checkboxes | `mujoco`, `drake`, `pinocchio`, `pendulum_stub` (`ENGINE_NAMES`). `pendulum_stub` is checked by default. At least one must be checked - this is a stated precondition. |
| Trials | count | Monte Carlo trials per engine. Range 1 to 500, default 10. |
| Amplitude | N*m | Standard deviation of the additive Gaussian torque noise. Range 0.0 to 5.0 N*m, default 0.1 N*m. |
| t_end | s | Total simulated duration. Range 0.1 s to 10.0 s, default 1.5 s. |
| dt | s | Integration time step, applied to **all** engines so the comparison is fair. Range 0.001 s to 0.1 s, default 0.01 s. |
| Seed | integer | Not exposed in the GUI. `CrossEngineSimConfig.seed` defaults to 42; trial *i* uses `seed + i`. |
| `--shape-per-engine` | flag | CLI / constructor option giving each engine a distinct marker shape in the trajectory overlay (colour-blind aid). |

Constraints enforced by `CrossEngineSimConfig.__post_init__`: `dt > 0`,
`t_end > 0`, `t_end > dt`, `noise_amplitude >= 0`, `n_trials > 0`.

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| Robustness score chart | dimensionless, `1 - CV` | Per engine, from the mean per-metric CV across trials. |
| CV per metric chart | dimensionless | Cross-engine coefficient of variation for each of the three metrics. |
| Trajectory overlay | rad | Axes labelled `q[0]` and `q[1]`; one marker shape per engine when enabled. |
| `total_energy_final` mean / std | see Limitations | Computed as `0.5 * dot(v_final, v_final)` - a unit-mass kinetic-energy proxy. |
| `end_effector_speed_final` mean / std | rad/s or m/s | The L2 norm of the final velocity vector; which unit applies depends on the engine's velocity convention. |
| `peak_end_effector_speed` mean / std | rad/s or m/s | Same convention caveat. |
| Backend provenance | `real` or `stub_2dof` | Every result declares whether real physics ran or the deterministic 2-DOF stub was substituted. |
| Status label | text | `Ready` / progress / errors. |
| Headless report | log output | `--no-gui` runs the comparison and logs the results. |

Per-engine velocity and unit conventions are recorded in `_ENGINE_CONVENTIONS`
in the module: Drake reports generalized velocity `v` with joint angles in rad,
MuJoCo reports `qvel` in the tangent space with angles in rad, OpenSim reports
coordinate speeds with rotational coordinates in **deg**, and the pendulum stub
reports `qdot`.

## Method

The compute path lives in `src/shared/python/analysis/cross_engine.py` (shared
with the web API), driven by `CrossEnginePerturbationRunner` and
`CrossEngineSimConfig` in
`src/shared/python/pendulum_simulator/cross_engine_perturbation.py`. For each
engine, each trial adds Gaussian torque noise of the configured standard
deviation and integrates to `t_end` at the shared `dt`. Three metrics are
extracted per trial (`METRIC_KEYS`), reduced to mean and standard deviation,
turned into a coefficient of variation, and finally into
`robustness_score = 1 - CV` (`robustness_score`, `per_engine_robustness`,
`cv_values`).

The GUI is `src/launchers/cross_engine_dashboard.py`. All heavy imports (PyQt6,
matplotlib, engine modules) are deferred to runtime. `--no-gui` gives a headless
path for CI, and the GUI path also falls back to headless when PyQt6 is absent.

## Limitations

- **`total_energy_final` is not in joules.** The docstring labels it "(J)", but
  the code computes `0.5 * ||v_final||^2` with no mass term - a unit-mass
  proxy. Treat it as a dimensionless dispersion metric, not an energy.
- **Speed units are engine-dependent.** The docstring itself says "rad/s or m/s
  depending on engine". The dashboard does not normalise them, so comparing
  absolute means across engines is not meaningful; only the CVs are.
- An engine you tick may not actually run. If the real package is unavailable
  the deterministic 2-DOF `StubEngine` is substituted, and the result is tagged
  `stub_2dof`. Check the provenance before believing a robustness number.
- The underlying model is a 2 degree-of-freedom pendulum, not a full golfer.
- The only perturbation is additive Gaussian torque noise. No parameter,
  initial-condition or model-structure perturbations.
- `opensim` and `myosuite` are **not** selectable: `ENGINE_NAMES` is
  `("mujoco", "drake", "pinocchio", "pendulum_stub")`, even though
  `_ENGINE_CONVENTIONS` carries an entry for OpenSim.
- The random seed is fixed at 42 and not editable from the GUI.
- A high robustness score means low sensitivity to torque noise. It is not a
  statement about physical accuracy or agreement with any other engine - for
  that, use the cross-validation in the
  [Simulation Backends tile](simulation_backends.md).

## See Also
- [Simulation Backends tile](simulation_backends.md)
- [Cross-engine comparison reports](../simulation_backends/cross_engine_comparison.md)
- [Engine capabilities matrix](../engines/engine_capabilities.md)
- [Engine support tiers](../engines/support_tiers.md)
- [Engine selection guide](engine_selection.md)
- [Analysis tools](analysis_tools.md)

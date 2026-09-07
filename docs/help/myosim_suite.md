---
title: MyoSuite
tile_id: myosim_suite
status: complete
---

# MyoSuite

## Purpose

The MyoSuite tile is a status and orientation surface for the muscle-actuated
MyoSuite engine. It tells you whether the optional MyoSuite stack is installed
and importable on this machine, and shows the Python snippet for driving the
engine yourself. It does not run a simulation.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| *Probe engine* button | user action | Triggers the lazy import of `MyoSuitePhysicsEngine` and instantiates it once as a smoke test. |

That is the complete input surface of the tile. There are no model, time-step,
duration, activation or environment-id fields; the widget deliberately does not
instantiate the engine on construction (see the module docstring in
`src/engines/physics_engines/myosuite/python/gui.py`).

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| Engine status | text | One of `not probed`, `available`, or `unavailable (<exception text>)`. |
| Programmatic usage snippet | read-only text | Shows `Engine()` / `load_model('myoElbowPose1D6MRandom-v0')` against the `PhysicsEngine` protocol. |
| Intro text | text | One paragraph describing MyoSuite as a Gym-based suite of musculoskeletal environments on MuJoCo. |

No physical quantities are produced by this tile, so there are no units to
report.

## Method

There is no calculation. `MainWidget._on_probe_clicked` performs a lazy
`from .myosuite_physics_engine import MyoSuitePhysicsEngine`, constructs the
engine once, discards the reference, and writes the outcome into a label. Any
exception is caught and rendered as the status string.

`MainWidget` and its `MainWindow` shell live in
`src/engines/physics_engines/myosuite/python/gui.py`. The actual engine is
`MyoSuitePhysicsEngine` in
`src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py`, loaded
by `load_myosim_engine()` in `src/engines/loaders.py` - documented in
[MyoSim engine reference](../engines/myosim.md), which also records that
`EngineType.MYOSIM` and the MyoSuite implementation are the same thing and that
there is no separate half-sarcomere solver in this repository.

## Limitations

- **This tile does not simulate.** It has no viewport, no stepping, no
  recording, no plots and no export. `cleanup()` is a no-op because the widget
  owns no timers or threads.
- Despite the registry declaring `musculoskeletal`, `muscle_control`,
  `neural_activation`, `mass_matrix` and `jacobian` capabilities for
  `myosim_suite` in `src/config/models.yaml`, none of those are reachable from
  this widget. They belong to the engine class, which you must drive from
  Python.
- Muscle-driven rollouts, activation optimisation and RL training all live in
  the MyoSuite Python API and the shared muscle-analysis tooling, not here.
- Requires the optional extra to report `available`:
  `pip install "upstream-drift[biomechanics]"` (or `pip install myosuite>=2.0.0`).
- The probe result is a point-in-time import check. It does not validate that
  any particular MyoSuite environment id can be constructed.

## See Also
- [MyoSim engine reference](../engines/myosim.md)
- [MuJoCo engine reference](../engines/mujoco.md) - MyoSuite is built on MuJoCo
- [Engine capabilities matrix](../engines/engine_capabilities.md)
- [Engine support tiers](../engines/support_tiers.md)
- [Engine selection guide](engine_selection.md)

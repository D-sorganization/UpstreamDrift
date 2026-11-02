# sg_optimizer — shared (headless) library

Engine-agnostic, Qt-free implementation of the Strokes Gained Optimizer.

Epic: [#6269](https://github.com/D-sorganization/UpstreamDrift/issues/6269) · Phase 1: [#6270](https://github.com/D-sorganization/UpstreamDrift/issues/6270)

## Layout

```
sg_optimizer/
├── shot_model/      # TiltedBivariateGaussian, baseline, PlayerProfile, putting
├── course/          # CourseConditions (rough/trees/greens), synthetic rasterizer
├── mdp/             # State, ActionSet, transition sampler, value iteration
├── cli.py           # headless entry-point
```

## Putting Providers

`shot_model/putting_provider.py` makes the putting model selectable (#14177).
`putting_provider()` returns the logistic model, which stays the default until
putting validation closes. `putting_provider("simulation", curve)` serves make
probability and miss leave from a `SimulatedPuttingCurve`, built by
`putting_analytics.sg_curve.simulate_putting_curve` from a skill profile. Both
apply the same green-speed and `PuttingSkill` layer. The simulated curve is not
calibrated against published make rates yet.

## Principles

- **TDD** — every public symbol has at least one unit test before/with implementation.
- **Design by Contract** — invariants enforced via [`src.shared.python.contracts`](../contracts.py) (`require`, `ensure`, `precondition`).
- **Law of Demeter** — UI talks only to top-level facades; never reaches into private internals.
- **DRY** — `PlayerProfile`, `CourseConditions`, `LieRaster` are the single source of truth.

## Pitfall map (spec §6)

| Pitfall                            | Where handled                                                   |
| ---------------------------------- | --------------------------------------------------------------- |
| #1 baseline vs V\*                 | `mdp/value_iteration.py::HoleMDP._initial_value`                |
| #2 coordinate frames               | `mdp/action.py::rotate`, `mdp/transition.py`                    |
| #6 putting separation              | `shot_model/putting.py` and `putting_provider.py` are standalone |
| #11 vectorization correctness      | `bellman_backup_scalar` is paired with `HoleMDP.bellman_backup` |
| #12 starting-lie modifier          | `mdp/transition.py::_condition_modifiers`                       |
| #14 conditions baked into geometry | lie codes are geometric; conditions are configurable            |
| #17 stimp / approach coupling      | `GreenModel.effective_green_depth_multiplier`                   |

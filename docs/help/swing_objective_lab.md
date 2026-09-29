---
title: Swing Objective Lab
tile_id: swing_objective_lab
status: stub
---

# Swing Objective Lab

## Purpose

Ask what a downswing would look like if the golfer were optimising for
something other than raw clubhead speed - centrifugal delivery, Coriolis
coupling, energy transfer, hand-path impulse - and compare those optimised
downswings against each other under one shared effort budget. The comparison
engine is provided by an external repository, so this page documents the
contract this repository holds it to rather than the tile's own controls, which
are not present in this checkout.

## Inputs

From `SwingObjectiveCompareRequest` in `src/api/routes/swing_objectives.py`,
which is the in-repo contract for a comparison run:

| Field                | Unit  | Constraint       | Default |
| -------------------- | ----- | ---------------- | ------- |
| `arm_mass_kg`        | kg    | > 0              | 5.0     |
| `shaft_mass_kg`      | kg    | > 0              | preset  |
| `clubhead_mass_kg`   | kg    | > 0              | preset  |
| `arm_length_m`       | m     | > 0              | 0.65    |
| `club_length_m`      | m     | > 0              | 1.10    |
| `top_arm_angle_rad`  | rad   | -                | 2.618   |
| `top_wrist_cock_rad` | rad   | -                | 1.745   |
| `duration_s`         | s     | 0.05 < d <= 1.5  | 0.28    |
| `hub_torque_nm`      | N*m   | 1.0 < t <= 2000  | 250.0   |
| `wrist_torque_nm`    | N*m   | 0.1 < t <= 500   | 20.0    |
| `node_count`         | count | 5 to 101         | 21      |
| `preset_name`        | -     | optional         | none    |
| `objective_keys`     | -     | optional, min 2  | all     |

`arm_length_m` is hub-to-hands; `club_length_m` is hands-to-head.
`top_arm_angle_rad` and `top_wrist_cock_rad` are the configuration at the top
of the backswing. `duration_s`, `hub_torque_nm`, and `wrist_torque_nm` together
form the shared effort budget every objective must respect, so the comparison
is between mechanisms rather than between efforts. `node_count` is the direct
collocation node count.

## Outputs

From `SwingComparisonResponse` (comparison schema version 1.0.0):

- `objective_keys`: the objectives compared. Six exist; the ones named in the
  in-repo tests are `clubhead_speed`, `centrifugal`, and `hand_path_impulse`.
- `units`: a per-metric unit map. The payload carries its own units, so read
  them from the response rather than assuming any.
- `raw_values`: each objective's metric values, per objective.
- `matrix`: the cross-comparison matrix - each optimised swing scored under
  every objective.
- `torque_saturation`: per-objective saturation series, showing where a
  solution sat on its torque bound.
- `swing_distance`: pairwise distances between the optimised swings.
- `is_degenerate`: set when the compared solutions are not meaningfully
  distinct.
- `diagnostics`: per-objective solver diagnostics.

## Method

Comparison is a direct collocation trajectory optimisation over a
pendulum-class golfer model: for each selected objective, optimise the
downswing under the shared duration and torque budget, then score every
resulting swing under every objective to build the matrix.

The optimiser is **not in this repository**. Both the API route
(`src/api/routes/swing_objectives.py`) and the launcher adapter
(`src/launchers/adapters/swing_objective_lab_embed.py`) resolve
`double_pendulum_golf.swing_objectives` from an external provider - the
`pendulum_simulator` model pack under `../Tools/src/pendulum_simulator/src`,
with `vendor/ud-tools/...` as a fallback and `TOOLS_REPO_ROOT` as an override.
The route calls `so.build_config`, `so.compare_objectives`, and
`so.comparison_to_payload` on that module and returns whatever it produces. The
adapter lives on the consumer side deliberately, so the provider never imports
UpstreamDrift ([ADR-0013](../adr/0013-launcher-composability.md),
[ADR-0004](../adr/0004-launcher-provider-migration.md)).

The underlying model class - the double, triple, and closed-loop golfer
pendulums - is documented on the
[Pendulum Simulator](pendulum_simulator.md) page and in
[../engines/pendulum.md](../engines/pendulum.md).

## Limitations

- **Every swing this tile compares is optimiser output, not measurement.** No
  golfer performed any of these downswings. They are what a reduced pendulum
  model does when a solver is told to maximise a particular quantity.
- The tile is unavailable unless the external provider pack is present. When
  the import fails the API route returns HTTP 503 ("Swing objective
  optimization engine unavailable") and the launcher adapter's widget cannot be
  built.
- Rankings are conditional on the shared budget. Change `duration_s` or either
  torque bound and the ordering between objectives can change; a result is only
  meaningful alongside the budget that produced it.
- Solutions sitting on a torque bound are artefacts of that bound, not
  statements about a mechanism. Read `torque_saturation` before interpreting
  any ranking.
- `is_degenerate` must be checked. When set, the compared swings are not
  distinct enough for the matrix to mean anything.
- No ball, no impact, no ball flight. The comparison ends at swing mechanics;
  it does not produce carry or launch conditions.
- No muscles or metabolic cost. The effort budget is a duration and two torque
  ceilings, nothing physiological.
- Do not assume units. Read the `units` map from the response.

## Unclear

The tile's own user interface could not be inspected, so its controls,
defaults, and on-screen readouts are undocumented here; everything above is
derived from the in-repo API contract and the launcher adapter, not from the
widget.

- The registry gives the path as
  `src/double_pendulum_golf/swing_objectives/__main__.py`. That path does not
  exist in this repository - there is no `src/double_pendulum_golf` directory,
  and nothing matching `double_pendulum_golf/**/swing_objectives/**` anywhere
  in the tree.
- Being a `provider: "tools"` tile, the path is expected to resolve against the
  provider pack root rather than this repo. In the Tools checkout on this
  machine, `src/pendulum_simulator/src/double_pendulum_golf/` exists but
  contains no `swing_objectives/` subpackage, so the widget module the adapter
  imports (`double_pendulum_golf.swing_objectives._embed_adapter.get_dockable_ui`)
  could not be read either.
- The six objective keys are not enumerated anywhere in this repository. Three
  are named in `tests/api/test_routes_swing_objectives.py`
  (`clubhead_speed`, `centrifugal`, `hand_path_impulse`), and the registry
  description additionally mentions Coriolis, energy, and impulse transfer, but
  the exact key strings and their definitions live in the provider.
- Note that `src/shared/python/optimization/_swing_objectives.py` is a
  different, unrelated optimiser with its own objectives
  (`CLUBHEAD_VELOCITY`, `INJURY_RISK`, `ENERGY_EFFICIENCY`). It does not back
  this tile.

Files inspected: `src/api/routes/swing_objectives.py`,
`src/launchers/adapters/swing_objective_lab_embed.py`,
`tests/api/test_routes_swing_objectives.py`,
`src/shared/python/optimization/_swing_objectives.py`,
`src/config/launcher_manifest.json`.

## See Also
- [Pendulum Simulator](pendulum_simulator.md) - the model class being optimised
- [Pendulum Models](../engines/pendulum.md)
- [ADR-0004: Launcher Provider Migration](../adr/0004-launcher-provider-migration.md)
- [ADR-0013: Launcher Composability](../adr/0013-launcher-composability.md)
- [Simulation Controls](simulation_controls.md)

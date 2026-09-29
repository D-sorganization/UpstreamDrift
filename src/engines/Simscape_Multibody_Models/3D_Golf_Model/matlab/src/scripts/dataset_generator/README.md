# Dataset generator (random-sweep) — output schema and Simscape logging

This is the MATLAB side of the random-input swing-dataset pipeline. It runs
`GolfSwing3D_Kinetic.slx` thousands of times under randomized polynomial
torque-coefficient inputs and writes one CSV per trial plus a master CSV
(later compacted to parquet by the Python compactor).

## Simscape full-block logging is pinned on every run

Every generation run pins `SimscapeLogType='all'` so the simlog includes
**every Simscape block's state** (joint angle/rate/accel, constraint
forces, internal moments, ...) without spending Simscape "virtual signal"
markers (which the home license caps aggressively).

This is the home-license workaround documented in
[`setModelParameters.m`](../../functions/dataset_generator/setModelParameters.m)
and baked into the .slx at MDL line 4256 (see
[`mdl_reference/README.md`](../../model/mdl_reference/README.md)). We
re-pin it on every `Simulink.SimulationInput` so swept runs cannot
inadvertently drop it.

The setting is applied in three places, by design:

1. **Inside `setModelParameters.m`** — canonical owner; runs for every
   trial input prep.
2. **Inside `runSimulation.m`** just before `parsim` — defensive re-pin
   on every batch entry.
3. **Inside `runSingleTrial.m`** just before `sim` — defensive re-pin on
   the sequential path.

If you ever see a regression where the master CSV/parquet is back to
~1956 columns with no per-block joint state, check `SimscapeLogType` first.

## Output schema

Per-trial timestep rows include:

- **Bus signals** (no prefix): `q/qd/qdd/tau/r_clubhead/r_grip/...` for
  every joint, exposed via the `CombinedSignalBus` output.
- **Logsout signals** (no prefix): whatever the .slx logs by name.
- **Simscape per-block signals** (prefixed `simlog_*`): every block state
  surfaced by the full-block log, e.g. `simlog_LScap_q`, `simlog_LScap_qd`,
  `simlog_LScap_qdd`. The prefix is the on-disk schema contract --
  Python compactors and ML feature pipelines rely on it to distinguish
  bus columns from per-block log columns. **Do not strip it.**

The `simlog_` prefix is applied idempotently in
[`extractSimscapeDataRecursive.m`](../../functions/dataset_generator/extractSimscapeDataRecursive.m).

## Re-generating the 10k-trial parquet

The 10 000-trial parquet currently on disk was generated **before** these
changes landed and therefore lacks the `simlog_*` columns. It does NOT
need to be regenerated for any current consumer.

If you do choose to regenerate (e.g. with realistic coefficient
distributions) you should expect:

- ~3x the column count (bus columns + logsout + per-block simlog columns
  for every joint subsystem).
- ~30 % larger raw file size on disk vs the existing dump.
- Modestly longer per-trial sim time (Simscape full logging is not free).

Estimate disk before kicking off a sweep: column-count grows roughly
linearly with the number of joint subsystems exposed in `simlog`, and
each adds 3-9 scalar fields per timestep depending on DOFs.

## Running

See `Dataset_GUI.m` for the interactive entry point or
`createSimulationConfig.m` + `runSimulation.m` for the headless API
(used by automated sweeps).

## Tests

Unit tests live in
[`../../../tests/dataset_generator/`](../../../tests/dataset_generator/);
the Simscape-logging contract test is
`test_simscape_full_logging.m`. Tests gate on Simulink/Simscape
availability with `assumeFail`, so CI runners without those toolboxes
skip rather than fail.

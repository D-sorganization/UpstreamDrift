---
title: Dataset Generator
tile_id: dataset_generator
status: complete
---

# Dataset Generator

## Purpose

Dataset Generator produces machine-learning training datasets by running
the loaded physics engine many times over varied initial conditions and
control profiles, recording the kinematics, kinetics, and model
quantities of every run. It can also import an existing swing capture
instead of synthesising one. Use it when you need a corpus to train or
validate a model, not when you want to inspect a single simulation.

## Inputs

`GeneratorConfig`
([`config.py`](../../src/shared/python/data_io/dataset_generator/config.py))
carries the run settings. Units are explicit in its docstring:

| Field | Default | Unit / meaning |
| --- | --- | --- |
| `num_samples` | 100 | count of simulation runs, must be > 0 |
| `duration` | 2.0 | seconds per run, must be > 0 and >= `timestep` |
| `timestep` | 0.002 | seconds, must be > 0 |
| `seed` | 42 | integer RNG seed; same seed reproduces the dataset |
| `vary_initial_positions` | `True` | randomise initial joint positions |
| `vary_initial_velocities` | `False` | randomise initial joint velocities |
| `position_ranges` / `velocity_ranges` | empty | lists of `ParameterRange` |
| `control_profiles` | `[zero]` | list of `ControlProfile` |
| `record_*` flags | see below | which quantities to record |
| `output_fields` | `None` | explicit field allow-list; `None` records all |

`ParameterRange(name, min_val, max_val, distribution, num_points)`
requires `min_val <= max_val` and a `distribution` of `uniform`,
`normal`, or `linspace`. The units are the underlying state's units:
rad for revolute joint positions, rad/s for their velocities.

`ControlProfile(name, profile_type, parameters)` accepts `profile_type`
of `zero`, `constant`, `sinusoidal`, `random`, or `step`. Profile
parameters are read from the `parameters` dict: `magnitude`,
`frequency` in Hz, `amplitude`, `scale`, and `step_time` in seconds.

Recording flags default to on for `record_mass_matrix`,
`record_bias_forces`, `record_gravity`, `record_contact_forces`, and
`record_drift_control`; off for `record_jacobians` and
`record_counterfactuals` (ZTCF / ZVCF).

Swing import (`POST /dataset/import-swing`) reads capture files, and
only from allow-listed roots: `data`, `tests/fixtures`, and
`src/shared/urdf`.

An engine implementing the `PhysicsEngine` protocol must already be
loaded. Every engine-dependent endpoint calls `_require_active_engine`.

## Outputs

A `TrainingDataset` of `SimulationSample` records, exportable in the six
formats that `GET /dataset/export/formats` returns:

| Format | Notes |
| --- | --- |
| `hdf5` | HDF5 hierarchical data; recommended for large datasets |
| `sqlite` | SQLite database, queryable and structured |
| `csv` | one CSV file per sample, human-readable |
| `mat` | MATLAB `.mat`; requires scipy |
| `json` | small datasets and configuration |
| `c3d` | C3D motion-capture format; requires ezc3d |

Exports are written only under the allow-listed output roots `output`
and `data`. The web layer returns a `GenerateResult` of `dataset_id`,
`name`, `rows`, `columns`, and `created_at`. Provenance metadata is
attached to every dataset, and the generator's postconditions require
the data be validated free of NaN and Inf in physics quantities.

## Method

The engine is
[`DatasetGenerator`](../../src/shared/python/data_io/dataset_generator/core.py),
a `_DatasetExportMixin` subclass under Design-by-Contract decorators. It
samples parameter ranges with a seeded `numpy.random.Generator`, applies
a control profile of shape `(n_steps, n_actuators)`, steps the engine,
and records kinematics (q, v, a), kinetics (tau, forces, energies), and
model data (inertia, bias forces, Jacobians). A documented invariant is
that the original engine state is restored after generation.

The tile's own surface is the web page at `/tools/dataset`:
`ui/src/pages/DatasetGenerator.tsx`, driven by
`ui/src/api/useDatasetGenerator.ts`, against
[`src/api/routes/dataset.py`](../../src/api/routes/dataset.py) (router
prefix `/dataset`). That router exposes `POST /generate`,
`POST /import-swing`, `GET /control`, `GET /control/state`,
`POST /control/configure`, `GET /control/strategies`, `GET /features`,
`POST /execute`, `GET /plots/types`, and `GET /export/formats`.

## Limitations

- **The registry gives this tile an empty `path`, and that is
  deliberate.** Its `surface_reason` reads "web page (/tools/dataset);
  the native path is the MATLAB chooser", and its `surfaces` list is
  `["web"]` only. There is no PyQt6 Dataset Generator window; the native
  route goes through `src/launchers/matlab_suite_dialog.py` into the
  Simscape MATLAB scripts, which need a MATLAB licence and are not
  covered by this page.
- Nothing generates without a loaded engine. `POST /generate` and every
  control, feature, and strategy endpoint fail if no engine is active.
  Only `GET /plots/types` and `GET /export/formats` are static.
- Path handling is deliberately restrictive. Captures outside `data`,
  `tests/fixtures`, or `src/shared/urdf` cannot be imported, and exports
  outside `output` or `data` cannot be written; this is a security
  control, not a bug.
- `mat` export needs scipy and `c3d` export needs ezc3d. Neither is
  guaranteed present.
- Sampling is per-run scalar sampling from independent ranges. There is
  no correlated or latin-hypercube design, and no adaptive or
  active-learning sample selection.
- The generator trains nothing. It only produces the corpus; see
  Training Controller for the training side.

## See Also
- [Training Controller](training_controller.md) - consumes generated datasets
- [Data Explorer](data_explorer.md) - inspects generated datasets
- [Analysis tools](analysis_tools.md)
- [Feature parity matrix](../development/feature_parity_matrix.md)

# Random-Sweep Dataset Schema

This document defines the parquet schema for the dataset of randomly-generated swings that Options 2, 3, and 4 will train on.

> **Status: PROPOSED.** The user is copying the dataset into the repo for review. Once it lands, this document will be updated to reflect the actual schema and any deltas will be flagged in Issue #019.

## Source

The dataset is the output of the existing MATLAB dataset generator at [matlab/src/scripts/dataset_generator/runSimulation.m](../../src/scripts/dataset_generator/runSimulation.m). Each "trial" is one forward simulation of `GolfSwing3D_Kinetic.slx` with a randomly sampled coefficient vector.

## Logical structure

The user has stated: **each timestep is treated as a separate sample of (kinematics, torques)**. That means rows are timesteps, not trials. The training input/output for the NN options is constructed from this row-wise format.

```
trial_id   |  t  |  q (joint angles)  |  qd (joint velocities)  |  τ (joint torques)  |  club kinematics
─────────────────────────────────────────────────────────────────────────────────────────────────────
   0       | 0.0 |   [...n_joints]    |     [...n_joints]       |    [...n_joints]    |   [12-vec]
   0       | 1ms |        ...          |          ...            |         ...         |     ...
   ...
   1       | 0.0 |        ...          |          ...            |         ...         |     ...
```

Plus a **trial-level** table with the coefficients used:

```
trial_id   |   coefficients (n_joints × 7 flattened)   |   metadata
```

## Proposed parquet layout

Two files per dataset, joinable on `trial_id`:

### `trials.parquet` (one row per simulation)

| Column                   | Type            | Notes                                                                                          |
| ------------------------ | --------------- | ---------------------------------------------------------------------------------------------- |
| `trial_id`               | `uint32`        | Unique within the dataset                                                                      |
| `coefficients`           | `list<float64>` | Flat vector, length `n_joints × 7`, ordering matches `joint_names` × `[A,B,C,D,E,F,G]`         |
| `joint_names`            | `list<string>`  | Length `n_joints`. Same for every trial in a given dataset, but stored per-row for portability |
| `simulation_time_s`      | `float64`       | Total simulation duration                                                                      |
| `sample_rate_hz`         | `float64`       | Sample rate of the timesteps table                                                             |
| `solver_status`          | `string`        | `"success"` / `"warning"` / `"failed"`                                                         |
| `clubhead_speed_max_mph` | `float64`       | Convenience metric for filtering                                                               |
| `total_work_J`           | `float64`       | Sum over joints, integrated over the trial                                                     |
| `dataset_run_id`         | `string`        | The dated folder name, e.g. `"20251030"`                                                       |
| `seed`                   | `int64`         | RNG seed used to generate this trial's coefficients                                            |

### `timesteps.parquet` (one row per simulation timestep)

| Column       | Type            | Notes                                                   |
| ------------ | --------------- | ------------------------------------------------------- |
| `trial_id`   | `uint32`        | FK to `trials.parquet`                                  |
| `t`          | `float64`       | Time in seconds, monotonic per trial_id                 |
| `q`          | `list<float64>` | Joint angles (rad), length `n_joints`                   |
| `qd`         | `list<float64>` | Joint angular velocities (rad/s), length `n_joints`     |
| `qdd`        | `list<float64>` | Joint angular accelerations (rad/s²), length `n_joints` |
| `tau`        | `list<float64>` | Joint torques (N·m), length `n_joints`                  |
| `r_butt`     | `list<float64>` | (3,) butt position in metres                            |
| `r_clubhead` | `list<float64>` | (3,) clubhead position in metres                        |
| `q_club`     | `list<float64>` | (4,) club orientation quaternion `[w,x,y,z]`            |
| `v_clubhead` | `list<float64>` | (3,) clubhead linear velocity (m/s)                     |
| `omega_club` | `list<float64>` | (3,) club angular velocity (rad/s)                      |

Partition `timesteps.parquet` by `trial_id` (or chunks of `trial_id`) to keep per-trial reads fast.

### Indexing and filters

For training, the typical access pattern is:

- **Surrogate (Option 2):** read `coefficients` from `trials` + the _full_ timestep sequence per trial → `(coeffs, kinematic_trajectory)` pair.
- **Inverse NN (Option 3):** same but with the input/output swapped.
- **Per-timestep matching:** `(q, qd, qdd) → tau` — load `timesteps.parquet` only, no join needed. This is the user's stated framing.

## Loader interface

Issue #019 implements the loader. Reference signature:

```python
@dataclass(frozen=True)
class SweepDataset:
    trials: pd.DataFrame       # one row per trial
    timesteps: pd.DataFrame    # one row per timestep, joinable on trial_id
    joint_names: list[str]
    schema_version: str

@precondition(lambda path: path.exists(), "dataset folder must exist")
@postcondition(lambda d: len(d.trials) > 0, "dataset must contain at least one trial")
def load_sweep_dataset(path: Path, *, lazy: bool = True) -> SweepDataset:
    """Load the random-sweep parquet dataset.

    Args:
        path: Folder containing trials.parquet and timesteps.parquet (or a glob).
        lazy: If True, return polars LazyFrames instead of pandas DataFrames.

    Returns:
        SweepDataset with cross-validated schema.
    """
```

A MATLAB equivalent (`load_sweep_dataset.m`) reads via the parquet support introduced in R2019a (`parquetread`), or via `pyrunfile` to call the Python loader. Issue #019 picks the path.

## Validation rules

The loader checks every dataset on load and rejects malformed files loudly:

1. `trials.trial_id` are unique.
2. Every `timesteps.trial_id` exists in `trials.trial_id`.
3. `timesteps.t` is monotonic non-decreasing within each trial; first value is 0; last value is `≈ simulation_time_s`.
4. All `list<float64>` columns have the documented length (e.g., `q` has length `n_joints` for every row).
5. No NaN/Inf in any numeric column except where `solver_status != "success"` (in which case the trial is excluded from training).
6. Coordinate-system spot check: `‖r_clubhead - r_butt‖` ≈ shaft length (≈ 1.1 m for driver) for every timestep — gross deviations indicate a units bug.

## Open questions for the user

- Confirm parquet vs HDF5: the issue backlog item [001_dataset_generator_neural_network.md](../../../../../../../docs/issues/backlog/001_dataset_generator_neural_network.md) mentions HDF5 + SQLite. Parquet is preferred for column-store training pipelines; let me know if the existing artefact is HDF5 instead.
- Confirm the joint set and ordering. The proposal here matches what [getPolynomialParameterInfo.m](../../src/functions/dataset_generator/getPolynomialParameterInfo.m) emits at runtime. If the dataset was generated with a different model state, we may need a `joint_names` column on every row (already in the schema) or a separate manifest.
- Provenance: did the generator capture `git_commit` of the repo at generation time? If not, plan to add it to the loader output via metadata enrichment.
- Storage budget: how many trials per dataset run? `20251030` is the most recent.

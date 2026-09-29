# Compact Swing Dataset — canonical training schema

The raw Simscape dump (`Repositories/data/TenThousandFiles.parquet`, 9.1 GB,
1956 string-encoded columns × 310 000 rows × 10 000 trials × 31 timesteps)
is too large and too wide to feed directly into PyTorch. This document
defines the **compact** parquet schema produced by
`scripts/compact_swing_dataset.py` — a much smaller artefact that retains
exactly the signals the motion-matching surrogate / inverse models need.

> **Status: AUTHORITATIVE.** All Option-2 / Option-3 / leaderboard work
> consumes this schema, not the raw dump. The raw dump is treated as
> immutable input that we never want to regenerate.

## Source

`Repositories/data/TenThousandFiles.parquet` (gitignored; lives outside
the repo). 1 row group per trial (10 000), 31 rows per trial. Every
column is `string` — they are scientific-notation floats stored as
strings, which is why the file is so large (~9 GB on disk vs ~2.5 GB
serialized, vs the ~500 MB this compact format produces).

## Logical structure

Two parquet files joinable on `trial_id`. Per the user's stated framing:
**each timestep is treated as a separate sample of (kinematics, torques)**,
so `timesteps.parquet` is the row-wise training matrix; `trials.parquet`
is small and exists to support trial-level conditioning (e.g. the input
coefficients).

## `trials.parquet` (one row per simulation, ~10 000 rows)

| Column                   | Type                 | Notes                                                                                               |
| ------------------------ | -------------------- | --------------------------------------------------------------------------------------------------- |
| `trial_id`               | `uint32`             | Unique within the dataset; matches the raw dump's `trial_id`                                        |
| `coefficients`           | `list<float64>[189]` | Polynomial coefficients flattened in `(joint, letter)` order — see "Joint and coefficient ordering" |
| `joint_names`            | `list<string>[27]`   | Same for every trial; stored per-row for portability                                                |
| `coefficient_letters`    | `list<string>[7]`    | `["A","B","C","D","E","F","G"]`; bounds in PROJECT_SPEC.md                                          |
| `simulation_time_s`      | `float64`            | Total simulation duration (≈ 0.30 s for the 31-sample dump)                                         |
| `sample_rate_hz`         | `float64`            | Sample rate of the timesteps table (≈ 100 Hz for 31 samples × 0.30 s)                               |
| `clubhead_speed_max_mph` | `float64`            | Max of `ClubLogs_CHS__mph_` over the trial — convenience filter                                     |
| `total_work_J`           | `float64`            | Trapezoidal integral of Σⱼ τⱼ · ωⱼ; convenience metric                                              |
| `solver_status`          | `string`             | `"success"`; failed trials filtered out at compaction time                                          |

## `timesteps.parquet` (one row per simulation timestep, ~310 000 rows)

| Column               | Type            | Length | Notes                                                             |
| -------------------- | --------------- | ------ | ----------------------------------------------------------------- |
| `trial_id`           | `uint32`        | scalar | FK to `trials.parquet`                                            |
| `t`                  | `float64`       | scalar | Time in seconds, monotonic per trial_id, starts at 0              |
| `q`                  | `list<float64>` | 27     | Joint positions in canonical joint order (see below)              |
| `qd`                 | `list<float64>` | 27     | Joint velocities, same order                                      |
| `qdd`                | `list<float64>` | 27     | Joint accelerations, same order                                   |
| `tau`                | `list<float64>` | 27     | Applied joint torques (input/Actuator/control torques)            |
| `r_clubhead`         | `list<float64>` | 3      | Clubhead world xyz (m), from `ClubLogs_CHGlobalPosition_*`        |
| `v_clubhead`         | `list<float64>` | 3      | Clubhead world velocity (m/s), from `ClubLogs_CHGlobalVelocity_*` |
| `r_buttend`          | `list<float64>` | 3      | Club tip / butt-end xyz (m), from `ClubLogs_TipPosition_*`        |
| `r_lhand`            | `list<float64>` | 3      | Left-hand global xyz (m), from `LWLogs_LHGlobalPosition_*`        |
| `r_rhand`            | `list<float64>` | 3      | Right-hand global xyz (m), from `RWLogs_RHGlobalPosition_*`       |
| `r_grip`             | `list<float64>` | 3      | Mid-hands position (computed: `(r_lhand + r_rhand) / 2`)          |
| `clubhead_speed_mph` | `float64`       | scalar | Convenience copy of `ClubLogs_CHS__mph_`                          |

**Storage estimate.** 31 × 10 000 = 310 000 rows × (4 + 8 + 4·27·8 + 6·3·8 + 8) ≈
1.05 KB per row × 310 000 ≈ 326 MB uncompressed; with Snappy compression ≈ 100 MB.
Plus `trials.parquet` ≈ 15 MB. Total ≈ **115 MB** vs **9.1 GB** raw — an 80×
reduction.

## Joint and coefficient ordering

Joints (length 27, fixed across all trials):

```
0:  HipX        7:  LSX        14: RE          21: SpineX
1:  HipY        8:  LSY        15: RF          22: SpineY
2:  HipZ        9:  LSZ        16: RSX         23: Torso
3:  LE          10: LScapX     17: RSY         24: TranslationX
4:  LF          11: LScapY     18: RSZ         25: TranslationY
5:  LWX         12: REAngular  19: RScapX      26: TranslationZ
6:  LWY         13: -          20: RScapY
```

(Exact ordering is documented in `compact_swing_dataset.py::CANONICAL_JOINTS`
and persisted in every `trials.joint_names` for self-describing portability.)

The 7 polynomial coefficients per joint follow the model's existing convention:
`τ_j(t; θ) = A_j·t^6 + B_j·t^5 + C_j·t^4 + D_j·t^3 + E_j·t^2 + F_j·t + G_j`
with bounds `|A,B|≤1000, |C,D|≤500, |E,F|≤100, |G|≤25` (PROJECT_SPEC.md §4).
The flat 189-vec is in `(joint_index × 7) + letter_index` order.

## Raw → compact column mapping

Built by reading the raw parquet's `AngularKinematicsLogs_*` group for q/qd/qdd,
the per-block `ActuatorTorque*` / `TorqueLocal_*` / `*TorqueInput` cols for
applied torques, and the `ClubLogs_CH*` / `LWLogs_LHGlobalPosition_*` /
`RWLogs_RHGlobalPosition_*` cols for hand path. The exact mapping is
defined in `scripts/compact_swing_dataset.py::RAW_COLUMN_MAP` and
unit-tested in `tests/test_compact_swing_dataset.py`.

## Validation rules (enforced by the loader)

The loader (`load_compact_swing_dataset`) checks every dataset on load and
rejects malformed files loudly:

1. `trials.trial_id` are unique.
2. Every `timesteps.trial_id` exists in `trials.trial_id`.
3. `timesteps.t` is monotonic non-decreasing within each trial; first value is 0;
   last value is `≈ simulation_time_s`.
4. All `list<float64>` columns have the documented length (e.g., `q` length == 27).
5. No NaN/Inf in any numeric column.
6. Coordinate-system spot check: `‖r_clubhead - r_grip‖` ≈ shaft length
   (≈ 1.1 m for driver) — gross deviations indicate a units bug in compaction.

## Loader interface

```python
@dataclass(frozen=True)
class CompactSwingDataset:
    trials: pd.DataFrame
    timesteps: pd.DataFrame
    joint_names: list[str]
    coefficient_letters: list[str]   # ["A".."G"]
    schema_version: str               # "compact-1.0"

@precondition(lambda path: path.exists())
@postcondition(lambda d: len(d.trials) > 0)
def load_compact_swing_dataset(path: Path, *, lazy: bool = True) -> CompactSwingDataset:
    """Load the compact swing parquet dataset.

    Args:
        path: Folder containing trials.parquet + timesteps.parquet.
        lazy: If True, return polars LazyFrames; else pandas DataFrames.

    Returns:
        CompactSwingDataset with cross-validated schema.
    """
```

## Why we did this

The raw parquet has every block's bus state — masses, COMs, constraint
forces, segment-inertia tensors, etc. — most of which is invariant
across trials or derivable from `q` via FK. For training a model that
maps `(applied torques, current state)` → `(future hand positions /
accelerations)`, none of that bonus state is useful, and including it
would 80× the dataset size. The compact format is the **minimum
sufficient statistic** for the surrogate / inverse training tasks
described in PROJECT_SPEC.md §3 (Options 2 and 3).

If a future model needs additional raw columns, extend
`scripts/compact_swing_dataset.py::RAW_COLUMN_MAP` and bump the
`schema_version`. Don't read from the raw 9 GB file at training time.

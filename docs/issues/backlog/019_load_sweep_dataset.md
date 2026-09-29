# Issue: Implement load_sweep_dataset Parquet Loader (Python Primary, MATLAB Shim)

## Summary

Implement the random-sweep dataset loader: a Python primary (`load_sweep_dataset`)
that reads `trials.parquet` + `timesteps.parquet` per `DATASET_SCHEMA.md`, and a
thin MATLAB shim (`load_sweep_dataset.m`) that calls the Python loader via
`pyrunfile` for cross-language consistency.

## Motivation

See `motion_matching/shared/DATASET_SCHEMA.md`. Options 2, 3, and 4 train on this
dataset; the MATLAB-side option 1 multistart can use it for warm-starts. A single
loader with cross-validated schema prevents quietly-divergent column orderings or
unit conventions between Python and MATLAB.

## Dependencies

None.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\shared\python\motion_matching\dataset.py` (`SweepDataset`, `load_sweep_dataset`, schema validators)
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\load_sweep_dataset.m` (MATLAB shim)
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\test_dataset.py`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_load_sweep_dataset_matlab_shim.m`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\fixtures\synthetic_dataset\trials.parquet` (small fixture, 4 trials)
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\fixtures\synthetic_dataset\timesteps.parquet` (small fixture)

## Public API

Verbatim from `DATASET_SCHEMA.md`:

```python
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

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

MATLAB shim:

```matlab
function dataset = load_sweep_dataset(path, opts)
%LOAD_SWEEP_DATASET  Read the random-sweep parquet dataset.
%   dataset = LOAD_SWEEP_DATASET(PATH, OPTS) calls the Python loader via
%   pyrunfile and converts the result to MATLAB tables.
```

## Required tests (TDD)

Python:

- `test_loader_reads_trials_and_timesteps_into_dataframes`
- `test_loader_validates_trial_id_uniqueness_in_trials_table`
- `test_loader_validates_every_timestep_trial_id_exists_in_trials`
- `test_loader_validates_t_monotonic_per_trial_starts_at_zero`
- `test_loader_validates_list_column_lengths_match_n_joints`
- `test_loader_validates_no_nan_inf_in_successful_trials`
- `test_loader_excludes_trials_with_solver_status_failed_from_training_view`
- `test_loader_validates_clubhead_butt_distance_within_shaft_length_tolerance`
- `test_loader_lazy_mode_returns_polars_lazyframes`
- `test_loader_eager_mode_returns_pandas_dataframes`
- `test_loader_rejects_missing_trials_parquet_with_clear_error`
- `test_loader_rejects_missing_timesteps_parquet_with_clear_error`
- `test_loader_records_schema_version_in_returned_dataset`
- `test_loader_handles_partitioned_timesteps_parquet_by_trial_id`

MATLAB shim:

- `test_matlab_shim_returns_struct_with_trials_and_timesteps_tables`
- `test_matlab_shim_results_match_python_loader_on_same_fixture`
- `test_matlab_shim_handles_missing_python_environment_with_clear_error`

## DbC contract

Preconditions:

- `path` exists.
- Contains `trials.parquet` (or partitioned equivalent) and `timesteps.parquet`.

Postconditions (per `DATASET_SCHEMA.md` §"Validation rules"):

- `trials.trial_id` are unique.
- Every `timesteps.trial_id` exists in `trials.trial_id`.
- `timesteps.t` is monotonic non-decreasing per trial; first value is 0;
  last value is `≈ simulation_time_s`.
- All `list<float64>` columns have the documented length.
- No NaN/Inf where `solver_status == "success"`.
- `‖r_clubhead - r_butt‖` within plausible shaft-length range for every timestep.

## Acceptance Criteria

- [ ] Python loader reads schema-valid datasets and rejects malformed ones loudly.
- [ ] MATLAB shim returns numerically equivalent data on the same fixture.
- [ ] All listed tests pass.
- [ ] Synthetic fixture under `tests/motion_matching/fixtures/synthetic_dataset/`
      committed (small, `<100 KB` total).
- [ ] DbC decorators applied; postconditions enforced.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `shared`, `python`, `matlab`, `tdd`, `dbc`, `infra`

## Effort estimate

M (1-3 days). Schema validation is the bulk of the work; the shim is small.

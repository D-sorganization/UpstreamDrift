# Data Dictionary

## Common Rules

All physical values use SI units unless a field name explicitly ends in
`_deg`. Time arrays are seconds and monotonically increasing. Forces act in the
direction declared by the evidence interface; couples and moments use the
right-hand rule. Every model result is conditional on its named tier and may
not be promoted to a human or physiological result.

The artifact index table lives in [`DATA_DICTIONARY_ARTIFACTS_CORE.md`](DATA_DICTIONARY_ARTIFACTS_CORE.md) and [`DATA_DICTIONARY_ARTIFACTS_EXTENDED.md`](DATA_DICTIONARY_ARTIFACTS_EXTENDED.md) so this overview stays within the documentation size budget.

## Recurring Field Families

- `schema_version`, `study_id`, and `model_tier` identify the contract and
  evidential scope.
- `source_sha256`, `model_hash`, and `array_artifact` connect summaries to
  executable sources and dense arrays.
- `time_s`, `q`, `qd`, and `qdd` are time, generalized position, velocity, and
  acceleration.
- `force_*_n`, `wrench_*`, and `couple_*_nm` are forces, six-dimensional
  wrenches, and scalar couples at named references.
- `power_*_w`, `work_*_j`, and `energy_*_j` are instantaneous power, integrated
  work, and stored/mechanical energy.
- `constraint_*`, `kkt_*`, `power_residual_*`, and `work_energy_residual_*`
  quantify numerical closure rather than physical effects.
- `training`, `held_out`, `q05`, `median`, `q95`, `q10`, and `q90` identify
  declared ensemble partitions or quantiles; held-out model cases are not human
  validation.

NPZ member names and shapes are inventoried programmatically by the release
manifest and preserved exactly by the checksum file. Reviewers should load NPZ
files with `allow_pickle=False`.

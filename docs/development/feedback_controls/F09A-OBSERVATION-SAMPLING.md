# F09A Native Output Sampling on the Observation Clock

## Scope

This F09 readiness slice provides a positions-only boundary between the
irregular or engine-selected samples emitted by a native replay and the exact
clock retained by measured observations. It enables existing replay metrics to
compare arrays with different native output and observation grids. It does not
run an engine, interpolate generalized state, score physical acceptance, add
tolerances, or qualify any model or capture.

## Reused Interfaces and Limits

`motion_matching.replay_metrics.compute_replay_five_metrics` remains the
authoritative marker metric implementation. Its current array contract requires
prediction and target arrays to have the same frame count, and it uses the
supplied `time_s` for the early window. The new
`align_native_positions_to_observations` boundary samples native 3-D marker
positions onto the observation clock before passing its result to that metric.
The output retains both source clocks and separate SHA-256 identities for the
native output grid, exact observation grid, native output payload, measured
observations, and source replay identity.

`align_to_simulation_grid` solves the reverse operation for loader targets and
also resamples quaternion orientation. It is not used for acceptance scoring:
its `interp_xyz_series` helper delegates to `numpy.interp`, which clamps to
endpoint values outside its source interval. F09A requires no extrapolation
and must not introduce Euclidean interpolation over engine state or
quaternions.

## Contract

- Both clocks are finite, one-dimensional, strictly increasing, and contain at
  least two samples. The observation interval must lie inside the native output
  interval; otherwise alignment fails rather than endpoint-clamping.
- Native output and observations declare the same frame and timebase. Their
  ordered, unique marker labels must match exactly.
- The interpolation identifier is frozen as `linear_position/v1` and accepts
  only arrays shaped `(time, marker, xyz)`. It cannot accept q/v state, unit
  quaternions, or arbitrary channels.
- Every native output marker position and every valid observation position
  must be finite. Missing observations remain represented by the existing
  boolean validity mask; invalid target positions may remain non-finite and are
  excluded by the canonical metric function.
- The returned observation clock is an immutable copy of the exact supplied
  values. Its content digest is distinct from the native output clock digest.
  Input-policy time-grid hashes remain a separate replay identity and are not
  substituted for either output/observation clock.
- The result records path-free source, sampled-output and observation content
  identities. Sampling identity changes when its interpolation method,
  source replay identity, output, observation, frame, timebase, marker order or
  either clock changes.

## Focused Regression Cases

The synthetic tests use an analytic linear marker trajectory to prove values
sample correctly at different clocks. They also cover no-extrapolation at
either endpoint, non-increasing clocks, frame/timebase and marker-order
mismatch, unsupported interpolation, q-like four-value rows, non-finite
predictions, masked missing observations, exact clock retention, and identity
changes. They invoke the existing five-metric calculator to verify scoring is
performed on the observation times and mask.

The tests intentionally define no numeric acceptance limits and use no private
capture material. Native replay consumers must supply actual marker-position
outputs and a digest for their source replay identity; this helper does not
create or validate that engine evidence.

## Validation

```text
python3 -m pytest tests/unit/motion_matching/test_replay_observation_sampling.py -q
python3 -m ruff check src/shared/python/motion_matching/replay_metrics.py src/shared/python/motion_matching/__init__.py tests/unit/motion_matching/test_replay_observation_sampling.py
python3 -m ruff format --check src/shared/python/motion_matching/replay_metrics.py src/shared/python/motion_matching/__init__.py tests/unit/motion_matching/test_replay_observation_sampling.py
```

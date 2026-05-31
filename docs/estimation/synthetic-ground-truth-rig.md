# Synthetic Ground-Truth Rig

Issue #6790 adds a deterministic substrate for estimator falsification before
real-data fitting. The implementation lives under
`src/shared/python/estimation/` and uses the current CIR contracts from
`motion_pipeline` on `origin/main`.

## Public Surface

- `SyntheticObservationRig` projects known `JointTrajectory` frames through one
  or more calibrated pinhole cameras.
- `ForwardModel` is a small protocol: adapters map one known state frame to
  world-space landmark positions. `SkeletonRigForwardModel` provides a CIR
  fixture implementation; Pinocchio/CanonicalState adapters can plug into the
  same protocol after the dependent CC issues merge.
- `NoiseModel` adds reproducible pixel Gaussian noise.
- `ObservationPolicy` adds dropout and spherical occlusion controls.
- `probe_identifiability` builds a finite-difference stacked observation
  Jacobian and reports the SVD, rank, condition number, and nullspace
  directions.
- `synthetic_fixtures.py` provides reusable two-link fixtures for CC-19/CC-20
  segment-length recovery tests, including an intentionally unobservable
  `mass_scale` parameter.

## Current Constraint

`origin/main` does not yet expose the `CanonicalState` value type named in
CC-17. This slice therefore anchors on `JointTrajectory` and a forward-model
protocol instead of inventing a parallel state schema.

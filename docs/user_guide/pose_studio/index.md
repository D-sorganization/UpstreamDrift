# Pose Studio — User guide

Pose Studio is the interactive cross-engine pose editor (Subtask 5 of
EPIC [#4895](https://github.com/D-sorganization/UpstreamDrift/issues/4895)).
Hand-edit a canonical skeleton pose, see it rendered through any of
the five supported engines (Drake, MuJoCo, Pinocchio, OpenSim,
Simscape) without restarting the process, and save the result as an
engine-native starting-state file that `fit_swing_full_pipeline.m`
and other downstream consumers ingest directly.

## Pages

- [Quickstart](quickstart.md) — end-user walkthrough: launch, pick
  an engine, edit a pose, save as a starting state, feed it into the
  MATLAB pipeline.
- [Cross-engine conventions](cross_engine_conventions.md) — the
  per-engine convention table (units, pelvis layout, quaternion
  order, sign-flip gotchas, OpenSim XYZ Euler, Simscape torso-twist,
  Wiffle xlsx CM-vs-inches).
- [Save formats](save_formats.md) — the on-disk shape of every
  supported save format with a downstream-reader code example for
  each (Drake pickle, MuJoCo JSON, Pinocchio `.npz`, OpenSim `.sto`,
  Simscape `starting_pose_offsets.json`, motion-matching target JSON,
  reference-pose library).

## Background

- [ADR-0012 — Canonical pose interchange](../../adr/0012-canonical-pose-interchange.md)
  — design rationale for the canonical convention.
- `src/shared/python/pose_interchange/` — the foundation library
  (`CanonicalPose`, `PoseConventionAdapter`, `LiveKinematicsService`,
  `pose_io`).
- `src/tools/pose_studio/` — the desktop tool itself (PyQt6 GUI,
  PyQt-free `core.py` + `controllers/` for headless tests).

## Related work in EPIC #4895

| Subtask | Status  | Issue                                                                 |
| ------- | ------- | --------------------------------------------------------------------- |
| 1       | merged  | [#4896](https://github.com/D-sorganization/UpstreamDrift/issues/4896) |
| 2       | merged  | [#4897](https://github.com/D-sorganization/UpstreamDrift/issues/4897) |
| 3       | merged  | [#4898](https://github.com/D-sorganization/UpstreamDrift/issues/4898) |
| 5       | merged  | [#4899](https://github.com/D-sorganization/UpstreamDrift/issues/4899) |
| 6       | merged  | [#4900](https://github.com/D-sorganization/UpstreamDrift/issues/4900) |
| 7       | pending | [#4901](https://github.com/D-sorganization/UpstreamDrift/issues/4901) |
| 8       | this PR | [#4902](https://github.com/D-sorganization/UpstreamDrift/issues/4902) |

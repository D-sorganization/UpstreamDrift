# ADR-0053: OpenCap Sidecar, Licence, and Privacy Boundary

- Status: Proposed
- Date: 2026-10-03
- Decision Makers: repository owner (acceptance pending); proposed in the #11400 planning session
- Related Issues/PRs: epic #11400, children #11401–#11409; #6787 (CC-14 OpenCap importer); #11169; #9627; ADR-0041

## Context

OpenCap (Stanford NMBL) produces OpenSim models and kinematics from two or
more synchronized phone videos. Its pipeline detects 2-D keypoints, triangulates
them, runs an LSTM "marker augmenter" that maps about 20 keypoints to 43
anatomical markers of its LaiUhlrich2022 model, then scales that model and runs
inverse kinematics. `opencap-core` (processing) and `opencap-processing`
(session download, analysis) are Apache-2.0.

UpstreamDrift already imports OpenCap marker files (#6787). Epic #11400 extends
that to the scaled model and kinematics, and plans to reuse the augmenter and
optionally run OpenCap locally. Three boundaries need settling first.

1. **Ownership.** ADR-0041 gives Tools authority over calibration and
   reconstruction records. OpenCap has its own calibration and triangulation.
2. **Licensing.** OpenCap's default 2-D detector is OpenPose, whose licence is
   non-commercial. The HRNet/mmpose detector OpenCap also supports is
   Apache-2.0. The augmenter needs TensorFlow.
3. **Privacy.** The hosted OpenCap service uploads raw video to a third-party
   server.

## Decision

1. **Files and sidecars only.** UpstreamDrift consumes OpenCap through its
   output files (the session layout `opencap-processing` writes) or by running
   a separately installed `opencap-core` as a subprocess or container. OpenCap
   source is never vendored, imported, linked, or bundled, and neither
   TensorFlow nor OpenCap is a core dependency. This mirrors the FreeMoCap
   sidecar rule of ADR-0041.
2. **Commercial default detector.** Any UpstreamDrift-launched OpenCap run
   uses HRNet/mmpose or a detector from UpstreamDrift's own pose registry by
   default. OpenPose is opt-in only and is labelled non-commercial wherever it
   is offered.
3. **Hosted processing is opt-in.** Downloading from, or uploading to, the
   hosted OpenCap service is off by default, requires recorded consent, and
   keeps the access token in typed settings, never in the repository. Raw
   video retention follows ADR-0041.
4. **One marker vocabulary.** OpenCap observations keep the LaiUhlrich2022
   marker names verbatim (`r.ASIS_study`, `RHJC_study`, ...), defined once in
   `src/shared/python/motion_pipeline/sources/opencap_markers.py`. Legacy
   labels map onto that vocabulary; detector keypoints are never aliased onto
   augmented markers.
5. **Evidence level.** OpenCap kinematics are model-conditioned estimates (a
   learned augmenter followed by IK), not observed 3-D markers. OpenCap's
   calibration and triangulation are provider evidence, not Tools
   reconstruction records; they do not satisfy ADR-0041 reconstruction
   acceptance.
6. **Units come from the model.** OpenSim motion files mix degrees and metres.
   OpenCap kinematics are unit-converted only with the scaled `.osim` model to
   say which coordinates are translations. Without the model they are not
   loaded.

## Alternatives Considered

1. **Vendor `opencap-core`.** Rejected: duplicates Tools-owned reconstruction,
   drags TensorFlow and OpenPose into the product, and forks a moving upstream.
2. **Rename OpenCap markers to the Vicon-style vocabulary.** Rejected: the
   OpenCap scaled model carries the LaiUhlrich2022 names, so IK against it
   would need renaming back, and several augmented markers have no Vicon
   equivalent.
3. **Default to the hosted service for convenience.** Rejected on privacy
   grounds and because results would depend on an external service's
   availability.

## Consequences

- Positive: OpenSim-ready models and kinematics from inexpensive capture,
  with no new core dependency and no change to Tools authority.
- Positive: the augmenter can complement the self-calibrating pipeline (#9627)
  without replacing it.
- Negative: local OpenCap runs need a separate install and a GPU for
  reasonable speed.
- Negative: OpenCap was validated on gait, squats and jumps. Golf accuracy is
  unknown until #11408 measures it against a marker-based reference, which
  needs physical capture.
- Follow-ups: #11404–#11409.

## Validation

- `tests/architecture/test_opencap_boundary.py` pins the statements above,
  checks that no `src/` module imports OpenCap or TensorFlow at module scope,
  and checks the marker vocabulary has a single definition.
- `tests/unit/motion_pipeline/sources/test_opencap_adapter.py` and
  `test_opencap_session.py` exercise the vocabulary and unit rules against
  fixtures built from OpenCap's published layout and marker names.

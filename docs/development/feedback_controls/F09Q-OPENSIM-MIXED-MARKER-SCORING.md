# F09Q OpenSim Mixed Marker Scoring Turnover

Issue #12195 adds an opt-in observation handoff on the #12184 constrained
mixed replay base. `native_mixed_marker_observation.py` accepts the frozen T01
bundle, exact mixed profile, source-bound cold-start declaration, native model
path, explicit ordered frame-local marker bindings, and measured positions.
It performs one fresh native replay, restores each complete named state for
model-owned OpenSim marker FK, and delegates observation-clock alignment and
metrics to the existing shared scorer.

The observer is reused through the structural `NativeNamedStateReplay`
interface; no coordinate-only FK helper, integrator, or scorer was copied.
The exact observation clock and missingness mask remain authoritative, and
the shared aligner rejects extrapolation. Output arrays are bytes-backed and
immutable. The digest records the T01 bundle identities, compiled profile,
native replay states/actuation/constraint audits, executed loaded-model
identity, ordered marker bindings, observer implementation, and alignment.
All exported marker and alignment arrays use bytes-backed read-only storage;
the diagnostic qualification field cannot be overridden through construction
or `dataclasses.replace`.

The returned observation and score are diagnostic records with
`qualification="unqualified"`. They do not represent an authenticated
execution receipt, F01 admission, approved calibration, production anatomy,
physiology/contact evidence, private-capture fit, full-horizon result, or
six-engine parity. Explicit marker paths/offsets are inputs; this change does
not infer them or promote provisional calibration.

Validation in the installed OpenSim 4.6 Python 3.12 environment:

- `test_coupled_mixed_replay_produces_observation_scorer_positions`
- `test_mixed_marker_scorer_rejects_changed_declared_source`
- `test_mixed_marker_scorer_rejects_observations_beyond_replay_horizon`
- Result: all 12 tests in the affected constrained-mixed module passed in
  28.07 s. This includes the integration, source-mismatch, observation-horizon,
  and immutable-output cases. The temporary pytest configuration enabled the
  repository test path; unknown-marker warnings came from that older SDK
  environment's pytest version.
- Two portable contract tests verify that the diagnostic-only qualification
  cannot be overridden through construction or `dataclasses.replace`.

The default interpreter skips OpenSim-native execution when the optional
provider is unavailable; that skip is not native coverage. Canonical chapter:
`manuals/upstreamdrift/chapters/49-native-opensim-mixed-marker-observation.qmd`.
The engineering manual remains at zero registered calculations and
`blocked-inventory-required`.

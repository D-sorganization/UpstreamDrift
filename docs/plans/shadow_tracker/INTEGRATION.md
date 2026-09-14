# Repository Integration Map

## Evidence and Scope

Inspected against `f8daa71aa263a60c54c785b1ac2d4bc060eb2a71`. These are reusable
source boundaries, not verified runtime integrations. Read current public APIs,
providers, consumers, and tests before editing. The agent-context catalog covers
registered components; use source inspection for unregistered ones. Update its
boundary contracts/reviews when implementation changes registered behavior.

Paths below are relative to the repository root.

| Existing Boundary                                                 | Reuse                                                                  | Gap and Required Test                                                                                                     |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `src/shared/python/pose_estimation/observations.py`               | `CameraCalibration`, intrinsics/extrinsics, per-camera observations    | Add silhouette envelope around existing camera types; reject unsupported camera models and preserve image transforms      |
| `src/shared/python/motion_pipeline/contracts.py`                  | Pipeline CIR and stage conventions                                     | Separate camera contract also exists here; explicit conversion and round-trip test, not a third camera algebra            |
| `src/shared/python/pose_interchange/__init__.py`                  | `CanonicalState`, `CanonicalPose`, SE(3) helpers                       | v1 angles are degrees; v2 state uses radians and manifold conventions; test native-to-canonical round trips               |
| `src/shared/python/motion_matching/__init__.py`                   | Targets, validators, `CanonicalFitResult`, control evaluation          | Existing marker RMSE is not pixel error; result adapter must not fill it with invented zeros                              |
| `src/shared/python/motion_matching/full_body_spec.py`             | Validated full-body geometry/inertia and hashes                        | Confirm DOFs, head, limbs, contact and grip can represent the footage                                                     |
| `src/shared/python/motion_matching/full_body_ik.py`               | `solve_full_body_ik_trajectory` and adapter patterns                   | Warm start only; do not convert silhouettes to fake measured 3D targets                                                   |
| `src/shared/python/motion_matching/full_body_forward_dynamics.py` | `simulate_full_body_forward`, `RolloutOptions`, `ForwardRolloutResult` | Current entry needs `TourCapture`, markers and native state; extract a tested rollout boundary without dummy capture data |
| `src/shared/python/motion_matching/contact_law.py`                | Shared contact parameters and audits                                   | Measure friction, penetration and contact transitions under chosen model                                                  |
| `src/shared/python/motion_matching/cross_engine_replay.py`        | `compute_step_size_convergence`, `compare_engine_replays`              | Extend existing audit patterns; preserve model/contact assumptions and distinguish image from marker metrics              |
| `src/shared/python/motion_matching/visual_skeleton.py`            | Shared visual skeleton and model bindings                              | Reuse binding evidence; qualify filled silhouette geometry and head/limb fidelity separately                              |
| `src/shared/python/motion_matching/diagnostics/__init__.py`       | Public FK/reference/overlay helpers                                    | Skeleton renderer is not a calibrated silhouette renderer                                                                 |
| `src/shared/python/body_part_viz/__init__.py`                     | `BodyPartShape`, `ShapeRenderer`, `SegmentVizSet`                      | Reuse shape vocabulary; add calibrated mask output with renderer parity tests                                             |
| `src/shared/python/anthropometrics/`                              | Subject/segment inertial properties                                    | Keep appearance fitting separate; propagate assumed mass and inertia ranges                                               |
| `src/shared/python/humanoid_character_builder/`                   | Existing primitive/mesh construction                                   | Optional mesh licensing and morphology binding need qualification                                                         |
| `src/shared/python/simulation_backends/protocol.py`               | Capability/trace patterns                                              | Existing double-pendulum backends do not establish full-body support                                                      |
| `src/tools/video_analyzer/`                                       | Video-analysis presentation surface                                    | Inspect actual decoder, identity and timeline behavior; extend, do not fork                                               |
| `src/tools/capture_rig/`                                          | Sessions, calibration, comparison and reference views                  | Preserve camera synchronization and transform provenance through import                                                   |
| `src/shared/python/motion_pipeline/`                              | Pipeline service and API orchestration                                 | Add explicit image-evidence stage without breaking mocap consumers                                                        |

## Important Existing Gaps

`simulate_full_body_forward` currently couples rollout to marker-based
`TourCapture` metrics and performs automatic ground-height calibration for a
near-zero plane. A Shadow Tracker adapter must freeze and record the intended
ground calibration, avoid repeated hidden mutation, and separate rollout from
marker scoring with regression tests. Do not pass fabricated markers or reuse
private engine internals in the new façade.

`CanonicalFitResult` has required `final_rmse_m` and marker-oriented fields.
Shadow results need a typed wrapper/extension that preserves pixel metrics and
unknown metric status. Decide the compatibility approach in ST-01; test existing
callers. A silhouette score is not a distance in metres.

Canonical dynamic state uses a quaternion with `nq != nv`, whereas some full-body
adapters use native coordinate arrays. Never infer ordering from vector length
or assume “41 coordinates” means canonical-v2 layout. Explicit mappings must
round-trip state and velocity, with configuration hashes.

## Engine Qualification Matrix

| Engine                         | Initial Role                          | Gate Before Advertising Support                                                                |
| ------------------------------ | ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| MuJoCo                         | First full-body candidate             | Native model geometry, controlled continuous rollout, contact/grip audit, calibrated rendering |
| Pinocchio                      | Dynamics/convention comparison        | Contact and integrator wrapper verification; equivalent model/control mappings                 |
| Drake                          | Independent full-body candidate       | State/velocity convention tests and contact/closure behavior                                   |
| OpenSim                        | Later biomechanical model integration | Proven forward execution and actuator mapping; muscle claims remain separate                   |
| Simscape                       | Existing high-fidelity model pathway  | Explicit MATLAB R2025b, saved model/control provenance, matching and replay acceptance         |
| Analytical/GPU Double Pendulum | Small software fixtures only          | Never label it a full golfer reconstruction                                                    |

Comparisons use the same morphology, control law, time basis and contact
assumptions. Where contact laws differ, document the difference and compare
appropriate metrics rather than requiring bit equality. Preserve all current
tour-average fitting artifacts and their version-specific evidence.

## Product Registration

Do not register a working tile for this planning scaffold. ST-11 must implement
the shared service first. If a separate Shadow Tracker tool is justified by the
UI review, add `__main__.py`, lazy embed adapter, package entry point and
`src/config/models.yaml` together, plus `src/config/feature_parity.json` and
generated parity matrix. Otherwise extend existing tools and register the
new user-facing capability there. Include API schema/version tests, job ownership,
progress, cancellation, saved-session round trip and missing-dependency behavior.

Scientific pathways added later must update the authoritative QMD manual and
calculation registry under design-manual governance; these planning documents
are not a substitute for that qualification.

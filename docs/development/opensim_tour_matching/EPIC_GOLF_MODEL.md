# OpenSim Golf Model Improvement Epic

Epic: #10394, retained and expanded under #10363. Planning/turnover PR: #10393.
Reviewed 2026-09-18. This plan authorizes implementation tasks; it does not
claim that the current anatomy, address or full-swing dynamics are accepted.

## Intended Product

A coherent anatomical golfer holds a visible driver or iron with both hands,
starts in a measured tour-average address pose, and reproduces the capture
through follow-through. The viewer shows the club, skeleton, reference markers
and truthful kinematic/dynamic status. A versioned torque-actuated baseline
remains usable while compatible muscle/tendon variants are developed. The
model declares anatomical omissions rather than advertising every human joint.

## Verified Defects and Open Questions

Source reviewed: Claude recovery ref `handoff/10341-opensim-20260918`, commit
`6914383f5`, with the actual saved Moco model and full IK motion. See the
[inspection receipt](evidence/anatomical_review_20260918/inspection.json).

1. **Missing club rendering is confirmed:** `/bodyset/Club/attached_geometry`
   is empty, although Club mass is 0.32 kg and the right-hand weld exists.
   Adding another physical club body would duplicate the existing equipment.
2. **Scaling is incomplete:** `os3b_scale_and_full_ik.py:apply_segment_scaling`
   edits joint-frame translations only. Arm meshes remain at scale `1 1 1`.
   The scaling receipt gives humerus R=1.4567005142, L=1.2436231531 and radius
   R=1.1641304709, L=1.1547149771. Marker-to-marker proxy lengths are not
   automatically anatomical joint-center lengths. This is a supported cause
   candidate for the observed gaps, not proof of every arm/pose defect.
3. **Address/orientation remains unqualified:** separate camera, proper rigid
   capture registration, anatomical scaling, marker offsets, joint conventions
   and two-hand constraints. Do not rotate a screenshot or stretch the arm to
   conceal an incorrect fit. Current club connection is right-hand-only.
4. **Muscle scope is absent:** 23 bodies, 39 coordinates and 39 coordinate
   actuators, zero muscles. Skull is torso-fixed; individual finger bones are
   visual meshes. Existing muscle CMC scaffolding is a reuse candidate, not
   evidence of a functional full-body golf muscle model.
5. The 1.813888889 s IK animation is kinematic evidence. The saved 0.60 s native
   replay is rejected. OpenSim 4.5 local viewing does not requalify results
   produced with the 4.6 native runtime. Old ControlTower job IDs are snapshots;
   inspect current identity and preserve the owner's running work.

## Work Breakdown and Sequence

| Task                                                                       | Issue  | Execution Tier                           | Depends On                        |
| -------------------------------------------------------------------------- | ------ | ---------------------------------------- | --------------------------------- |
| OG-01: Freeze the Anatomical Baseline and Failure Fixtures                 | #10395 | cheap                                    | Baseline                          |
| OG-02: Make Segment Scaling Anatomically and Physically Consistent         | #10396 | expert-reviewed implementation           | OG-01                             |
| OG-03: Add a Visible Parameterized Golf Club and Grip Frames               | #10397 | cheap with adapter review                | OG-01                             |
| OG-04: Qualify Capture Registration and Golf Camera Views                  | #10398 | cheap with frame review                  | OG-01                             |
| OG-05: Calibrate and Match a Two-Handed Address Pose                       | #10399 | expert-reviewed fit                      | OG-02, OG-03, OG-04               |
| OG-06: Rebuild Full-Swing Tracking from the Qualified Address              | #10400 | expert-reviewed native execution         | OG-05                             |
| OG-07: Introduce Versioned Model Variants and Actuation Capabilities       | #10401 | cheap after interface review             | OG-01                             |
| OG-08: Qualify Muscle and Tendon Extensions Without Replacing the Baseline | #10402 | expert                                   | OG-02, OG-05, OG-07               |
| OG-09: Package a Golf-Like Native Viewer and Release Evidence              | #10403 | cheap integration with reviewed receipts | OG-03, OG-04, OG-05, OG-06, OG-07 |

Start OG-01, then the bounded visible-club fix OG-03. OG-02 and OG-04 supply
reviewed geometry and frame evidence before OG-05 address fitting. OG-07 is
a narrow contract extension that can be developed from the baseline without
waiting for expensive fits. OG-06 consumes the corrected model/address and
coordinates with #10341. OG-09 completes the torque-viewer delivery. OG-08
is a separately reviewed muscle qualification milestone and keeps the epic
open until its declared scope passes. A cheap agent chooses one ready child;
this plan does not ask it to autonomously redesign biomechanics.

## Existing Program Boundaries

- #10339/#10340 own shared OpenSim model/marker conversion; OG-02/03/07 must
  agree their interface and apply fixes to the source generator, not hand-edit
  competing generated models. Preserve the native diagnostic model identity.
- #10341 owns active native Moco fitting; OG-05/06 supply corrected inputs and
  verified source integration. Existing evidence/checkpoints are retained and
  invalidated for changed model hashes unless an explicit mapping is verified.
- #10352 owns contact/closure semantics. A right-hand weld plus an arbitrary
  left-hand weld can overconstrain the system; review constraint rank and
  degrees of freedom before choosing rigid, compliant or measured grip rules.
- #10374 owns dynamic acceptance; #10375 scientific validation; #10376 model
  inventory; #10377 runtimes; #10378 driver/iron full-swing coverage; #10380
  release. This epic adds visual/anatomical and address gates, not new copies
  of those services or relaxed thresholds.
- Compare other engines only on matched capture/time/frame/model semantics.
  Their fitted outputs are diagnostic comparisons, not anatomical ground truth.

## Contracts and Numerical Gates

Use a versioned profile approved before optimization. OG-05 proposes a static
engineering target of valid-marker RMS <=12 mm, maximum <=30 mm and bilateral
grip positional closure <=5 mm. These require expert review and provenance;
they are not established clinical tolerances. Until approved, status is
unverified. Review any change separately with rationale and apply it to all
candidates; never tune a threshold to accept a failed fit. Orientation and
anatomical surface tolerances depend on landmark observability and must be
explicit in that profile. Synthetic rigid-transform recovery uses 1e-8 m.

Report per-marker/per-frame errors and validity coverage, joint limits, grip
position/orientation residuals, foot clearance/contact, segment lengths,
root residuals, input transforms and uncertainties. Missing inputs fail closed.
No valid markers does not mean zero error. Keep the original clock; trimmed
or resampled data records its mapping and cannot masquerade as full coverage.

Geometry-only edits preserve mass, inertia, coordinates, forces and FK.
Physical edits create a new model hash and invalidate native receipts. Club
spec conversion explicitly maps head-origin shared frames to OpenSim's
existing grip-origin frame. Named landmarks, not equal coordinate-array
indices, drive cross-variant comparisons. Serialize initial muscle/tendon
states and actuator meanings when a variant adds internal dynamics.

## TDD, DbC, LoD and DRY

Every child names its RED fixture and acceptance output. Use analytical or
independent native oracles, not tests that repeat the implementation. Tests
must distinguish pure contracts, native model loading/FK, fitting, independent
replay and rendered evidence. Missing native dependencies never count as a
release pass. Record test counts and exact source/model/runtime hashes.

Reuse `src/shared/python/contracts.py` for public pre/postconditions and
invariants. Validate finite numbers, units, proper transforms, identity,
bounds, capability and state/control mappings. Keep optional features typed
as unavailable. SDK access stays inside the existing OpenSim adapters;
shared/UI layers consume protocols/records and never traverse SDK internals.

Reuse TourCapture/TRC readers, shared marker calibration, ClubSpec, existing
opensim_golf FK, pose tools, model registry, receipt schema and launcher
provider. Add small model-variant metadata to that infrastructure instead of
another registry, parser or generic biomechanics framework. Review the
current main/native branch difference before selecting an integration base.

## Reuse Map

- `src/engines/physics_engines/opensim/python/tour_matching/scale.py` and
  `docs/development/opensim_tour_matching/os3b_scale_and_full_ik.py`.
- `src/shared/python/motion_matching/marker_calibration.py` and
  `tour_capture_contract.py`; OpenSim marker_calibration is a re-export.
- `src/shared/python/motion_matching/club_models.py` (`ClubSpec`) and
  `src/engines/physics_engines/opensim/python/opensim_golf/fk.py`.
- Existing `motion_matching/provider.py`, `opensim_physics_engine.py`,
  and the shared pose-interchange boundaries; inspect their public APIs first.
  (`skeleton_extractors/opensim.py` was removed — issue #8866 — as it had no
  callers; future OpenSim pose extraction should use `pose_interchange`.)
- `muscle_analysis.py`, `POST_MVP_MUSCLES.md`, tests/opensim/test_muscle_cmc.py
  and #4296. Historical fixture/license notes require re-verification.

## Delivery Evidence and Turnover

Each child commits its configuration, input hashes, before/after receipt,
RED/GREEN validation, limitations and one next action. Artifacts go beneath
this directory's evidence tree. Never overwrite rejected historical evidence.
The viewer package records canonical camera settings and includes address,
top, observed impact and finish stills plus full motion video. If impact is
not observed, label the event unavailable. Validate visible geometry separately
from numerical marker/physics checks.

Full torque-model completion requires OG-01 through OG-07 and OG-09 with
upstream gates. Full muscle capability additionally requires OG-08 and its
independent validation. A manifest that permits muscles is future readiness,
not a claim that muscle-driven simulation has passed.

## Primary Technical References

OpenSim's [scaling documentation](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089158/How%2BScaling%2BWorks)
explains that consistent scaling includes geometry, joint frames, mass-related
properties and muscle path components; strength needs separate treatment.
Use the installed SDK version's supported APIs, verified by native tests.
The official [Moco inverse example](https://github.com/opensim-org/opensim-core/blob/main/Bindings/Python/examples/Moco/example3DWalking/exampleMocoInverse.py)
illustrates model processing for muscles; it is a reference, not a golf model
or validation of this project's physiology.

## Planning Validation

On source 6914383f5, `python -m pytest tests/opensim/test_segment_scale.py
tests/opensim/test_marker_calibration.py tests/opensim/test_tour_capture_contract.py
-q -o addopts=""` passed 15 tests (five existing deprecation warnings). This
checks existing pure behavior; it does not qualify corrected geometry or
address, and the new RED fixtures remain implementation work. The focused
document title audit passed. GitHub confirms nine native sub-issues.

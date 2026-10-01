# Necromatcher Native Fitting Turnover

## Scope and Current State

Issue #11235 belongs to the integrated Necromatcher epic #11232. The historical
captures are stored and reviewable; image observations do not establish physical
time, scale, joint torques or an identifiable three-dimensional reconstruction.
Continue through actual native fitting, measured residuals, independent replay
and simulation/impact/analysis handoffs. No historical fit is accepted yet.

Owned branch: `feat/necromatcher-native-fit-11235`, based on workspace branch
`feat/necromatcher-workspace-11234`. Lease uses the current Codex session.

## Native Closure Repair

An actual call to the public `MatchingPlant.closure_residuals(q)` failed because
the MuJoCo implementation constructed an IK adapter with an empty attachment
mapping. That adapter rejects empty markers; its inherited position-residual
method is also unimplemented. Closure must be available independently of image
marker observations.

The native model now evaluates a detached world-space grip separation vector
after forward kinematics. The plant validates one finite coordinate per declared
DOF and delegates to this native boundary. Units remain metres; this is a position
residual, not a weld velocity, orientation residual or dynamics qualification.

Five regression cases failed before the repair. Neutral and bent-elbow poses
are compared with independently compiled MuJoCo MJCF and named native grip sites;
invalid size, nonfinite values and matrix-shaped coordinates are rejected. The
test also mutates a returned array and verifies native state remains unchanged.
The broader plant/model run passes 12 tests; one real-Drake test skips because
the installed Drake package is mocked rather than a native runtime.

The actual driver anthropometry specification loads 44 coordinates and model
SHA-256 `da181bae388e8e4ac1e5f905b11d5c80195bbe177bc685cbb6083a5815053ae8`.
Its neutral grip displacement is approximately `[-0.014840, -0.373188, -0.058067]`
metres. This nonzero displacement is reported as measured; no closure success
is inferred. This generic specimen is not a fitted Tiger or Hogan model.

## Reusable Boundaries and Next Steps

- Use public `motion_matching.pipeline.plant.get_plant` and its engine protocol.
  The old motion-pipeline MuJoCo IK backend is retired and raises explicitly.
- Reuse `estimation.project_pinhole` and `reprojection_residual_from_points`
  for image evidence. Preserve missingness and unknown visibility. Camera,
  metric scale, depth and physical clock assumptions must be recorded separately
  from detector observations.
- The existing Shadow Tracker control fitter consumes actual masks. Do not
  create masks from landmarks and call them observed segmentation; #11227 tracks
  the synthetic segmentation fallback.
- Do not use `ModelMatchHandoffCoordinator` fixed fit numbers as evidence.
  The old torque-matching implementation comparing a reference to itself does
  not establish independent forward replay.
- Fit and save an explicitly qualified research hypothesis for each player,
  with original capture/model/code hashes and measured projection/replay errors.
  Validate engine joint order and external resources before importing a native
  version; library storage alone does not perform this validation.
- Source `tiger-usga-range-capture-v1` has 210 observed frames from the official
  USGA 2000 broadcast. `hogan-practice-capture-v1` has 750 original practice
  frames. Both remain image-observation evidence with physical time unknown.
- MATLAB scientific acceptance requires R2025b. Manual governance currently
  reports `blocked-inventory-required`; this repair does not change that status.

## Validation Procedure

Initialize the exact pinned Tools dependency, set repository `src` on
`PYTHONPATH`, and run:

```powershell
python3 -m pytest tests/unit/motion_matching/pipeline/test_mujoco_closure.py tests/unit/motion_matching/pipeline/test_plant_protocol.py tests/unit/motion_matching/test_full_body_mujoco.py -q --no-cov
python3 scripts/ci/check_lod.py src --baseline scripts/ci/lod_baseline.txt
python3 -m scripts.check_design_manual_governance
```

Keep the full goal active until the player fits and downstream handoffs are
implemented and verified. Native closure repair is a prerequisite only.

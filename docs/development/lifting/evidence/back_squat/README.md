# Back-Squat Same-Input Bundle (LIFT-5, #11745)

`back_squat_bundle.npz` is the first lift trajectory packaged as a same-input
`InputBundle` (`src/shared/python/motion_matching/same_input/bundle.py`), so
every engine can replay identical efforts from an identical start state.

## How It Is Built

1. **Planar solution.** The Tools Movement Optimizer
   (`movement_optimizer`, consumed read-only from `vendor/ud-tools/src`) solves
   the `full_squat` exercise (stand, bottom, stand) as a sagittal
   shank/thigh/trunk chain: 80 kg, 1.78 m lifter, 100 kg bar, 3.0 s, 151
   samples.
2. **Retargeting.** `src/shared/python/lifting/sagittal_retarget.py` rotates
   each segment of the full-body model's solved standing pose rigidly about
   the lateral axis until its sagittal angle equals the optimizer's. Ankles
   stay planted; knees, hips, shoulders and the bar landmark follow, keeping
   the model's own segment lengths and left/right offsets.
3. **Full-body IK.** `src/shared/python/lifting/bundles.py` tracks the nine
   landmarks (ankles, knees, hips, shoulders, bar) with the shared MuJoCo
   marker IK (`FullBodyMarkerKinematics`), four foot spheres held flat on the
   ground, arm coordinates held at the seed pose.
4. **Bundle.** A 0.2 s window centred on the squat bottom (lowest bar sample)
   is tracked in MuJoCo by the pipeline computed-torque controller held per
   step (`generate_reference_bundle`), giving 200 steps at 1 ms.

## Provenance in the Manifest

`InputBundle.load(path).provenance` holds the source commit, Tools pin,
optimizer version and parameters, the full planar solution
(`sagittal_solution`), the IK summary (per-landmark maximum residuals,
closure error, foot-sphere heights, options, standing pose), the IK bar path
over the whole lift, the window's bar path (`window_bar_path_m`), the seed
pose record and its SHA-256, and the model limitations.

## Measured IK Residuals

Maximum over all 151 frames, metres, rounded up to 0.1 mm (also asserted by
`tests/unit/lifting/test_lift_bundles.py`):

| Landmark   | Max Residual |
| ---------- | ------------ |
| ankle_l    | 0.0083       |
| ankle_r    | 0.0052       |
| knee_l     | 0.0270       |
| knee_r     | 0.0099       |
| hip_l      | 0.0149       |
| hip_r      | 0.0138       |
| shoulder_l | 0.0097       |
| shoulder_r | 0.0097       |
| bar        | 0.0044       |

Grip-closure error stays below 1.5e-6 m and the lowest foot sphere penetrates
at most 0.54 mm. The largest residuals come from the full-body spec's own
left/right asymmetry (segment lengths, calibrated hip centres), which a
symmetric planar chain cannot reproduce.

## Limitations

- The only spec the same-input harness replays in MuJoCo, Drake and
  Pinocchio is the golf full-body spec, so the lifter keeps its arms and
  dual-grip club; the arms are held fixed, which keeps the grip closure exact.
- There is no barbell body or load in the full-body dynamics. The bar
  landmark is the spec's `Hub` frame (shoulder-girdle centre), matching the
  optimizer's back-squat bar at the shoulder joint. The barbell and hand
  contact arrive with LIFT-4 (#11744).
- Literature comparison of bar path, bar velocity and phase timing is not yet
  recorded: TODO(#11745) add cited reference values as evidence, not tuning
  targets.

## Regenerate

```bash
MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py
```

Generation takes a few minutes (most of it the 200-step MuJoCo tracking run).

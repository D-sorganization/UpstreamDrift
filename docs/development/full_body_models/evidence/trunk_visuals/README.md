# HRG-D3 Trunk Visual Stills (#13174)

Address, top and impact MuJoCo stills of the full-body-v2 pelvis, abdomen and thorax visuals.

Regenerate (needs MuJoCo; use `MUJOCO_GL=osmesa` without EGL):

```bash
MUJOCO_GL=osmesa python3 docs/development/full_body_models/evidence/trunk_visuals/render_trunk_visuals_evidence.py
```

Artifacts:

- `full_body_v2_anthro_driver.json`: the v2 spec. Its trunk is derived from `full_body_v1_source_spec.json`, and it carries the run's hip, ankle, shank and ground calibration from `ground_support/anthro_driver_drake/full_body_spec_hipcal_scaled.json`, which is its `source_spec_sha256` (`f53c9916…`, #15670).
- `driver_v2_bundle.npz`: the same-input bundle (the v1 `anthro_driver_drake` export, trunk-migrated). It is the Model Explorer demo swing.
- `full_body_v1_source_spec.json`: the run's uncalibrated base (`ab724a31…`). That is `full_body_spec_anthro_driver.json` as of `99a4b0c5ac1`, before the neutral-grip work moved its grip frames. The v2 trunk landmarks are derived from it, because L5/S1 placement refuses a hip-calibrated spec without an HRG-C3 pelvis relocation (#14727).
- `render_out/`: six PNGs (face-on and down-the-line) plus `v2_trunk_stills_manifest_capture-A.json`. They were rendered on the earlier spec `4063705f` (see below).
- `receipt.json`: spec sha256, event frames, PNG hashes and the command echo of those stills.

## Calibrated Bundle (#15670)

Spec `4063705f` was the v2 trunk on the uncalibrated base, without the run's hip calibration. Replaying the run's coordinates on that body put the feet 6–10 cm off at the hips and 1.3 cm off at the shanks, with no ground calibration. With zero root efforts the pelvis slid off its contact: it sank to 0.72 m by 0.8 s and to −0.5 m by 1.4 s, so the demo golfer fell over. The regenerated bundle keeps the pelvis within 2.3 cm (height) and 5.9 cm (planar) of the run's target over the whole swing. Both peaks fall at impact or on the last follow-through sample, and the original pipeline run itself reaches 1.4 cm and 3.0 cm. `test_driver_v2_bundle_reproduces.py` pins that.

The stills and `receipt.json` still record `4063705f`. Its trunk visuals are identical to the current spec's; only the lower-body calibration differs. Re-render them with the command above when GL is available.

Stills are visual evidence only, not physics validation.

## Reproducing the Bundle (#14727)

The bundle's `reference_q` is the rollout of its run, not its tracking target. Feeding it back in as the target is a second-generation run that diverges by design. The true inputs are:

1. `python -m scripts.same_input_bundle export --run-dir docs/development/full_body_models/evidence/ground_support/anthro_driver_drake --out v1.npz --duration 1.65`.
2. `TrunkMigrator(run_spec, v2_spec).migrate_coordinates(v1_reference_q, order)`, with the run's `full_body_spec_hipcal_scaled.json`.
3. `generate_reference_bundle(v2_bytes, times, q_v2)`.

`tests/unit/motion_matching/same_input/test_driver_v2_bundle_reproduces.py` pins a 30 ms horizon of this to 1e-11 rad. Exclude the last sample of a truncated horizon from any comparison, because its reference rate changes.

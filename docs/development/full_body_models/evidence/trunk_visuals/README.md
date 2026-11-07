# HRG-D3 Trunk Visual Stills (#13174)

Address, top and impact MuJoCo stills of the full-body-v2 pelvis, abdomen and thorax visuals.

Regenerate (needs MuJoCo; use `MUJOCO_GL=osmesa` without EGL):

```bash
MUJOCO_GL=osmesa python3 docs/development/full_body_models/evidence/trunk_visuals/render_trunk_visuals_evidence.py
```

Artifacts:

- `full_body_v2_anthro_driver.json` — v2 spec derived from `full_body_spec_anthro_driver.json`
- `driver_v2_bundle.npz` — same-input bundle (v1 `anthro_driver_drake` export, trunk-migrated)
- `full_body_v1_source_spec.json` — the v1 source of the v2 spec (`source_spec_sha256` `ab724a31…`): `full_body_spec_anthro_driver.json` as of `99a4b0c5ac1`, before the neutral-grip work moved its grip frames. Committed so the bundle can be regenerated from its true inputs (#14727).
- `render_out/` — six PNGs (face-on and down-the-line) plus `v2_trunk_stills_manifest_capture-A.json`
- `receipt.json` — spec sha256, event frames, PNG hashes, command echo

Stills are visual evidence only, not physics validation.

## Reproducing the Bundle (#14727)

The bundle's `reference_q` is the rollout of its run, not its tracking target. Feeding it back in as the target is a second-generation run that diverges by design. The true inputs are:

1. `python -m scripts.same_input_bundle export --run-dir docs/development/full_body_models/evidence/ground_support/anthro_driver_drake --out v1.npz --duration 1.65`.
2. `TrunkMigrator(v1_source, v2_spec).migrate_coordinates(v1_reference_q, order)`, with the v1 source spec above.
3. `generate_reference_bundle(v2_bytes, times, q_v2)`.

`tests/unit/motion_matching/same_input/test_driver_v2_bundle_reproduces.py` pins a 30 ms horizon of this to 1e-11 rad. The observed difference is 1.4e-13 rad through step 99. Exclude the last sample of a truncated horizon from any comparison, because its reference rate changes.

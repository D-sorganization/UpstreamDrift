# HRG-D3 Trunk Visual Stills (#13174)

Address, top and impact MuJoCo stills of the full-body-v2 pelvis, abdomen and thorax visuals.

Regenerate (needs MuJoCo; use `MUJOCO_GL=osmesa` without EGL):

```bash
MUJOCO_GL=osmesa python3 docs/development/full_body_models/evidence/trunk_visuals/render_trunk_visuals_evidence.py
```

Artifacts:

- `full_body_v2_anthro_driver.json` — v2 spec derived from `full_body_spec_anthro_driver.json`
- `driver_v2_bundle.npz` — same-input bundle (v1 `anthro_driver_drake` export, trunk-migrated)
- `render_out/` — six PNGs (face-on and down-the-line) plus `v2_trunk_stills_manifest_capture-A.json`
- `receipt.json` — spec sha256, event frames, PNG hashes, command echo

Stills are visual evidence only, not physics validation.

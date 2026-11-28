# Structured baseball-cap render evidence

These are neutral-view renders of the new opt-in cap style on an appearance-only
candidate. The profile was rendered on the Tiger Woods fitted reference-skin
physics spec from `assets/player_models/player-models.zip`; the physics spec
SHA-256 is `269930ec2062b6e9c1dc084461394bb6f2bbed0ddca633cd2832ccb658a4859c`.
The cap style is `baseball_cap_structured`, material `cap_white`, brim length
`0.30` head lengths. Existing baseball and historical styles are unchanged.

The face remains a generic CC0 head with the selected profile's existing
stylized settings. This render shows cap geometry only; it does not validate
facial likeness or golfer identity. The full-body phone view uses the same
fitted reference physics spec and does not represent the player's roster height
or mass.

Recreate the appearance-only model from the repository root:

```bash
unzip -p assets/player_models/player-models.zip reference-skins/tiger_woods/model.json > /tmp/tiger-reference-model.json
PYTHONPATH=.:src MUJOCO_GL=egl python scripts/export_player_models.py \
  --player rory_mcilroy --base-spec /tmp/tiger-reference-model.json \
  --cap-style baseball_cap_structured --output /tmp/structured-baseball-cap
PYTHONPATH=.:src MUJOCO_GL=egl python scripts/export_player_models.py \
  --player rory_mcilroy --render-existing --output /tmp/structured-baseball-cap
PYTHONPATH=.:src MUJOCO_GL=egl python docs/development/evidence/appearance/structured_baseball_cap/render_head_views.py \
  --repo . --model-dir /tmp/structured-baseball-cap/rory_mcilroy \
  --output-dir docs/development/evidence/appearance/structured_baseball_cap
```

Desktop full-body views and the phone full-body view come from the public model
preview exporter. `render_head_views.py` frames the same exported head at
azimuth offsets 0 degrees (front), 35 degrees (three-quarter) and 90 degrees
(profile) at desktop and phone viewports. These images are visual evidence, not
measured anatomy or fitted motion.

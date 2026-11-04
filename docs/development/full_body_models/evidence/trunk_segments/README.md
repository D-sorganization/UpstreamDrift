# SHOW-A3 Trunk Segment Before/After Stills (#14552)

BEFORE (`before_v1_rigid_*`) is the v1 driver with one rigid upper trunk. AFTER
(`after_v2_segmented_*`) is the full-body-v2 lumbar plus thoracic trunk derived
from the same driver spec. The "top" pose applies the same 70 degree total trunk
axial turn to both; v2 splits it 17 percent lumbar and 83 percent thoracic (Bruno 2015
regional split, `src/config/spine_coupling_laws.json`). Poses are synthetic
illustrations of the added degree of freedom, not a fitted swing, and not physics
validation.

Regenerate (headless MuJoCo, `MUJOCO_GL=glfw` offscreen on Windows):

```bash
python3 docs/development/full_body_models/evidence/trunk_segments/render_trunk_segments.py
```

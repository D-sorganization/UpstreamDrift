---
issue: 15398
summary: "S4: straight angular-momentum arrows (whole-body + groups, fixed scale) for MuJoCo/MeshCat"
---

## Scale Calibration (B1)

Default `DEFAULT_SCALE_M_PER_NMS = 0.01159` m/(N·m·s) is fixed for the whole swing (not per-frame autoscaled).

Derived 2026-11-21 on `FULL_BODY_GOLF_SWING_XML`: 2000 random states with
`qvel_i ~ N(0, 0.4 rad/s)`; p99 `||H_whole|| = 56.1` N·m·s maps to a
0.65 m arrow (`SCALE_CALIBRATION_*` constants). Arrows shorter than
`DEFAULT_MIN_DRAW_LENGTH_M` (2 mm) are omitted—never lengthened.

Legend text repeats the FULL_BODY p99 reference and notes that spin-arc radii are schematic (rotation sense only).
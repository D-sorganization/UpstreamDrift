# Hand visual backends (epic [#15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150))

Parallel tracks share:

- `build_hand_visual(grip_spec, side, joint_angles)` → meshes in the **grip-hand frame**
- `registry.py` — discover backends; other lanes call `register_backend(...)`
- `scripts/render_hand_backend_comparison.py` — contact sheet + `metrics.json`

## This lane (PR #15146, lofted procedural skin)

**Distinct from** implicit SDF sculpted hand (lane `bc-dd40e13d`): we use **Catmull–Rom tapered tubes** (one continuous finger) plus a **cupped palm grid**, thenar pad, and thumb sweep. `sdf_meshing.py` is shared math for the SDF track only; this backend does not voxel-merge the whole hand.

Adopt **shared renderer** (`bc-59b6630a`) and **curl-to-contact poser** (`bc-c9c36777`) when those modules land on `main`; until then `preview.py` and local render scripts are interim.

Post iteration learnings on #15150 (Refs, not Fixes).
# Hand visual backends (epic [#15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150))

Parallel tracks share:

- The shared contract and registry from ADR-0057 (`types.py`, `registry.py`, conformance test `test_hand_backend_conformance.py`); `procedural_skin` registers through `register()`
- `comparison.py` — harness columns: the capsule baseline (`capsule.py`, not registered) plus every enabled registered backend
- `scripts/render_hand_backend_comparison.py` — contact sheet + `metrics.json`

## This lane (PR #15146, lofted procedural skin)

**Distinct from** implicit SDF sculpted hand (lane `bc-dd40e13d`): we use **Catmull–Rom tapered tubes** (one continuous finger) plus a **cupped palm grid**, thenar pad, and thumb sweep. `sdf_meshing.py` is shared math for the SDF track only; this backend does not voxel-merge the whole hand.

Adopt **shared renderer** (`bc-59b6630a`) and **curl-to-contact poser** (`bc-c9c36777`) when those modules land on `main`; until then `preview.py` and local render scripts are interim.

Post iteration learnings on #15150 (Refs, not Fixes).
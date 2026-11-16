---
issue: 15144
summary: "Calibrate MyoHand anatomical contact solver, bilateral mirroring, skeleton JSON export, and 74% Z-buffer framing"
dl_state: "in_review"
next_step: "Integrate shared posing.py (#15164) and render.py (#15154) when landed while keeping research_only=True until #13763 owner sign-off"
branch: "cursor/myohand-visual-backend-dcca"
---

- Calibrated anatomical contact solver (`contact_solver.py`) using measured zero-pose MuJoCo triad axes so all four fingers and opposable thumb wrap `GripSpec` cylinders within configurable tolerances (`penetration_tolerance_m = 0.002`, `middle_contact_tolerance_m = 0.0105`, `thumb_contact_tolerance_m = 0.0055`).
- Added thumb bone-segment XY chord clearance guard (`_xy_segment_min_r`, `_thumb_pose_cost`, `_relax_thumb_penetration`) across `0°` and `±20°` shaft roll so `firstmc -> proximal_thumb -> distal_thumb` wraps the grip without chord pass-through.
- Unified single left-hand mirror transform + post-solve shaft roll (`myohand.py`, `mujoco_extract.py`) with determinant parity winding flip (`(det < 0) ^ mirror_y`) and cofactor axial transforms in `skeleton_export.py`, keeping `myohand_skeleton_grip_spec_neutral.json` healthy and finite for downstream tracks.
- Upgraded `render_evidence.py` with vectorized `uint64` packed `ss=2` Gouraud Z-buffer rasterization and exposed `render_frame_fill: 0.74` in `hand_backend_myohand.json` / `MyoHandVisualSettings` for 1024 px evidence across `lead`, `trail`, and `both` views (`0°`, `-20°`, `+20°` shaft roll).
- Added TDD suites `test_myohand_contact_and_framing.py` and `test_myohand_cycle2_thumb_cup_framing.py` (`27 passed, 4 warnings` across `tests/unit/model_appearance/hand_backends/`).
- Adopted the shared ADR-0057 hand backend registry (#15197): `myohand` registers with `default_enabled=False` (only when the myo_sim submodule is initialised), `build_hand_visual` follows the contract (one mesh, `grip_radius_m`, `skin_rgb` defaulting to bone colour), the per-bone dict moved to `build_myohand_bone_meshes`, and the branch-local capsule backend, types and registry were dropped (the capsule stays the drawn default outside the registry). MyoHand passes the conformance suite.

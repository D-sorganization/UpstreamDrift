---
issue: 15144
summary: "Calibrate MyoHand anatomical contact solver, bilateral mirroring, skeleton JSON export, and 70% Z-buffer framing"
dl_state: "in_review"
next_step: "Await #13763 license sign-off before enabling allow_research=False; track Cycle 2 feedback under #15150"
branch: "cursor/myohand-visual-backend-dcca"
---

- Calibrated anatomical contact solver (`contact_solver.py`) using measured zero-pose MuJoCo triad axes so all four fingers and opposable thumb wrap `GripSpec` cylinders within configurable tolerances (`penetration_tolerance_m = 0.002`, `middle_contact_tolerance_m = 0.0105`, `thumb_contact_tolerance_m = 0.0055`).
- Unified single left-hand mirror transform + post-solve shaft roll (`myohand.py`, `mujoco_extract.py`) with determinant parity winding flip (`(det < 0) ^ mirror_y`) and cofactor axial transforms in `skeleton_export.py`.
- Upgraded `render_evidence.py` with vectorized `uint64` packed `ss=2` Gouraud Z-buffer rasterization for 1024 px `~70%` grip/club framing across `lead`, `trail`, and `both` views (`0°`, `-20°`, `+20°` shaft roll).
- Added TDD suite `tests/unit/model_appearance/hand_backends/test_myohand_contact_and_framing.py` (5 new tests; 19/19 total backend unit tests passing).

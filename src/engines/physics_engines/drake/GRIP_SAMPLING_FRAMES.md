# Drake Grip and Clubhead Sampling Frames (Issue #13450)

Section 3.6 of [DRAKE_PARITY_SPEC.md](DRAKE_PARITY_SPEC.md).

- **Frames.** `simulate_with_coefficients` and both autodiff fit plants
  sample the grip at the `mid_hands` body origin and the clubhead at the
  `club_head` body origin (`GRIP_FRAME_NAME`, `CLUBHEAD_FRAME_NAME` in
  `motion_matching/simulate.py`, the `SimOptions` defaults). They are the
  same frames Pinocchio samples, on a byte-identical canonical URDF
  (`drake/models/generated/golfer.urdf` equals the Pinocchio copy), so the
  two engines log the same physical points with no adapter.
- **Aliases only.** `_simulate_support._resolve_body_name` maps the legacy
  names to those frames and back: `club_grip`/`grip` ↔ `mid_hands`,
  `clubhead` ↔ `club_head`. An exact name always wins.
- **Fail closed.** The resolver never substitutes a different body. Before
  #13450 a missing `club_grip` silently resolved to `right_hand` (the wrist
  origin, not the grip) and a missing clubhead to the `club_shaft` origin.
  The fit then fit against the wrong point and target synthesis agreed
  with it, so nothing flagged the error. A model without the frames now
  makes the fit raise `ValueError` and `simulate` report partial poses.
- **No new URDF frame.** The canonical golfer already carries `mid_hands`
  and `club_head`. Adding a `club_grip` link would break the byte-identity
  with Pinocchio and the regeneration gate without adding a point.
- **Limitation.** The generator humanoid from `render_urdf_string` (used
  by the native grip-contact adapter, not by `simulate` or the fit) has
  neither frame. Its grip contact points come from the
  `club_lead_grip`/`club_trail_grip` frames added at load. Simulating it
  needs an explicit `grip_body_name`.
- **Reproduce:**
  `python3 -m pytest tests/engines/drake/test_drake_grip_body_resolution_13450.py tests/engines/drake/test_drake_simulate.py`.
  With pydrake installed, load `CANONICAL_URDF` and check
  `_resolve_body_name(plant, SimOptions().grip_body_name) == "mid_hands"`.

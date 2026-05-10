# feat(motion-matcher): integrate body_part_viz into live view controller

Depends on `body_part_viz` package (#1–#8) and the C3D Viewer integration (#9).

## Why

The Motion-Match Preview tile's `LiveViewController` (`src/tools/starting_pose_matcher/live_view_controller.py`) uses its own simple line-based segment rendering. Switch it to `body_part_viz` so the matcher and the C3D Viewer share the same shape stack.

## What

- `LiveViewController.set_target` builds `MatplotlibRenderer` instance + `SegmentVizSet`.
- A new control: "Body skeleton style" — combobox: Lines (default) / Library shapes (richer figure).
- "Library shapes" mode populates the body skeleton with library meshes (head, upper_arm, forearm, etc.) bound by canonical marker pairs.
- All other matcher features unchanged.

## Tests

- Headless: load `data/C3D_TA_Driver.c3d`; switch to Library shapes; scrub timeline; assert the renderer's artist count > 0 and frames update without errors.
- Switch back to Lines; assert the artists swap cleanly.
- Performance: 301-frame target, 26 library shapes; assert ≥ 30 fps scrub.

## Acceptance criteria

- [ ] Body skeleton style combo present and persisted via session schema.
- [ ] Mode switch is non-destructive (loaded target stays).
- [ ] Coverage ≥ 80%.

## Files touched

- Edit: `src/tools/starting_pose_matcher/gui.py` (combo widget)
- Edit: `src/tools/starting_pose_matcher/live_view_controller.py` (renderer swap)
- Edit: `src/tools/starting_pose_matcher/session_schema.py` (persist style)
- New: `tests/unit/tools/starting_pose_matcher/test_body_style_switch.py`

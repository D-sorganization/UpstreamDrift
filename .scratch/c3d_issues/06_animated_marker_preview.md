# feat(starting-pose-matcher): animated full-trajectory marker preview with timeline scrubber

## Why

The starting-pose matcher (`src/tools/starting_pose_matcher/gui.py`) today renders **a single static frame** of mocap markers (the chosen event: address / top / impact / finish) overlaid on the model skeleton. To validate that the model matches the data we need to see the **full motion play**, with markers moving through space, alongside the simulated model trajectory.

This is the visual surface where the user wants to confirm that the C3D body markers and the model agree, and is the primary integration point for the new `BodyTarget`.

## What to build

Extend `gui.py` with an animated 3D view backed by `matplotlib.animation.FuncAnimation` (the existing matplotlib backend; no new GPU stack).

### UI elements (added to the existing right-panel control stack)

- A **timeline slider** (`QSlider`) spanning `0 .. N-1` of the active target's resampled timegrid.
- A **play / pause** button (`QToolButton` with `QStyle.SP_MediaPlay` / `SP_MediaPause` icons).
- A **speed combo** (`QComboBox`: 0.1×, 0.25×, 0.5×, 1×, 2×, 4×).
- A **loop-playback** checkbox (default on).
- A **show-trail** checkbox (default on): when on, draws fading polylines for each marker covering the last N frames (configurable, default 30 frames).
- A **frame-counter label** (`12 / 301`) tied to the slider.

### Rendering layers

Each layer is a separate matplotlib `Artist` (Line3D / Line3DCollection / quiver) with a per-layer visibility checkbox:

| Layer                         | Source                                 | When shown                        |
| ----------------------------- | -------------------------------------- | --------------------------------- |
| Body markers (points)         | `BodyTarget.marker_xyz[t, :, :]`       | when a BodyTarget is loaded       |
| Body skeleton segments        | computed via the body-segments issue   | when a BodyTarget is loaded       |
| Club mid-hands trace          | `ClubTarget.butt[:t+1]` (trail)        | when a ClubTarget is loaded       |
| Clubface trace                | `ClubTarget.clubhead[:t+1]`            | when a ClubTarget is loaded       |
| Clubface frame triad          | `club_quat[t]` × 3 unit vectors        | when a ClubTarget is loaded       |
| Ball impact point             | `BallImpactState.position_at_impact_m` | when a ClubBallTarget is loaded   |
| Model skeleton (current pose) | provider's FK at slider time           | when a model provider is selected |

### Performance budget

Animation must run at ≥30 fps on a typical office laptop with a 301-sample target. Approach:

- Pre-compute all per-frame segment endpoints once after loading.
- Update artist data via `set_data_3d` / `set_segments` rather than re-creating artists each frame.
- Pause the `FuncAnimation` when the window is hidden (use `QShowEvent` / `QHideEvent`).

### State persistence

Add timeline state to the existing session JSON (schema bump in `src/tools/starting_pose_matcher/session_schema.py`):

```json
"playback": { "current_frame": 0, "speed": 1.0, "loop": true, "trail_frames": 30 }
```

Existing v3 sessions still load (treat missing block as defaults).

## Generic naming

UI labels: "Mocap markers", "Body skeleton", "Club mid-hands trace", "Clubface trace", "Clubface triad", "Ball impact". No source names anywhere.

## Acceptance criteria

- [ ] Timeline slider, play/pause, speed combo, loop, show-trail, layer visibility checkboxes all present and functional.
- [ ] Animation runs ≥30 fps with the 301-sample resampled `BodyTarget` from `data/C3D_TA_Driver.c3d`.
- [ ] No animation tearing when toggling layer visibility mid-playback.
- [ ] Session JSON round-trips the playback block.
- [ ] Headless smoke test (matplotlib `Agg` backend) confirms a full play-through completes without errors and the final-frame artist coordinates match the expected ball-impact frame.
- [ ] No print / no TODO without an issue / file-size budget respected (split helpers into `gui_playback.py` if `gui.py` would exceed 1200 lines).

## Out of scope

- WebGL / Tauri rendering: this is matplotlib-only.
- Body skeleton segment connectivity: covered by the body-segments issue.
- Source-toggle UI: covered by the source-toggle issue (this issue assumes the toggles already exist or are stubbed).

## Files touched

- Edit: `src/tools/starting_pose_matcher/gui.py` (or new `gui_playback.py` to stay under file-size budget).
- Edit: `src/tools/starting_pose_matcher/session_schema.py` (schema bump).
- New: `tests/unit/tools/starting_pose_matcher/test_animation.py`.
- Edit: `src/tools/starting_pose_matcher/README.md` to document the timeline UI.

## References

- Existing static skeleton renderer: `src/shared/python/motion_matching/diagnostics/_skeleton_render.py`
- Existing tests pattern: `tests/unit/tools/starting_pose_matcher/`

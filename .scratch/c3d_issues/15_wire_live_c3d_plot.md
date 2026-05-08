# feat(starting-pose-matcher): wire C3D body markers into the live 3D matplotlib view (END-TO-END)

## Why

This is the **headline user-facing milestone**. After a long sequence of plumbing PRs, all the pieces now exist:

- `BodyTarget` (on `main` via #4504): canonical full-body marker dataclass.
- `load_body_target_c3d` (on `main` via #4504): loads `data/C3D_TA_*.c3d` into `BodyTarget`.
- `BodyMarkerLayer`, `BodySkeletonLayer`, `TrailLayer`, `PlaybackController` (on `main` via #4502): matplotlib artists + animation harness.
- `default_body_segments` (PR #4496 open): segment connectivity for the stick figure.
- `DataSourcesPanel` + `MultiSourceTarget` (PR #4505 open): source-toggle UI emitting `targets_changed`.

What is missing is the **glue**: when the user picks a `.c3d` file in the Body source row, the matcher's existing 3D matplotlib axes must show the markers moving through space — not just print confirmation in the log. The user wants to see this working end to end.

## Current state of `src/tools/starting_pose_matcher/gui.py`

- It draws **static** marker positions for a single chosen event frame (Address / Top / Impact / Finish) from a Wiffle xlsx.
- The 3D axes (`self._ax`) are managed by `_setup_axes` and redrawn whenever the user changes event/phase or rigid-transform sliders.
- There is **no** subscription to live timeline scrubbing, and **no** code path that reads a `BodyTarget`/`MultiSourceTarget` and routes its per-frame data to the existing axes.

## What to build

A new orchestration module `src/tools/starting_pose_matcher/live_view_controller.py` with:

```python
class LiveViewController(QObject):
    """Orchestrates rendering of a MultiSourceTarget on the matcher's 3D axes.

    Owns the layer instances (BodyMarkerLayer, BodySkeletonLayer, ClubTraceLayer,
    ClubfaceTriadLayer, BallImpactLayer, TrailLayer), subscribes to:

    - DataSourcesPanel.targets_changed  -> rebuild layers with new target
    - timeline slider                   -> draw_frame(t)
    - layer-visibility checkboxes       -> toggle layer.set_visible(...)
    - rigid-transform sliders           -> apply transform to body+club traces
    """
```

### Behaviour spec

1. **On `targets_changed`**: tear down old layers, build a fresh stack from the new `MultiSourceTarget`. If `target.has_body()` is true, build `BodyMarkerLayer` and `BodySkeletonLayer` (using `default_body_segments(target.body.marker_names)`). If `target.has_club()` is true, build `ClubTraceLayer` (mid-hands + clubface) and `ClubfaceTriadLayer`. If the club is `ClubBallTarget`, also build `BallImpactLayer`. Trail layer applies to whichever traces are visible.

2. **On timeline scrub**: call each layer's `update(frame_idx)`. Re-blit / `draw_idle()` once at the end (not per layer).

3. **On rigid-transform change** (`Tx/Ty/Tz/Rx/Ry/Rz/Scale`): apply the same affine to body markers, body segments, and club traces before passing to layers. Model skeleton continues to use its existing FK path (untouched).

4. **Empty / partial target**: every layer must accept `None`/missing slots without crashing. If the user toggles off Body, hide the body layers and immediately redraw.

5. **Performance**: one redraw per scrub event, not one per layer. Draw via `FigureCanvas.draw_idle()`. Use blitting where the existing matcher already does.

### Wiring into `gui.py`

In `MainWindow.__init__` (or the equivalent): instantiate `LiveViewController(self._ax, self._fig.canvas)` and connect:

- `self._data_sources_panel.targets_changed.connect(self._live_view.set_target)`
- `self._timeline_slider.valueChanged.connect(self._live_view.set_frame)`
- The play/pause/speed/loop wiring goes through the existing `PlaybackController` and ends in `self._live_view.set_frame(t)`.
- Layer-visibility checkboxes call `self._live_view.set_layer_visible(name, on)`.

### Default behaviour for first-time users

When the user clicks "Browse" on the Body source row and picks `data/C3D_TA_Driver.c3d`:

1. Loader returns a `BodyTarget` with 27 markers, 301 samples, impact at frame 250.
2. `LiveViewController.set_target` builds layers, draws frame 0 (address), runs the timeline auto-fit on the 3D axes (`equalize_3d_axes` from existing helpers).
3. User scrubs the timeline → markers visibly move; clubface trace (if also loaded) lights up.
4. Toggling "Show body skeleton" off/on works without re-loading.

## Acceptance criteria

- [ ] `LiveViewController` instantiated in the matcher MainWindow.
- [ ] Selecting a `.c3d` body file via the Data Sources panel renders 27 markers visibly on the 3D axes within 1 s.
- [ ] Timeline slider scrubs the markers through the swing.
- [ ] Toggling layers on/off updates the canvas immediately.
- [ ] Auto-fit re-frames the axes the first time a target is loaded.
- [ ] No regressions in the existing static-event-frame view (Address/Top/Impact/Finish radio still works for the xlsx Wiffle data path).
- [ ] Headless smoke test (`tests/ui/test_live_view_controller.py`, `QT_QPA_PLATFORM=offscreen`, `Agg`):
  - Load a `BodyTarget` from `data/C3D_TA_Driver.c3d`.
  - Set frame to 0, 100, 250 (impact), and the last frame; assert the BodyMarkerLayer's artist data matches `target.body.marker_xyz[t,:,:]` finite mask.
  - Toggle the body-skeleton layer off and assert the segment artist visibility flag.
- [ ] Mypy + ruff + file-size budget clean. If `gui.py` would exceed 1200 lines, the `LiveViewController` and any helpers go into separate files.

## Out of scope

- Comparing the body markers against a model skeleton overlay (the existing FK path stays as-is; visual comparison is the user's eyeball, not a numerical cost yet).
- Cost-function changes that consume the body target (separate downstream issue).
- Moving the matcher off matplotlib onto a GPU surface.

## Files touched

- New: `src/tools/starting_pose_matcher/live_view_controller.py` (≤ 600 lines)
- Edit: `src/tools/starting_pose_matcher/gui.py` (instantiate + connect)
- New: `tests/ui/test_live_view_controller.py`

## Severity / priority

**Top priority**. This is the milestone the user has been waiting for. It cannot land before #4496 (body skeleton) and #4505 (source-toggle UI) merge, but the PR can be staged on top of those branches and rebased on `main` once they merge.

## Sequencing

1. Wait for / push merge of #4496 and #4505.
2. Open this PR off `main` once both have landed.
3. Once merged, the user-facing acceptance test is: open the Motion-Match Preview tile, browse to `data/C3D_TA_Driver.c3d` in the Body row, scrub the timeline, see markers move.

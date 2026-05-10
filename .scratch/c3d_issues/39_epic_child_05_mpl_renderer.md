# feat(body-part-viz): MatplotlibRenderer (3D Poly3DCollection backend)

Depends on contracts (#1) and primitives (#2).

## Why

The C3D Viewer and the Motion-Match Preview matcher both render in matplotlib 3D today. This is the renderer that integrates with both. It's also the headless-test-friendly backend.

## API

### `MatplotlibRenderer` (in `renderers/matplotlib_renderer.py`)

```python
class MatplotlibRenderer:
    """Render body_part_viz shapes onto a matplotlib Axes3D.

    Owns the matplotlib artist instances; updates them in place rather
    than rebuilding on every frame change. Matches the per-frame
    performance budget (60 fps for 26 segments × 200-vertex meshes).
    """

    def __init__(self, ax: Axes3D) -> None: ...

    # ShapeRenderer protocol:
    def add_shape(self, shape, fitted, theme) -> str: ...
    def update_frame(self, handle, frame_idx) -> None: ...
    def set_visible(self, handle, visible) -> None: ...
    def remove(self, handle) -> None: ...

    # Convenience:
    def add_segment_set(
        self,
        segments: Iterable[FittedShape],
        theme_resolver: Callable[[FittedShape], ShapeTheme],
    ) -> list[str]: ...

    def clear(self) -> None: ...
```

## Performance contract

- **One** matplotlib artist (`Poly3DCollection` or `Line3DCollection`) per shape.
- `update_frame` calls `set_verts()` / `set_segments()` on the existing artist; does NOT clear the axes.
- `update_frame` makes ONE call to `canvas.draw_idle()` at the end (or none — the host triggers redraw).
- For a line shape: use `Line3DCollection`. For mesh / cylinder / ellipsoid / capsule / composite: `Poly3DCollection`.

## Tests

`tests/unit/body_part_viz/renderers/test_matplotlib_renderer.py`:
- Headless: `QT_QPA_PLATFORM=offscreen`, matplotlib `Agg` backend.
- Build axes, add 3 cylinders + 1 line shape; assert 4 artists added.
- `update_frame(handle, 5)` updates the artist's verts; `update_frame(handle, 5)` again is idempotent.
- `set_visible(False)` toggles artist's `_visible3d`.
- `remove(handle)` removes the artist; subsequent `update_frame` raises `KeyError`.
- Performance: time 1000 `update_frame` calls; assert mean <= 1 ms (this is loose — the headless Agg path is fast).

## Acceptance criteria

- [ ] `MatplotlibRenderer` implements `ShapeRenderer` (Protocol runtime check).
- [ ] One artist per shape; per-frame update via `set_verts` / `set_segments`.
- [ ] No `ax.clear()` call anywhere.
- [ ] `add_segment_set` convenience is a thin wrapper over `add_shape`.
- [ ] ≥ 90% line coverage.
- [ ] Performance test in CI for ≥ 60 fps target.

## Files touched

- New: `src/shared/python/body_part_viz/renderers/matplotlib_renderer.py`
- Edit: `src/shared/python/body_part_viz/renderers/__init__.py`
- New: `tests/unit/body_part_viz/renderers/test_matplotlib_renderer.py`

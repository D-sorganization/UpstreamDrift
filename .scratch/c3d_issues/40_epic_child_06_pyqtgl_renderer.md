# feat(body-part-viz): PyQtGLRenderer (pyqtgraph.opengl backend) — performant 3D

Depends on contracts (#1) and primitives (#2).

## Why

Matplotlib 3D is fine for ~30 segments at modest mesh resolution but stalls for high-poly meshes or many segments. `pyqtgraph.opengl` (already on the dependency tree as a transitive dep of pyqtgraph) gives us GPU-accelerated rendering for the matcher's 3D view when richer shapes are loaded.

## API

`renderers/pyqtgl_renderer.py`:

```python
class PyQtGLRenderer:
    """GPU-accelerated 3D renderer using pyqtgraph.opengl.

    Use when the matplotlib backend's per-frame update budget is
    exceeded (typically: > 30 segments × > 1000 vertices each).

    Stays optional — the package's __init__ does not import pyqtgraph
    at module load time; callers explicitly construct the renderer.
    """

    def __init__(self, gl_view_widget: pyqtgraph.opengl.GLViewWidget) -> None: ...

    # ShapeRenderer protocol:
    def add_shape(self, shape, fitted, theme) -> str: ...
    def update_frame(self, handle, frame_idx) -> None: ...
    def set_visible(self, handle, visible) -> None: ...
    def remove(self, handle) -> None: ...
```

## Optional dependency handling

- `pyqtgraph` is imported lazily inside `pyqtgl_renderer.py`.
- `body_part_viz/__init__.py` does NOT import this module at package load.
- A `setup.cfg` / `pyproject.toml` extra `body-part-viz-gl` lists pyqtgraph + PyOpenGL.

## Tests

`tests/unit/body_part_viz/renderers/test_pyqtgl_renderer.py`:
- `pytest.importorskip("pyqtgraph")`.
- Headless `QT_QPA_PLATFORM=offscreen`.
- Build a `GLViewWidget` (in `app.exec` mode? — no, just instantiate and never show).
- Add a few shapes; assert items added.
- Performance: render a 26-segment / 5000-vertex setup; assert per-frame update < 16 ms (60 fps).

## Acceptance criteria

- [ ] `PyQtGLRenderer` implements `ShapeRenderer`.
- [ ] Optional dependency: `body_part_viz` works without pyqtgraph.
- [ ] Performance ≥ 60 fps for the target workload.
- [ ] ≥ 80% coverage (a bit lower than mpl renderer because GL ops are harder to fully cover headless).

## Files touched

- New: `src/shared/python/body_part_viz/renderers/pyqtgl_renderer.py`
- New: `tests/unit/body_part_viz/renderers/test_pyqtgl_renderer.py`
- Edit: `pyproject.toml` (add optional extra)

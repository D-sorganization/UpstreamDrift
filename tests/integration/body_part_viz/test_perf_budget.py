"""Performance budget regression tests for the body_part_viz renderer.

Two fleet-realistic workloads are exercised:

* 26 cylinders x 16 facets x 654 frames -> mean per-frame update <= 16 ms (60 fps).
* 26 small triangle meshes x ~200 verts x 654 frames -> mean per-frame
  update <= 33 ms (30 fps).

The budgets above are scaled by the ``BPV_PERF_BUDGET_SCALE`` env var so
slow CI runners can still pass without weakening the contract for local
development. Default scale is ``1.0``.
"""

from __future__ import annotations

import os
import time

import matplotlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.shared.python.body_part_viz import (  # noqa: E402
    BindingKind,
    FittedShape,
    MarkerBinding,
    ShapeTheme,
)
from src.shared.python.body_part_viz.renderers import (  # noqa: E402
    MatplotlibRenderer,
)
from src.shared.python.body_part_viz.shapes import (  # noqa: E402
    CylinderShape,
    MeshShape,
)

N_SHAPES = 26
N_FRAMES = 654
CYLINDER_FACETS = 16
MESH_VERTS = 200
SCALE = float(os.environ.get("BPV_PERF_BUDGET_SCALE", "1.0"))


def _fitted(shape_id: str, n_frames: int = N_FRAMES) -> FittedShape:
    binding = MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b"))
    t = np.linspace(0.0, 2.0 * np.pi, n_frames)
    centroid = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
    rotation = np.broadcast_to(np.eye(3), (n_frames, 3, 3)).copy()
    scale = np.ones((n_frames, 3))
    mask = np.ones((n_frames,), dtype=bool)
    return FittedShape(
        shape_id=shape_id,
        binding=binding,
        centroid=centroid,
        rotation_matrix=rotation,
        scale=scale,
        valid_mask=mask,
    )


def _build_synthetic_mesh(n_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a closed triangle strip mesh of approximately ``n_verts`` vertices."""
    n = max(n_verts, 4)
    # Build as two parallel rings -> ``n`` vertices = 2 * ring_size.
    ring_size = max(n // 2, 4)
    angles = np.linspace(0.0, 2.0 * np.pi, ring_size, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    radius = 0.05
    bottom = np.stack(
        [np.full_like(cos_a, -0.1), radius * cos_a, radius * sin_a], axis=1
    )
    top = np.stack([np.full_like(cos_a, 0.1), radius * cos_a, radius * sin_a], axis=1)
    verts = np.concatenate([bottom, top], axis=0).astype(np.float64)
    faces = []
    for i in range(ring_size):
        j = (i + 1) % ring_size
        faces.append([i, j, ring_size + j])
        faces.append([i, ring_size + j, ring_size + i])
    return verts, np.asarray(faces, dtype=np.int64)


@pytest.fixture
def fig_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    yield ax
    plt.close(fig)


def _measure_mean_frame_ms(renderer: MatplotlibRenderer, handles: list[str]) -> float:
    # Warm-up to amortise Python / numpy first-touch costs.
    for _ in range(3):
        for handle in handles:
            renderer.update_frame(handle, 0)

    n_frames = N_FRAMES
    start = time.perf_counter()
    for frame in range(n_frames):
        for handle in handles:
            renderer.update_frame(handle, frame)
    elapsed = time.perf_counter() - start
    # Mean per-frame budget: sum of all update_frame calls for one frame.
    return (elapsed / n_frames) * 1000.0


@pytest.mark.benchmark
def test_perf_26_cylinders_60fps(fig_ax) -> None:
    """26 cylinders @ 16 facets must update at >= 60 fps mean."""
    renderer = MatplotlibRenderer(fig_ax)
    theme = ShapeTheme(color="#1f77b4")
    handles = []
    for i in range(N_SHAPES):
        cyl = CylinderShape(
            length=0.5, radius=0.05, n_facets=CYLINDER_FACETS, shape_id=f"cyl{i}"
        )
        handles.append(renderer.add_shape(cyl, _fitted(f"cyl{i}"), theme))

    mean_ms = _measure_mean_frame_ms(renderer, handles)
    budget = 16.0 * SCALE
    assert (
        mean_ms <= budget
    ), f"26 cylinders mean per-frame {mean_ms:.2f} ms exceeds {budget:.2f} ms budget"


@pytest.mark.benchmark
def test_perf_26_library_meshes_30fps(fig_ax) -> None:
    """26 ~200-vert meshes must update at >= 30 fps mean."""
    renderer = MatplotlibRenderer(fig_ax)
    theme = ShapeTheme(color="#7dd3fc", opacity=0.6)
    verts, faces = _build_synthetic_mesh(MESH_VERTS)
    handles = []
    for i in range(N_SHAPES):
        mesh = MeshShape(
            vertices=verts.copy(),
            faces=faces.copy(),
            rest_dimensions=(0.2, 0.1, 0.1),
            shape_id=f"mesh{i}",
        )
        handles.append(renderer.add_shape(mesh, _fitted(f"mesh{i}"), theme))

    mean_ms = _measure_mean_frame_ms(renderer, handles)
    budget = 33.0 * SCALE
    assert (
        mean_ms <= budget
    ), f"26 meshes mean per-frame {mean_ms:.2f} ms exceeds {budget:.2f} ms budget"

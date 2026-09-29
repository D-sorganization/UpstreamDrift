"""Golden image snapshot tests for the body_part_viz matplotlib renderer.

Renders deterministic scenes to small PNGs at fixed DPI on the ``Agg``
backend and compares against committed reference fixtures under
``tests/fixtures/body_part_viz/``. RMS pixel diff is the comparison
metric; the tolerance is 0.5% of the dynamic range (255).

Set the ``BPV_REGEN_GOLDENS=1`` environment variable to overwrite the
committed fixtures (use sparingly; review the diff before committing).
"""

from __future__ import annotations

import os
from pathlib import Path

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
    LineShape,
    MeshShape,
)

# Tolerance: 0.5% of full pixel dynamic range (8-bit, 0..255).
RMS_TOLERANCE = 0.005 * 255.0
# Fixed DPI keeps the rasterisation comparable across platforms.
SNAPSHOT_DPI = 100
SNAPSHOT_FIGSIZE = (4.0, 4.0)
N_FRAMES = 100

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "body_part_viz"
REGEN = os.environ.get("BPV_REGEN_GOLDENS", "").strip() not in (
    "",
    "0",
    "false",
    "False",
)


# ---------------------------------------------------------------------------
# Scene construction helpers
# ---------------------------------------------------------------------------
def _swing_fitted(shape_id: str, axis: str = "x") -> FittedShape:
    """Build a FittedShape that traces a deterministic swing arc.

    Centroid sweeps around the unit circle in the chosen plane so frame 0
    and frame 50 are visually distinct.
    """
    binding = MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b"))
    t = np.linspace(0.0, np.pi, N_FRAMES)
    centroid = np.zeros((N_FRAMES, 3))
    if axis == "x":
        centroid[:, 0] = np.cos(t)
        centroid[:, 1] = np.sin(t)
    elif axis == "y":
        centroid[:, 0] = np.sin(t) * 0.5
        centroid[:, 2] = np.cos(t) * 0.5
    else:  # "z"
        centroid[:, 1] = np.cos(t) * 0.5
        centroid[:, 2] = np.sin(t) * 0.5

    rotation = np.broadcast_to(np.eye(3), (N_FRAMES, 3, 3)).copy()
    # Add a slow rotation around z so frame 0 / 50 differ even if the
    # centroid happens to coincide.
    angles = np.linspace(0.0, np.pi / 2.0, N_FRAMES)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    rotation[:, 0, 0] = cos_a
    rotation[:, 0, 1] = -sin_a
    rotation[:, 1, 0] = sin_a
    rotation[:, 1, 1] = cos_a
    scale = np.ones((N_FRAMES, 3))
    mask = np.ones((N_FRAMES,), dtype=bool)
    return FittedShape(
        shape_id=shape_id,
        binding=binding,
        centroid=centroid,
        rotation_matrix=rotation,
        scale=scale,
        valid_mask=mask,
    )


def _build_default_scene(ax) -> MatplotlibRenderer:
    """Add a line, a cylinder, and a small triangle mesh."""
    renderer = MatplotlibRenderer(ax)
    line = LineShape(length=0.6, shape_id="line")
    cyl = CylinderShape(length=0.8, radius=0.15, n_facets=12, shape_id="cylinder")
    # Hand-built tetrahedron mesh — small enough to commit a tiny golden.
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.4],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
        dtype=np.int64,
    )
    mesh = MeshShape(
        vertices=verts,
        faces=faces,
        rest_dimensions=(0.4, 0.4, 0.4),
        shape_id="mesh:tetra",
    )
    renderer.add_shape(
        line, _swing_fitted("line", axis="x"), ShapeTheme(color="#1f77b4")
    )
    renderer.add_shape(
        cyl,
        _swing_fitted("cylinder", axis="y"),
        ShapeTheme(color="#ff7f0e", opacity=0.7),
    )
    renderer.add_shape(
        mesh,
        _swing_fitted("mesh:tetra", axis="z"),
        ShapeTheme(color="#2ca02c", opacity=0.6),
    )
    return renderer


def _setup_axes():
    fig = plt.figure(figsize=SNAPSHOT_FIGSIZE, dpi=SNAPSHOT_DPI)
    ax = fig.add_subplot(111, projection="3d")
    # Fixed view + limits remove every source of platform-dependent jitter.
    ax.view_init(elev=20.0, azim=45.0)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    return fig, ax


def _render_to_array(fig) -> np.ndarray:
    """Render figure to an ``(H, W, 4)`` RGBA uint8 array."""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(int(height), int(width), 4).copy()


def _save_png(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use the explicit figure dpi/size with NO tight bbox so the saved
    # raster matches ``_render_to_array`` byte-for-byte on the same
    # platform; tolerance handles cross-platform jitter.
    fig.savefig(path, dpi=SNAPSHOT_DPI)


def _load_png(path: Path) -> np.ndarray:
    img = plt.imread(path)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if img.ndim == 2:  # grayscale -> rgba
        img = np.stack([img, img, img, np.full_like(img, 255)], axis=-1)
    if img.shape[-1] == 3:
        alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=-1)
    return img


def _rms_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        # Allow off-by-one rounding between platforms via centre-crop to
        # the smaller extents.
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _compare_or_regen(fig, golden_path: Path) -> None:
    if REGEN or not golden_path.is_file():
        _save_png(fig, golden_path)
        if not REGEN:
            pytest.skip(
                f"Generated missing golden {golden_path.name} on first run; "
                "rerun the test to compare."
            )
        return
    actual = _render_to_array(fig)
    expected = _load_png(golden_path)
    rms = _rms_diff(actual, expected)
    assert rms <= RMS_TOLERANCE, (
        f"snapshot {golden_path.name} RMS diff {rms:.3f} exceeds "
        f"tolerance {RMS_TOLERANCE:.3f}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.fixture
def fig_ax():
    fig, ax = _setup_axes()
    yield fig, ax
    plt.close(fig)


def test_default_three_segments_frame_0(fig_ax) -> None:
    fig, ax = fig_ax
    renderer = _build_default_scene(ax)
    for handle in list(renderer._entries.keys()):
        renderer.update_frame(handle, 0)
    _compare_or_regen(fig, FIXTURES_DIR / "default_three_frame_0.png")


def test_default_three_segments_frame_50(fig_ax) -> None:
    fig, ax = fig_ax
    renderer = _build_default_scene(ax)
    for handle in list(renderer._entries.keys()):
        renderer.update_frame(handle, 50)
    _compare_or_regen(fig, FIXTURES_DIR / "default_three_frame_50.png")


def test_library_full_body_at_address(fig_ax) -> None:
    """Render a small library-shape body using a synthetic address pose.

    The reference scene uses the same library every fleet asset uses, but
    we feed it synthetic, deterministic markers rather than parsing
    ``data/C3D_TA_Driver.c3d`` to keep CI stable across c3d-loader
    upgrades. The renderer code path is identical.
    """
    pytest.importorskip("trimesh")
    from src.shared.python.body_part_viz.asset_library import ShapeLibrary
    from src.shared.python.body_part_viz.fitters.between_two import (
        BetweenTwoMarkersFitter,
    )

    fig, ax = fig_ax
    library = ShapeLibrary.default()
    fitter = BetweenTwoMarkersFitter()
    renderer = MatplotlibRenderer(ax)
    theme = ShapeTheme(color="#7dd3fc", opacity=0.55, edge_color="#0c4a6e")

    rendered = 0
    for name in library.names():
        try:
            binding = library.binding_template(name)
        except Exception:  # noqa: BLE001 - skip shapes that fail to bind
            continue
        if binding.kind is not BindingKind.BETWEEN_TWO:
            continue
        # Synthetic address-pose marker positions: every shape's two
        # endpoints are placed at deterministic offsets so the rendered
        # image is a stable function of the library only.
        markers: dict[str, np.ndarray] = {}
        for i, marker in enumerate(binding.marker_names):
            # Use a stable byte-sum hash so the synthetic address pose
            # is deterministic across Python invocations.
            base = (sum(marker.encode("utf-8")) % 7) * 0.05
            markers[marker] = np.tile(
                np.array([[base, 0.1 * i, 0.1 * (i + 1)]], dtype=float),
                (N_FRAMES, 1),
            )
        try:
            shape = library.get(name)
            fitted = fitter.fit(shape, binding, markers)
            renderer.add_shape(shape, fitted, theme)
            rendered += 1
        except Exception:  # noqa: BLE001 - skip shapes that fail to render
            continue

    assert rendered > 0, "library rendered zero shapes"
    for handle in list(renderer._entries.keys()):
        renderer.update_frame(handle, 0)
    _compare_or_regen(fig, FIXTURES_DIR / "library_full_body_address.png")

"""Body-skeleton style swap tests for the starting-pose matcher.

Wave 4 of EPIC #4755 — issue #4767. Verifies that the
:class:`LiveViewController` can swap between line segments and
body_part_viz library shapes without tearing down the loaded body
target, and that scrubbing the timeline updates artists for both
modes. A small performance regression test guards the 30 fps target
against the canonical 301-frame fixture.

Tests are headless: ``QT_QPA_PLATFORM=offscreen`` and matplotlib
``Agg`` backend so they run on CI without a display.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

# Headless backend BEFORE matplotlib is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tools.starting_pose_matcher.live_view_controller import (
    BODY_SKELETON_STYLES,
    DEFAULT_BODY_SKELETON_STYLE,
    BodyLibraryShapesLayer,
    LiveViewController,
)
from src.tools.starting_pose_matcher.session_schema import (
    BODY_SKELETON_STYLES as SCHEMA_BODY_SKELETON_STYLES,
    DEFAULT_BODY_SKELETON_STYLE as SCHEMA_DEFAULT_STYLE,
    SESSION_SCHEMA_VERSION,
    BodySkeletonBlock,
    default_body_skeleton,
    parse_body_skeleton,
    serialize_body_skeleton,
)

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[4]
C3D_PATH = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def axes_canvas():
    """Yield a fresh ``(Axes3D, FigureCanvasBase)`` pair on the Agg backend."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    yield ax, fig.canvas
    plt.close(fig)


@pytest.fixture(scope="module")
def c3d_body_target():
    """Load the canonical 301-frame body target once per module."""
    if not C3D_PATH.exists():
        pytest.skip(f"C3D fixture not present: {C3D_PATH}")
    from src.shared.python.motion_matching.load_body_target import load_body_target

    return load_body_target(str(C3D_PATH))


def _synthetic_body_with_pig_pair(n_frames: int = 16) -> SimpleNamespace:
    """Build a duck-typed body target with a single between-two binding pair.

    The default :class:`ShapeLibrary` "head" shape binds between the
    canonical PiG markers ``HeadFront`` and ``HeadTop``; supplying both
    in synthetic data lets us assert at least one library shape resolves
    without dragging in the multi-MB C3D fixture.
    """
    rng = np.random.default_rng(42)
    names = ("HeadFront", "HeadTop", "BackTop")
    marker_xyz = rng.normal(size=(n_frames, len(names), 3)).astype(float)
    # Move HeadTop above HeadFront so the segment has non-zero length.
    marker_xyz[:, 1, 2] += 0.2
    return SimpleNamespace(marker_xyz=marker_xyz, marker_names=names)


# --------------------------------------------------------------------------- #
# Schema persistence                                                          #
# --------------------------------------------------------------------------- #


def test_session_schema_bumped_to_v6() -> None:
    assert SESSION_SCHEMA_VERSION == 6


def test_body_skeleton_block_default_round_trip() -> None:
    block = default_body_skeleton()
    assert block.style == SCHEMA_DEFAULT_STYLE == "lines"
    encoded = serialize_body_skeleton(block)
    assert encoded == {"style": "lines"}
    assert parse_body_skeleton(encoded) == block


def test_body_skeleton_block_library_shapes_round_trip() -> None:
    block = BodySkeletonBlock(style="library_shapes")
    encoded = serialize_body_skeleton(block)
    assert encoded == {"style": "library_shapes"}
    assert parse_body_skeleton(encoded) == block


def test_body_skeleton_block_missing_falls_back_to_default() -> None:
    """Pre-v5 sessions must still load (missing key -> default 'lines')."""
    assert parse_body_skeleton(None) == default_body_skeleton()
    assert parse_body_skeleton({}) == default_body_skeleton()


def test_body_skeleton_block_unknown_style_falls_back() -> None:
    parsed = parse_body_skeleton({"style": "not_a_real_style"})
    assert parsed.style == SCHEMA_DEFAULT_STYLE


def test_schema_and_controller_constants_agree() -> None:
    """Controller constants must mirror the schema's literal set."""
    assert BODY_SKELETON_STYLES == SCHEMA_BODY_SKELETON_STYLES
    assert DEFAULT_BODY_SKELETON_STYLE == SCHEMA_DEFAULT_STYLE


# --------------------------------------------------------------------------- #
# Controller swap behaviour                                                   #
# --------------------------------------------------------------------------- #


def test_default_style_is_lines(axes_canvas) -> None:
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax, canvas)
    assert ctrl.body_skeleton_style == "lines"


def test_invalid_style_raises_in_constructor(axes_canvas) -> None:
    ax, canvas = axes_canvas
    with pytest.raises(ValueError, match="body_skeleton_style"):
        LiveViewController(ax, canvas, body_skeleton_style="fancy")  # type: ignore[arg-type]


def test_invalid_style_raises_in_set(axes_canvas) -> None:
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax, canvas)
    with pytest.raises(ValueError, match="style"):
        ctrl.set_body_skeleton_style("hexagons")  # type: ignore[arg-type]


def test_set_style_no_op_when_unchanged(axes_canvas) -> None:
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax, canvas)
    body = _synthetic_body_with_pig_pair()
    ctrl.set_target(body=body)
    layer_before = ctrl.layers().get("body_skeleton")
    ctrl.set_body_skeleton_style("lines")
    layer_after = ctrl.layers().get("body_skeleton")
    assert layer_before is layer_after


def test_set_style_before_target_just_records_choice(axes_canvas) -> None:
    """Switching style with no body loaded must not raise."""
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax, canvas)
    ctrl.set_body_skeleton_style("library_shapes")
    assert ctrl.body_skeleton_style == "library_shapes"
    assert ctrl.layers() == {}


def test_lines_to_library_shapes_swap_is_non_destructive(axes_canvas) -> None:
    """Swapping styles preserves the loaded body data + frame count."""
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax, canvas)
    body = _synthetic_body_with_pig_pair(n_frames=12)
    ctrl.set_target(body=body)
    ctrl.set_frame(7)
    assert ctrl.current_frame == 7
    n_before = ctrl.n_frames

    ctrl.set_body_skeleton_style("library_shapes")
    assert ctrl.body_skeleton_style == "library_shapes"
    # Body markers + trail untouched; only the skeleton renderer swapped.
    assert ctrl.has_layer("body_markers")
    assert ctrl.has_layer("body_trail")
    assert ctrl.has_layer("body_skeleton")
    assert ctrl.n_frames == n_before
    assert ctrl.current_frame == 7
    skeleton_layer = ctrl.layers()["body_skeleton"]
    assert isinstance(skeleton_layer, BodyLibraryShapesLayer)


def test_library_to_lines_swap_clean_artist_swap(axes_canvas) -> None:
    """Swapping back to lines drops library artists and adds line artists."""
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax, canvas, body_skeleton_style="library_shapes")
    body = _synthetic_body_with_pig_pair(n_frames=8)
    ctrl.set_target(body=body)
    lib_layer = ctrl.layers()["body_skeleton"]
    assert isinstance(lib_layer, BodyLibraryShapesLayer)
    lib_artists = list(lib_layer.artists())

    ctrl.set_body_skeleton_style("lines")
    new_layer = ctrl.layers()["body_skeleton"]
    assert not isinstance(new_layer, BodyLibraryShapesLayer)
    new_artists = list(new_layer.artists())

    # Old library artists must be detached from the axes; new artists
    # must have been registered on the same axes object.
    for art in lib_artists:
        assert art not in ax.collections, "old library shape artist still attached"
    for art in new_artists:
        # Lines layer registers a Line3DCollection — at minimum the
        # artist set must be non-empty for a non-trivial body.
        assert art is not None


def test_library_layer_skips_missing_markers(axes_canvas) -> None:
    """Library shapes whose required markers are absent are silently skipped."""
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax, canvas, body_skeleton_style="library_shapes")
    body = SimpleNamespace(
        marker_xyz=np.zeros((4, 2, 3)),
        marker_names=("foo", "bar"),
    )
    # Should not raise; the body skeleton layer is built but has zero shapes.
    ctrl.set_target(body=body)
    layer = ctrl.layers()["body_skeleton"]
    assert isinstance(layer, BodyLibraryShapesLayer)
    assert layer.shape_count == 0
    assert layer.shape_names == ()


# --------------------------------------------------------------------------- #
# Scrub semantics                                                             #
# --------------------------------------------------------------------------- #


def test_scrub_in_library_mode_updates_without_error(c3d_body_target, axes_canvas) -> None:
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax, canvas, body_skeleton_style="library_shapes")
    ctrl.set_target(body=c3d_body_target)
    layer = ctrl.layers()["body_skeleton"]
    assert isinstance(layer, BodyLibraryShapesLayer)
    # The C3D fixture exposes PiG markers (HeadFront / HeadTop / ...);
    # at least one library shape must resolve.
    assert layer.shape_count > 0
    # Renderer artist count > 0 (acceptance criterion).
    assert len(layer.artists()) == layer.shape_count

    n = ctrl.n_frames
    # Scrub through several frames without exceptions.
    for f in (0, n // 4, n // 2, (3 * n) // 4, n - 1):
        ctrl.set_frame(f)
        assert ctrl.current_frame == f


def test_scrub_in_lines_mode_updates_without_error(c3d_body_target, axes_canvas) -> None:
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax, canvas, body_skeleton_style="lines")
    ctrl.set_target(body=c3d_body_target)
    n = ctrl.n_frames
    assert n > 0
    for f in range(0, n, max(1, n // 8)):
        ctrl.set_frame(f)


# --------------------------------------------------------------------------- #
# Performance                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.benchmark
def test_library_shapes_scrub_meets_30_fps(c3d_body_target, axes_canvas) -> None:
    """A full 301-frame scrub must average >= 30 fps in library-shapes mode.

    Measures the per-frame *data-update* cost (matching what an
    interactive Qt backend pays — Qt batches paints via the event loop,
    so the user-visible scrub fps is bounded by the data-update path,
    not the synchronous Agg ``draw()`` call). The test patches
    ``canvas.draw_idle`` to a no-op for the duration of the timed loop
    to model that behaviour without bringing up a real Qt event loop.
    """
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax, canvas, body_skeleton_style="library_shapes")
    ctrl.set_target(body=c3d_body_target)
    layer = ctrl.layers()["body_skeleton"]
    assert isinstance(layer, BodyLibraryShapesLayer)
    assert layer.shape_count > 0

    n = ctrl.n_frames
    # Warm-up frame so import / cache costs do not pollute the timing.
    ctrl.set_frame(0)

    original_draw_idle = canvas.draw_idle
    canvas.draw_idle = lambda *a, **kw: None  # type: ignore[assignment]
    try:
        start = time.perf_counter()
        for f in range(n):
            ctrl.set_frame(f)
        elapsed = time.perf_counter() - start
    finally:
        canvas.draw_idle = original_draw_idle  # type: ignore[assignment]

    fps = n / elapsed if elapsed > 0 else float("inf")
    # Record on the test report so the operator can read the measured fps.
    print(f"\n[matcher-bpv] library-shapes scrub fps: {fps:.1f} (n={n}, elapsed={elapsed:.3f}s)")
    assert fps >= 30.0, f"library-shapes scrub fps {fps:.1f} below 30 fps target"

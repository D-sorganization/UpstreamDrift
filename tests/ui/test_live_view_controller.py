"""Headless test for :class:`LiveViewController` (issue #4512).

Verifies the live-view controller can:

1. Bind a real ``BodyTarget`` loaded from ``data/C3D_TA_Driver.c3d`` to a
   matplotlib 3-D axis under the ``Agg`` backend.
2. Update its ``BodyMarkerLayer`` artist data per-frame to match
   ``target.marker_xyz[t]`` for finite samples (frames 0, 100, 250, last).
3. Toggle the ``body_markers`` layer off and have the artist's visibility
   flag follow.

Headless invariants: ``QT_QPA_PLATFORM=offscreen`` and
``MPLBACKEND=Agg`` are set before any matplotlib import.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

pytestmark = [pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER_C3D = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"


@pytest.fixture(scope="module")
def loaded_body():
    """Load a ``BodyTarget`` from the bundled driver C3D fixture."""
    if not DRIVER_C3D.exists():
        pytest.skip(f"C3D fixture not present: {DRIVER_C3D}")
    try:
        from src.shared.python.motion_matching.load_body_target import (
            load_body_target,
        )
    except Exception as exc:  # pragma: no cover - import-time failure
        pytest.skip(f"load_body_target unavailable: {exc}")
    try:
        return load_body_target(DRIVER_C3D)
    except ImportError as exc:  # ezc3d optional dependency missing
        pytest.skip(f"C3D loader dependency unavailable: {exc}")
    except Exception as exc:  # pragma: no cover - loader-side failure
        pytest.skip(f"failed to load C3D fixture: {exc}")


@pytest.fixture()
def axes_canvas():
    """Build a fresh ``Axes3D`` + ``Agg`` canvas pair."""
    fig = Figure(figsize=(6, 6), dpi=80)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    return ax, canvas, fig


def test_controller_renders_body_markers_per_frame(loaded_body, axes_canvas) -> None:
    from src.tools.starting_pose_matcher.live_view_controller import (
        LiveViewController,
    )

    ax, canvas, _fig = axes_canvas
    controller = LiveViewController(ax, canvas)
    controller.set_target(body=loaded_body, club=None)

    assert controller.has_layer("body_markers")
    assert controller.has_layer("body_skeleton")
    assert controller.n_frames == int(loaded_body.marker_xyz.shape[0])

    n = controller.n_frames
    test_frames = [0, 100, 250, n - 1]
    test_frames = [t for t in test_frames if 0 <= t < n]

    body_layer = controller.layers()["body_markers"]
    art = body_layer.artists()[0]

    for t in test_frames:
        controller.set_frame(t)
        # Pull artist data back. Line3D.get_data_3d returns three 1-D arrays.
        xs, ys, zs = art.get_data_3d()
        expected = loaded_body.marker_xyz[t]  # (M, 3)
        finite = np.isfinite(expected).all(axis=-1)
        np.testing.assert_allclose(
            np.asarray(xs)[finite],
            expected[finite, 0],
            atol=1e-9,
            err_msg=f"x mismatch at frame {t}",
        )
        np.testing.assert_allclose(
            np.asarray(ys)[finite],
            expected[finite, 1],
            atol=1e-9,
            err_msg=f"y mismatch at frame {t}",
        )
        np.testing.assert_allclose(
            np.asarray(zs)[finite],
            expected[finite, 2],
            atol=1e-9,
            err_msg=f"z mismatch at frame {t}",
        )


def test_controller_layer_visibility_toggle(loaded_body, axes_canvas) -> None:
    from src.tools.starting_pose_matcher.live_view_controller import (
        LiveViewController,
    )

    ax, canvas, _fig = axes_canvas
    controller = LiveViewController(ax, canvas)
    controller.set_target(body=loaded_body, club=None)

    body_layer = controller.layers()["body_markers"]
    art = body_layer.artists()[0]
    assert art.get_visible() is True

    controller.set_layer_visible("body_markers", False)
    assert art.get_visible() is False
    assert body_layer.visible is False

    controller.set_layer_visible("body_markers", True)
    assert art.get_visible() is True


def test_body_skeleton_segments_are_int_pairs_into_marker_names(
    loaded_body, axes_canvas
) -> None:
    """Pins #4582 default_body_segments fix.

    The body-skeleton layer must receive ``(int, int)`` index pairs, and
    every endpoint must be a valid index into ``marker_names`` — i.e. the
    name->index remap survives into the segment list rather than leaking
    raw ``BodySegment`` dataclasses.
    """
    from src.shared.python.motion_matching.body_skeleton import (
        default_body_segments,
    )
    from src.tools.starting_pose_matcher.live_view_controller import (
        _default_body_segments_safe,
    )

    pairs = _default_body_segments_safe(tuple(loaded_body.marker_names))
    assert pairs, "expected at least one body-skeleton segment"
    m = len(loaded_body.marker_names)
    for pair in pairs:
        assert isinstance(pair, tuple) and len(pair) == 2
        ia, ib = pair
        assert isinstance(ia, int) and not isinstance(ia, bool)
        assert isinstance(ib, int) and not isinstance(ib, bool)
        assert 0 <= ia < m, f"segment endpoint {ia} out of range"
        assert 0 <= ib < m, f"segment endpoint {ib} out of range"
    # Sanity: helper returned at least as many pairs as the upstream segment
    # list could resolve given this marker set (defensive against silent
    # name-miss fallthroughs).
    name_set = set(loaded_body.marker_names)
    resolvable = [
        s
        for s in default_body_segments(tuple(loaded_body.marker_names))
        if getattr(s, "a", None) in name_set and getattr(s, "b", None) in name_set
    ]
    assert len(pairs) >= len(resolvable)


def test_controller_handles_missing_club_and_ball(loaded_body, axes_canvas) -> None:
    """Body-only targets must not raise when club/ball slots are absent."""
    from src.tools.starting_pose_matcher.live_view_controller import (
        LiveViewController,
    )

    ax, canvas, _fig = axes_canvas
    controller = LiveViewController(ax, canvas)
    controller.set_target(body=loaded_body, club=None, ball=None)

    assert not controller.has_layer("club_midhands_trace")
    assert not controller.has_layer("clubface_trace")
    assert not controller.has_layer("ball_impact")

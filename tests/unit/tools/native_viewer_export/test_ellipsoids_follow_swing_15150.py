"""Export ellipsoids are evaluated at the pose of the frame they are drawn on (#15150).

A clip is sampled by time, so its frame ``k`` is not source state ``k``. Every
ellipsoid of a clip frame must sit on its point at that frame's pose, its shape
must follow the swing, and a feed that cannot follow the clip must be refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.hand_mobility import (  # noqa: E402
    HandMobilitySource,
)
from src.shared.python.motion_matching.full_body_spec import (  # noqa: E402
    load_full_body_spec,
)
from src.shared.python.motion_matching.same_input import InputBundle  # noqa: E402
from src.tools.native_viewer_export import flight_tracer  # noqa: E402
from src.tools.native_viewer_export.core import (  # noqa: E402
    ClipPlan,
    ExportSettings,
    OverlayFeed,
    SwingInput,
    _resampled,
)
from src.tools.native_viewer_export.overlay import (  # noqa: E402
    OverlayOptions,
    build_overlay_feed,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
#: source rate of the fixture swing; six source states per 60 fps clip frame
SOURCE_RATE_HZ = 360.0
SOURCE_STATES = 73
FPS = 60
#: an ellipsoid centre is the point's own forward kinematics at the frame pose
CENTRE_TOL_M = 1e-9
#: smallest relative change of the display radii between first and last frame
SHAPE_CHANGE_MIN = 0.02
KINDS = ("mobility", "effective-mass")
OPTIONS = OverlayOptions(
    hand_mobility="both", mobility_points="hands,feet,club", ellipsoid_color_by="kind"
)


@pytest.fixture(scope="module")
def spec_bytes() -> bytes:
    spec = load_full_body_spec(FULL, json.loads(UPPER.read_text(encoding="utf-8")))
    return json.dumps(spec).encode("utf-8")


@pytest.fixture(scope="module")
def source(spec_bytes: bytes) -> HandMobilitySource:
    return HandMobilitySource(spec_bytes)


@pytest.fixture(scope="module")
def swing(spec_bytes: bytes, source: HandMobilitySource) -> SwingInput:
    """A smooth sweep from the zero pose to a random pose at the source rate."""
    order = tuple(source.coordinate_order)
    end = np.random.default_rng(15150).normal(0.0, 0.35, len(order))
    ramp = np.linspace(0.0, 1.0, SOURCE_STATES)[:, None]
    q = ramp * end[None, :]
    v = np.gradient(q, 1.0 / SOURCE_RATE_HZ, axis=0)
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=order,
        dt_s=1.0 / SOURCE_RATE_HZ,
        q0=q[0],
        v0=v[0],
        efforts=np.zeros((SOURCE_STATES - 1, len(order))),
        reference_q=q,
        reference_v=v,
        reference_engine="mujoco",
    )
    return SwingInput(bundle, q, "sweep", "Driver", "mujoco", v=v)


@pytest.fixture(scope="module")
def clip(swing: SwingInput) -> tuple[SwingInput, OverlayFeed]:
    feed, _ = build_overlay_feed(swing, "mujoco", options=OPTIONS)
    settings = ExportSettings(
        views=("face_on",), speeds=(1.0,), fps=FPS, multiview=False
    )
    shown, shown_feed = _resampled(swing, feed, settings, ClipPlan("", 1.0))
    assert shown_feed is not None
    assert shown.n_states < swing.n_states // 5  # frames are not source states
    return shown, shown_feed


def _frames(shown: SwingInput) -> tuple[int, ...]:
    last = shown.n_states - 1
    return (0, last // 3, last // 2, (2 * last) // 3, last)


def _centres(feed: OverlayFeed, index: int, kind: str) -> dict[str, np.ndarray]:
    assert feed.hand_ellipsoids is not None
    return {
        g.label.split(" ")[0]: np.array(g.centre, dtype=float)
        for g in feed.hand_ellipsoids(index)
        if g.label.endswith(f" {kind}")
    }


def _coordinates(source: HandMobilitySource, q: np.ndarray) -> dict[str, float]:
    return dict(zip(source.coordinate_order, map(float, q), strict=True))


def test_hand_and_clubhead_ellipsoids_sit_on_their_points_in_every_frame(
    clip: tuple[SwingInput, OverlayFeed], source: HandMobilitySource
) -> None:
    shown, feed = clip
    for index in _frames(shown):
        coordinates = _coordinates(source, shown.q[index])
        head = np.asarray(source.frame_poses(coordinates)["Clubhead"])[:3, 3]
        for kind in KINDS:
            centres = _centres(feed, index, kind)
            for hand in ("lead", "trail"):
                expected = source.grip_point_world(hand, coordinates)
                error = np.abs(centres[f"{hand}_hand"] - expected).max()
                assert error < CENTRE_TOL_M, (index, kind, hand, error)
            error = np.abs(centres["clubhead"] - head).max()
            assert error < CENTRE_TOL_M, (index, kind, error)


def test_ellipsoid_shape_and_hud_change_through_the_clip(
    clip: tuple[SwingInput, OverlayFeed],
) -> None:
    shown, feed = clip
    assert feed.hand_ellipsoids is not None
    first = {g.label: np.array(g.radii) for g in feed.hand_ellipsoids(0)}
    last = {
        g.label: np.array(g.radii) for g in feed.hand_ellipsoids(shown.n_states - 1)
    }
    for label in ("lead_hand mobility", "clubhead mobility"):
        change = np.abs(last[label] - first[label]).max() / first[label].max()
        assert change > SHAPE_CHANGE_MIN, (label, change)
    assert feed.hud_lines_at(0) != feed.hud_lines_at(shown.n_states - 1)


def test_last_clip_frame_shows_the_last_source_state(
    clip: tuple[SwingInput, OverlayFeed], swing: SwingInput
) -> None:
    """The clip ends on the last source state, so both give the same ellipsoids."""
    shown, feed = clip
    assert shown.time_s(shown.n_states - 1) == swing.time_s(swing.n_states - 1)
    source_feed, _ = build_overlay_feed(swing, "mujoco", options=OPTIONS)
    assert source_feed.hand_ellipsoids is not None and feed.hand_ellipsoids is not None
    want = source_feed.hand_ellipsoids(swing.n_states - 1)
    got = feed.hand_ellipsoids(shown.n_states - 1)
    assert [g.label for g in got] == [g.label for g in want]
    for mine, theirs in zip(got, want, strict=True):
        assert np.allclose(mine.radii, theirs.radii, rtol=1e-9, atol=0.0), mine.label
        assert np.allclose(mine.centre, theirs.centre, atol=CENTRE_TOL_M), mine.label


def test_foot_ellipsoids_lie_inside_their_foot_contact_region(
    clip: tuple[SwingInput, OverlayFeed], source: HandMobilitySource
) -> None:
    """Foot centre inside the box of the foot body's own contact spheres."""
    shown, feed = clip
    model, data = source.model, source.data
    for index in _frames(shown):
        centres = _centres(feed, index, "mobility")
        source._set_state(shown.q[index])
        for name, body in (("lead_foot", "calcn_l"), ("trail_foot", "calcn_r")):
            geoms = np.flatnonzero(model.geom_bodyid == model.body(body).id)
            assert geoms.size >= 2, body
            pos = np.array(data.geom_xpos[geoms], dtype=float)
            pad = float(np.max(model.geom_size[geoms, 0]))
            low, high = pos.min(axis=0) - pad, pos.max(axis=0) + pad
            centre = centres[name]
            assert ((centre >= low) & (centre <= high)).all(), (index, name)
            assert np.abs(centre - pos.mean(axis=0)).max() < CENTRE_TOL_M


def test_a_feed_that_cannot_follow_the_clip_is_refused(swing: SwingInput) -> None:
    fixed = OverlayFeed(lambda _i: None, hand_ellipsoids=lambda _i: ())
    settings = ExportSettings(
        views=("face_on",), speeds=(1.0,), fps=FPS, multiview=False
    )
    with pytest.raises(ValueError, match="ellipsoid"):
        _resampled(swing, fixed, settings, ClipPlan("", 1.0))


def test_held_frames_repeat_the_last_ellipsoids(
    clip: tuple[SwingInput, OverlayFeed],
) -> None:
    shown, feed = clip
    assert shown.sample_times_s is not None
    end = shown.sample_times_s[-1]
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 0.0]])
    flight = flight_tracer.FlightPlayback(
        flight_tracer.load_tracer_settings(),
        "flight",
        impact_time_s=end,
        times_s=np.array([0.0, 0.1, 0.2]),
        path_m=path,
        radius_m=0.02,
    )
    held, held_feed = flight.held(shown, feed, 1.0 / FPS)
    assert held.n_states > shown.n_states
    last = shown.n_states - 1
    want = _centres(feed, last, "mobility")
    got = _centres(held_feed, held.n_states - 1, "mobility")
    assert set(got) == set(want)
    for name, centre in want.items():
        assert np.array_equal(got[name], centre), name
    assert held_feed.hud_lines_at(held.n_states - 1) == feed.hud_lines_at(last)


def test_scale_grouping_sizes_points_alike_and_fades_the_outer_kind(
    swing: SwingInput,
) -> None:
    """Per point every point peaks near the target; per kind they differ."""
    from dataclasses import replace

    from src.tools.native_viewer_export.ellipsoid_visibility import (
        load_visibility_settings,
    )
    from src.tools.native_viewer_export.overlay import (
        MobilitySelection,
        _build_hand_ellipsoid_feed,
    )

    def peaks(grouping: str, alpha: float = 1.0) -> dict[str, float]:
        vis = replace(
            load_visibility_settings(), scale_grouping=grouping, outer_alpha=alpha
        )
        feed, _, _ = _build_hand_ellipsoid_feed(
            swing,
            ["mobility"],
            "constrained",
            "auto",
            selection=MobilitySelection("hands,feet,club", visibility=vis),
        )
        out: dict[str, float] = {}
        for i in range(swing.n_states):
            for g in feed(i):
                out[g.label] = max(out.get(g.label, 0.0), max(g.radii))
        return out

    own, shared = peaks("per_point"), peaks("per_kind")
    assert own.keys() == shared.keys() and len(own) >= 3
    spread = {
        k: max(v.values()) / min(v.values()) for k, v in (("pp", own), ("pk", shared))
    }
    assert spread["pp"] < 1.2 < spread["pk"]  # per point: every point ~ the target

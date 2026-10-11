"""Impact parameter card on the impact frame of native exports (GCV-19, #11725)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.impact_parameters.panel_model import CardRow, ImpactCard
from src.tools.native_viewer_export import core, impact_card
from src.tools.native_viewer_export.compositor import (
    draw_impact_card,
    impact_card_bbox,
)
from src.tools.native_viewer_export.core import (
    ExportSettings,
    SwingInput,
    export_swing,
)
from src.shared.python.motion_matching.same_input import InputBundle

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO = Path(__file__).resolve().parents[4]
SPEC = REPO / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
FIXTURE = REPO / "tests/fixtures/club_face/swing_q_driver.npz"


def _card(with_unavailable: bool = True) -> ImpactCard:
    rows = [
        CardRow("clubhead_speed", "Clubhead Speed", "mph", 101.26),
        CardRow("attack_angle_deg", "Attack Angle", "deg", -3.04),
        CardRow("club_path_deg", "Club Path", "deg", 2.5),
    ]
    if with_unavailable:
        rows.append(
            CardRow(
                "face_angle_deg",
                "Face Angle",
                "deg",
                None,
                "face unobservable: club axial rotation",
            )
        )
    return ImpactCard(
        units="mph",
        impact_time_s=1.0,
        impact_time_source="explicit",
        frame={},
        rows=tuple(rows),
        d_plane={},
    )


def _blank(w: int = 1280, h: int = 720) -> np.ndarray:
    return np.full((h, w, 3), 40, np.uint8)


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080)])
def test_draw_changes_pixels_only_inside_the_panel(size: tuple[int, int]) -> None:
    img = _blank(*size)
    out = draw_impact_card(img, _card())
    assert out is not img and out.shape == img.shape and out.dtype == np.uint8
    assert (img == 40).all(), "input must not be modified"
    x0, y0, x1, y1 = impact_card_bbox(img.shape, _card())
    changed = (out != img).any(axis=2)
    assert changed.any()
    ys, xs = np.nonzero(changed)
    assert xs.min() >= x0 and xs.max() < x1 and ys.min() >= y0 and ys.max() < y1
    # right-hand side panel: it must not cover the middle of the frame
    assert x0 > img.shape[1] // 2


def test_panel_grows_with_rows_and_stays_inside_the_frame() -> None:
    shape = _blank().shape
    small = impact_card_bbox(shape, _card(with_unavailable=False))
    big = impact_card_bbox(shape, _card())
    assert (big[3] - big[1]) > (small[3] - small[1])
    assert 0 <= big[0] < big[2] <= shape[1] and 0 <= big[1] < big[3] <= shape[0]


def test_unavailable_row_is_drawn_not_zero() -> None:
    with_row = draw_impact_card(_blank(), _card())
    without = draw_impact_card(_blank(), _card(with_unavailable=False))
    x0, y0, x1, y1 = impact_card_bbox(_blank().shape, _card())
    # the unavailable row occupies area that the shorter card leaves untouched
    extra = (with_row != without).any(axis=2)[y0:y1, x0:x1]
    assert extra.any()


def test_zero_valued_row_differs_from_unavailable_row() -> None:
    zero = _card(with_unavailable=False)
    zero = ImpactCard(
        **{
            **zero.__dict__,
            "rows": (*zero.rows, CardRow("face_angle_deg", "Face Angle", "deg", 0.0)),
        }
    )
    a = draw_impact_card(_blank(), zero)
    b = draw_impact_card(_blank(), _card())
    assert (a != b).any()


def test_unavailable_card_draws_reason_panel() -> None:
    card = ImpactCard(
        units="mph",
        impact_time_s=float("nan"),
        impact_time_source="unavailable",
        frame={},
        rows=(),
        d_plane={},
        available=False,
        reason="no club in model",
    )
    out = draw_impact_card(_blank(), card)
    assert (out != _blank()).any()


def test_draw_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="uint8"):
        draw_impact_card(np.zeros((10, 10, 3)), _card())
    with pytest.raises(ValueError, match="shape"):
        draw_impact_card(np.zeros((10, 10), np.uint8), _card())
    with pytest.raises(TypeError, match="ImpactCard"):
        draw_impact_card(_blank(), {"rows": []})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="position"):
        draw_impact_card(_blank(), _card(), position="middle")


# --- display rule in the export loop -------------------------------------------------


class _Writer:
    def __init__(self, sink: dict, path: Path) -> None:
        self.frames: list[np.ndarray] = []
        sink[path.name] = self

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        pass


class _Backend:
    engine = "fake"

    def unavailable_reason(self) -> None:
        return None

    def render(self, swing, settings, indices, overlay):
        for _k in indices:
            yield {
                v: np.full((settings.height, settings.width, 3), 40, np.uint8)
                for v in settings.views
            }


def _swing(steps: int = 100) -> SwingInput:
    import json

    names = ("A", "B")
    q = np.zeros((steps + 1, 2))
    bundle = InputBundle(
        spec_bytes=json.dumps({"coordinate_order": list(names)}).encode(),
        coordinate_order=names,
        dt_s=0.01,
        q0=q[0],
        v0=q[0],
        efforts=np.zeros((steps, 2)),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
    )
    return SwingInput(bundle, q, "driver", "Driver", "mujoco")


def _export(monkeypatch, tmp_path, card, **overrides):
    monkeypatch.setattr(core, "impact_card_for", lambda swing, t, ball=None: card)
    settings = ExportSettings(
        views=("face_on",),
        multiview=False,
        width=640,
        height=360,
        speeds=(1.0,),
        impact_time_s=0.5,
        **overrides,
    )
    sink: dict = {}
    result = export_swing(
        _Backend(),
        _swing(),
        settings,
        tmp_path,
        writer_factory=lambda p, fps: _Writer(sink, p),
    )
    (writer,) = sink.values()
    return result, writer.frames, settings


def _panel_changed(frame: np.ndarray, card: ImpactCard) -> bool:
    x0, y0, x1, y1 = impact_card_bbox(frame.shape, card)
    return bool((frame[y0:y1, x0:x1] != 40).any())


def test_card_shown_from_the_impact_frame_onward(monkeypatch, tmp_path) -> None:
    card = _card()
    result, frames, _ = _export(monkeypatch, tmp_path, card)
    assert len(frames) == 61  # 1 s at 60 fps
    shown = [_panel_changed(f, card) for f in frames]
    assert not any(shown[:30])  # t < 0.5 s
    assert all(shown[31:])
    assert shown[-1]
    assert result.impact_card == card.to_dict()


def test_card_off_by_setting_or_hud(monkeypatch, tmp_path) -> None:
    card = _card()
    _, frames, _ = _export(monkeypatch, tmp_path, card, impact_card=False)
    assert not any(_panel_changed(f, card) for f in frames)
    _, frames, _ = _export(monkeypatch, tmp_path, card, hud=False)
    assert not any(_panel_changed(f, card) for f in frames)


def test_unavailable_reason_is_recorded_and_nothing_is_drawn(
    monkeypatch, tmp_path
) -> None:
    result, frames, _ = _export(monkeypatch, tmp_path, "mujoco is not installed")
    assert result.impact_card == "mujoco is not installed"
    assert not any(_panel_changed(f, _card()) for f in frames)


# --- impact_card_for -----------------------------------------------------------------


def test_impact_card_for_reports_reason_without_mujoco(monkeypatch) -> None:
    def boom(swing):
        raise core.BackendUnavailable("mujoco is not available: test")

    monkeypatch.setattr(impact_card, "_club_series", boom)
    out = impact_card.impact_card_for(_swing(), 0.5)
    assert isinstance(out, str) and "mujoco" in out


def test_impact_card_for_reports_reason_for_a_model_without_a_club() -> None:
    out = impact_card.impact_card_for(_swing(), 0.5)
    assert isinstance(out, str) and out


def test_impact_card_for_rejects_bad_impact_time() -> None:
    with pytest.raises(ValueError, match="finite"):
        impact_card.impact_card_for(_swing(), float("nan"))


def test_impact_card_on_the_club_face_fixture() -> None:
    pytest.importorskip("mujoco")
    if not (SPEC.exists() and FIXTURE.exists()):
        pytest.skip("capture-A specification or club-face fixture missing")
    import json

    q = np.load(FIXTURE)["q"].astype(float)
    names = tuple(json.loads(SPEC.read_text())["coordinate_order"])
    z = np.zeros_like(q)
    bundle = InputBundle(
        spec_bytes=SPEC.read_bytes(),
        coordinate_order=names,
        dt_s=0.002,
        q0=q[0],
        v0=z[0],
        efforts=np.zeros((len(q) - 1, len(names))),
        reference_q=q,
        reference_v=z,
        reference_engine="mujoco",
    )
    swing = SwingInput(bundle, q, "driver", "Driver", "mujoco")
    from src.tools.native_viewer_export.overlay import detect_impact_time_s

    detect_time = detect_impact_time_s(swing)
    card = impact_card.impact_card_for(swing, detect_time)
    assert isinstance(card, ImpactCard), card
    keys = {r.key: r for r in card.rows}
    assert keys["clubhead_speed"].value is not None
    # the committed fixture is a stale low-speed pose set (#12223): only the
    # plumbing is asserted here, not golf-realistic speeds
    assert keys["clubhead_speed"].value > 0.0 and card.units == "mph"
    assert len(card.rows) == 10
    assert card.impact_time_s <= detect_time


# --- true-face series (review fix) ---------------------------------------------------


def _fixture_swing(path: Path = FIXTURE) -> SwingInput:
    pytest.importorskip("mujoco")
    if not (SPEC.exists() and path.exists()):
        pytest.skip("capture-A specification or club-face fixture missing")
    import json

    q = np.load(path)["q"].astype(float)
    names = tuple(json.loads(SPEC.read_text())["coordinate_order"])
    z = np.zeros_like(q)
    bundle = InputBundle(
        spec_bytes=SPEC.read_bytes(),
        coordinate_order=names,
        dt_s=0.002,
        q0=q[0],
        v0=z[0],
        efforts=np.zeros((len(q) - 1, len(names))),
        reference_q=q,
        reference_v=z,
        reference_engine="mujoco",
    )
    return SwingInput(bundle, q, "driver", "Driver", "mujoco")


def _line_series(heading, speed: float = 40.0):
    from src.shared.python.impact_parameters import ClubheadSeries

    t = np.arange(0.0, 0.2, 0.001)
    n = len(t)
    heading = np.asarray(heading, dtype=float)
    return ClubheadSeries(
        times_s=t,
        face_center_m=np.outer(speed * t, heading),
        velocity_mps=np.tile(speed * heading, (n, 1)),
        face_normal=np.tile(heading, (n, 1)),
        toe_axis=np.tile([0.0, 0.0, -1.0], (n, 1)),
        grip_axis=np.tile([0.0, 0.0, 1.0], (n, 1)),
    )


def test_straight_square_swing_gives_known_parameters() -> None:
    card = impact_card.card_from_series(_line_series([1.0, 0.0, 0.0]), 0.15)
    rows = {r.key: r.value for r in card.rows}
    assert rows["clubhead_speed"] == pytest.approx(89.48, rel=0.005)
    assert rows["club_path_deg"] == pytest.approx(0.0, abs=0.05)
    assert rows["face_angle_deg"] == pytest.approx(0.0, abs=0.05)
    assert rows["attack_angle_deg"] == pytest.approx(0.0, abs=0.05)
    assert card.extras["target_source"] == "address_face_normal"


def test_target_follows_the_address_face_normal() -> None:
    card = impact_card.card_from_series(_line_series([0.0, -1.0, 0.0]), 0.15)
    rows = {r.key: r.value for r in card.rows}
    assert rows["club_path_deg"] == pytest.approx(0.0, abs=0.05)
    assert rows["face_angle_deg"] == pytest.approx(0.0, abs=0.05)


def test_club_face_world_is_consistent_with_the_ball() -> None:
    from src.shared.python.model_appearance.ball import (
        BALL_RADIUS_M,
        ball_position_at_address,
    )
    from src.tools.native_viewer_export.ball import address_ball
    from src.tools.native_viewer_export.club_pose import ClubPoseModel

    swing = _fixture_swing()
    model = ClubPoseModel(swing.bundle.spec_bytes)
    names = swing.bundle.coordinate_order
    coords = dict(zip(names, map(float, swing.q[0]), strict=True))
    face = model.face_world(coords)
    assert np.linalg.norm(face.normal) == pytest.approx(1.0)
    assert face.toe @ face.normal == pytest.approx(0.0, abs=1e-9)
    assert np.cross(face.toe, face.normal) == pytest.approx(face.grip, abs=1e-9)
    ball = ball_position_at_address(
        face.centre_m, face.normal, ground_height_m=model.ground_height_m
    )
    assert ball[:2] - BALL_RADIUS_M * face.normal[:2] == pytest.approx(
        face.centre_m[:2], abs=1e-3
    )
    placed = address_ball(swing, None)
    if not isinstance(placed, str):  # the ball module uses the same helper
        assert np.asarray(placed.position_m) == pytest.approx(ball, abs=1e-6)


_RADIUS = 0.02135


def _placement(series, offset=(0.0, 0.0, 0.0), source="address_geometry"):
    """Ball whose facing surface point is the face centre plus ``offset``.

    The face normal points from the face to the ball, so the centre is one
    radius ahead of the wanted contact point along the normal.
    """
    from src.tools.native_viewer_export.ball import BallPlacement

    idx = 149  # contact index 150 (t = 0.15 s) -> impact sample 149
    contact = np.asarray(series.face_center_m[idx]) + np.asarray(offset)
    centre = contact + _RADIUS * np.asarray(series.face_normal[idx])
    return BallPlacement(tuple(centre), _RADIUS, source)


def _rows(card):
    return {r.key: r for r in card.rows}


def _toe_high(card):
    """(toe_mm, high_mm) of the "Impact Location" row (high is in its note)."""
    row = _rows(card)["impact_location"]
    assert row.value is not None, row.reason
    return row.value, float(row.note.split()[1])


def test_square_strike_through_the_face_centre_reads_zero_zero() -> None:
    series = _line_series([1.0, 0.0, 0.0])
    card = impact_card.card_from_series(series, 0.15, _placement(series))
    assert _toe_high(card) == pytest.approx((0.0, 0.0), abs=0.5)
    assert card.extras["ball_source"] == "address_geometry"


def test_contact_offset_toward_the_toe_axis_is_positive_toe() -> None:
    # _face_location: toe_mm = (contact - face_centre) . toe_axis, so a contact
    # 10 mm along +toe_axis reads +10 mm toe; high is along toe x normal.
    series = _line_series([1.0, 0.0, 0.0])
    toe = np.asarray(series.toe_axis[0])
    card = impact_card.card_from_series(
        series, 0.15, _placement(series, 0.010 * toe, "bundle:ball_impact")
    )
    assert _toe_high(card) == pytest.approx((10.0, 0.0), abs=0.5)
    assert card.extras["ball_source"] == "bundle:ball_impact"


def test_contact_offset_along_the_high_axis_is_high() -> None:
    series = _line_series([1.0, 0.0, 0.0])
    toe = np.asarray(series.toe_axis[0])
    high = np.cross(toe, np.asarray(series.face_normal[0]))
    card = impact_card.card_from_series(series, 0.15, _placement(series, 0.005 * high))
    assert _toe_high(card) == pytest.approx((0.0, 5.0), abs=0.5)


def test_contact_off_the_face_is_unavailable_with_the_measured_miss() -> None:
    # a contact point 80 mm toward the toe is beyond any club face: the model
    # face missed the ball, so no strike location is claimed
    series = _line_series([1.0, 0.0, 0.0])
    toe = np.asarray(series.toe_axis[0])
    card = impact_card.card_from_series(series, 0.15, _placement(series, 0.080 * toe))
    row = _rows(card)["impact_location"]
    assert row.value is None
    assert "off the face" in row.reason
    assert "toe +80" in row.reason


def test_no_ball_leaves_impact_location_unavailable_with_a_reason() -> None:
    card = impact_card.card_from_series(_line_series([1.0, 0.0, 0.0]), 0.15)
    row = _rows(card)["impact_location"]
    assert row.value is None
    assert "no ball placement supplied" in row.reason
    assert "GCV-11" not in row.reason
    assert "GCV-13" not in row.reason
    assert "ball_source" not in card.extras


def test_fixture_card_with_the_address_ball_records_its_source() -> None:
    import dataclasses

    from src.tools.native_viewer_export.ball import address_ball
    from src.tools.native_viewer_export.overlay import detect_impact_time_s

    swing = _fixture_swing()
    placed = address_ball(swing, None)
    if isinstance(placed, str):
        pytest.skip(placed)
    swing = dataclasses.replace(swing, ball=placed)
    card = impact_card.impact_card_for(swing, detect_impact_time_s(swing), swing.ball)
    assert isinstance(card, ImpactCard), card
    # the committed fixture face passes the address ball far from its centre
    # (#12223), so the location is either a measured on-face strike or an
    # off-face reason -- never an off-face number
    row = _rows(card)["impact_location"]
    assert row.value is not None or "off the face" in row.reason
    assert card.extras["ball_source"] == placed.source
    without = impact_card.impact_card_for(swing, detect_impact_time_s(swing))
    assert _rows(without)["impact_location"].value is None


def test_fixture_series_uses_the_true_face() -> None:
    swing = _fixture_swing()
    series = impact_card._club_series(swing)
    assert len(series) == swing.n_states
    assert np.isfinite(series.velocity_mps).all()
    assert np.allclose(np.linalg.norm(np.asarray(series.face_normal), axis=1), 1.0)

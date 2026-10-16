"""Decorative golf ball in native exports (GCV-13, #11719)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.shared.python.model_appearance.ball import BALL_RADIUS_M
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export.backends import _ball
from src.tools.native_viewer_export.backends._worker_job import WorkerJob
from src.tools.native_viewer_export.ball import (
    SOURCE_ADDRESS,
    SOURCE_BUNDLE,
    BallPlacement,
    address_ball,
    attach_ball,
)
from src.tools.native_viewer_export.core import ExportSettings, SwingInput
from tests.support.real_engines import real_module

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURES = Path(__file__).resolve().parents[4] / "tests/fixtures/club_face"


def _placement(hide: float | None = 1.0) -> BallPlacement:
    return BallPlacement((0.5, 0.1, 0.02), BALL_RADIUS_M, "test", hide)


def _fixture_swing(club: str = "driver", spec: dict[str, Any] | None = None):
    spec_doc = spec or json.loads((FIXTURES / f"spec_{club}.json").read_text())
    with np.load(FIXTURES / f"swing_q_{club}.npz") as data:
        q = np.asarray(data["q"], dtype=float)[:6]
    names = tuple(spec_doc["coordinate_order"])
    bundle = InputBundle(
        spec_bytes=json.dumps(spec_doc).encode(),
        coordinate_order=names,
        dt_s=0.001,
        q0=q[0],
        v0=np.zeros(len(names)),
        efforts=np.zeros((len(q) - 1, len(names))),
        reference_q=q,
        reference_v=np.zeros_like(q),
        reference_engine="mujoco",
    )
    return SwingInput(bundle, q, "t", club, "mujoco")


# --- BallPlacement -----------------------------------------------------------


def test_visible_before_at_and_after_hide_time() -> None:
    ball = _placement(1.0)
    assert ball.visible_at(0.0)
    assert ball.visible_at(1.0)
    assert not ball.visible_at(1.0001)


def test_visible_always_without_impact() -> None:
    assert _placement(None).visible_at(1e9)


def test_dict_round_trip() -> None:
    ball = _placement(0.75)
    assert BallPlacement.from_dict(json.loads(json.dumps(ball.to_dict()))) == ball


@pytest.mark.parametrize(
    ("position", "radius"),
    [
        ((0.0, 0.0), 0.02),
        ((0.0, math.nan, 0.0), 0.02),
        ((0.0, 0.0, math.inf), 0.02),
        ((0.0, 0.0, 0.0), 0.0),
        ((0.0, 0.0, 0.0), -0.01),
        ((0.0, 0.0, 0.0), math.nan),
    ],
)
def test_invalid_placement_is_rejected(position: Any, radius: float) -> None:
    with pytest.raises(ValueError):
        BallPlacement(position, radius, "x")


def test_non_finite_hide_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="hide_after_s"):
        BallPlacement((0.0, 0.0, 0.0), 0.02, "x", math.nan)


# --- address_ball ------------------------------------------------------------


def test_prefers_address_geometry_over_bundle_ball_impact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The capture's recorded ball floats at face height; never draw it (#12955)."""
    pytest.importorskip("mujoco")
    swing = _fixture_swing()
    monkeypatch.setattr(
        InputBundle,
        "ball_impact",
        lambda self: SimpleNamespace(ball_centre_m=[0.31, -0.12, 0.0213]),
    )
    ball = address_ball(swing, 0.9)
    assert isinstance(ball, BallPlacement)
    assert ball.source == SOURCE_ADDRESS
    assert ball.hide_after_s == 0.9


def test_falls_back_to_bundle_ball_impact_without_address_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    swing = _fixture_swing()
    centre = [0.31, -0.12, 0.0213]
    monkeypatch.setattr(
        "src.tools.native_viewer_export.ball._address_geometry_ball",
        lambda s: "no address stance",
    )
    monkeypatch.setattr(
        InputBundle, "ball_impact", lambda self: SimpleNamespace(ball_centre_m=centre)
    )
    ball = address_ball(swing, 0.9)
    assert isinstance(ball, BallPlacement)
    assert ball.source == SOURCE_BUNDLE
    assert ball.position_m == tuple(centre)
    assert ball.hide_after_s == 0.9


@pytest.mark.unit
@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_ball_sits_at_every_clubs_face_and_on_the_ground(
    club: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#12955: ball within one radius of the face centre's ground projection."""
    pytest.importorskip("mujoco")
    from src.tools.native_viewer_export.club_pose import ClubPoseModel

    swing = _fixture_swing(club)
    # a stale recorded capture ball far from the face must not be drawn
    monkeypatch.setattr(
        InputBundle,
        "ball_impact",
        lambda self: SimpleNamespace(ball_centre_m=[0.5, 0.6, 0.0208]),
    )
    model = ClubPoseModel(swing.bundle.spec_bytes)
    coords = dict(zip(swing.bundle.coordinate_order, swing.q[0], strict=True))
    face = model.face_world(coords)
    ball = address_ball(swing, None)
    assert isinstance(ball, BallPlacement), ball
    horizontal = np.asarray(ball.position_m[:2]) - face.centre_m[:2]
    assert np.hypot(*horizontal) <= BALL_RADIUS_M
    assert ball.position_m[2] == pytest.approx(
        model.ground_height_m + BALL_RADIUS_M, abs=1e-9
    )


@pytest.mark.unit
def test_landmark_helper_requires_a_face_centre() -> None:
    from src.shared.python.model_appearance.ball import address_ball_from_landmarks

    with pytest.raises(ValueError, match="face-centre landmark"):
        address_ball_from_landmarks(None, [0.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="face-centre landmark"):
        address_ball_from_landmarks([0.0, 0.0, 0.1], None)
    ball = address_ball_from_landmarks(
        [0.1, 0.2, 0.09], [0.0, -1.0, 0.0], ground_height_m=0.0
    )
    assert ball == pytest.approx([0.1, 0.2 - BALL_RADIUS_M, BALL_RADIUS_M])


@pytest.mark.unit
def test_mujoco_scene_has_no_second_decorative_ball() -> None:
    """The exporter's static-pose ball duplicated the worker's (#12955)."""
    pytest.importorskip("mujoco")
    from src.tools.native_viewer_export.backends.mujoco_worker import build_scene_xml

    swing = _fixture_swing("iron7")
    xml, summary = build_scene_xml(swing.bundle.spec_bytes)
    assert "visual_ball" not in xml


def test_falls_back_to_address_geometry() -> None:
    pytest.importorskip("mujoco")
    swing = _fixture_swing()
    ball = address_ball(swing, None)
    assert isinstance(ball, BallPlacement), ball
    assert ball.source == SOURCE_ADDRESS
    assert ball.hide_after_s is None
    spec = json.loads(swing.bundle.spec_bytes)
    ground = float(spec["contact"]["ground"]["height_m"])
    assert ball.position_m[2] == pytest.approx(ground + BALL_RADIUS_M, abs=1e-9)


def test_address_ball_touches_the_face_centre() -> None:
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.overlay_source import (
        MujocoOverlaySource,
    )
    from src.shared.python.model_appearance.club_assembly import (
        assembly_from_spec,
        clubface_centre,
        clubface_vector,
    )

    swing = _fixture_swing()
    spec = json.loads(swing.bundle.spec_bytes)
    frame = next(f for f in spec["frames"] if f["name"] == "Clubhead")
    coords = dict(zip(swing.bundle.coordinate_order, swing.q[0], strict=True))
    site = MujocoOverlaySource(swing.bundle.spec_bytes).frame_poses(coords)
    body = site["Clubhead"] @ np.linalg.inv(np.asarray(frame["placement"]))
    club = assembly_from_spec(spec)
    centre = body[:3, :3] @ clubface_centre(club) + body[:3, 3]
    normal = body[:3, :3] @ clubface_vector(club)
    normal /= np.linalg.norm(normal)
    ball = address_ball(swing, None)
    assert isinstance(ball, BallPlacement), ball
    offset = np.asarray(ball.position_m) - centre
    # horizontal offset follows the face normal; the ball rests on the ground
    horizontal = np.hypot(*normal[:2])
    assert np.hypot(*offset[:2]) == pytest.approx(BALL_RADIUS_M * horizontal, abs=1e-3)
    assert np.dot(offset[:2], normal[:2]) > 0.0  # the ball is in front of the face


def test_reason_string_without_club() -> None:
    swing = _fixture_swing()
    spec = json.loads(swing.bundle.spec_bytes)
    spec["bodies"] = [b for b in spec["bodies"] if "Club" not in b["name"]]
    spec["frames"] = [
        f for f in spec["frames"] if f["name"] not in ("Clubhead", "Grip")
    ]
    stripped = SwingInput(
        InputBundle(
            spec_bytes=json.dumps(spec).encode(),
            coordinate_order=swing.bundle.coordinate_order,
            dt_s=swing.bundle.dt_s,
            q0=swing.bundle.q0,
            v0=swing.bundle.v0,
            efforts=swing.bundle.efforts,
            reference_q=swing.bundle.reference_q,
            reference_v=swing.bundle.reference_v,
            reference_engine="mujoco",
        ),
        swing.q,
        "t",
        "Driver",
        "mujoco",
    )
    result = address_ball(stripped, None)
    assert isinstance(result, str) and result


def test_address_ball_rejects_bad_arguments() -> None:
    with pytest.raises(TypeError):
        address_ball(object(), None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        address_ball(_fixture_swing(), math.nan)


def test_attach_ball_records_placement_or_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    swing = _fixture_swing()
    off, record = attach_ball(swing, ExportSettings(ball=False))
    assert off.ball is None and record is None
    monkeypatch.setattr(
        "src.tools.native_viewer_export.ball.address_ball", lambda s, t: "no club"
    )
    unavailable, record = attach_ball(swing, ExportSettings())
    assert unavailable.ball is None and record == {"unavailable": "no club"}
    placed = _placement()
    kept, record = attach_ball(
        SwingInput(**{**swing.__dict__, "ball": placed}), ExportSettings()
    )
    assert kept.ball == placed and record == placed.to_dict()


def test_attach_ball_records_backend_limitation() -> None:
    swing = SwingInput(**{**_fixture_swing().__dict__, "ball": _placement()})
    note = "ball stays visible after impact in this viewer"
    _, record = attach_ball(swing, ExportSettings(), note)
    assert record is not None and record["limitation"] == note
    _, plain = attach_ball(swing, ExportSettings())
    assert plain is not None and "limitation" not in plain


# --- worker plumbing ---------------------------------------------------------


def _job(**extra: Any) -> WorkerJob:
    return WorkerJob("b", "q", [0, 1], ["v"], 8, 8, [0, 0, 0], None, "out", **extra)


def test_worker_job_round_trip_with_ball(tmp_path: Path) -> None:
    ball = _placement().to_dict()
    job = _job(ball=ball, ball_visible=[True, False])
    job.dump(tmp_path / "job.json")
    loaded = WorkerJob.load(tmp_path / "job.json")
    assert loaded.ball == ball and loaded.ball_visible == [True, False]


def test_worker_job_without_ball_field_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    doc = json.loads(json.dumps(_job().__dict__))
    doc.pop("ball"), doc.pop("ball_visible")
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = WorkerJob.load(path)
    assert loaded.ball is None and loaded.ball_visible is None


def test_ball_visibility_follows_swing_time() -> None:
    swing = _fixture_swing()
    swing = SwingInput(**{**swing.__dict__, "ball": _placement(0.002)})
    assert _ball.ball_visibility(swing, [0, 1, 2, 3]) == [True, True, True, False]
    bare = SwingInput(**{**swing.__dict__, "ball": None})
    assert _ball.ball_visibility(bare, [0]) is None


# --- MuJoCo scene helper -----------------------------------------------------


def _scene():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    return mujoco, model, mujoco.MjvScene(model, maxgeom=20)


def test_scene_helper_adds_exactly_one_sphere() -> None:
    mujoco, _, scene = _scene()
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (
        add_ball_to_scene,
    )

    assert add_ball_to_scene(scene, [0.5, 0.1, 0.02], BALL_RADIUS_M)
    assert scene.ngeom == 1
    geom = scene.geoms[0]
    assert int(geom.type) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    assert geom.size[0] == pytest.approx(BALL_RADIUS_M)
    np.testing.assert_allclose(geom.pos, [0.5, 0.1, 0.02])
    with pytest.raises(ValueError):
        add_ball_to_scene(scene, [0.0, 0.0], BALL_RADIUS_M)


def test_draw_job_ball_only_when_visible() -> None:
    _, _, scene = _scene()
    job = _job(ball=_placement().to_dict(), ball_visible=[True, False])
    assert _ball.draw_job_ball(scene, job, 0) and scene.ngeom == 1
    assert not _ball.draw_job_ball(scene, job, 1) and scene.ngeom == 1
    assert not _ball.draw_job_ball(scene, _job(), 0) and scene.ngeom == 1


def test_render_scene_leaves_model_dynamics_unchanged() -> None:
    mujoco, model, scene = _scene()
    mass, nbody = float(model.body_subtreemass[0]), model.nbody
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (
        add_ball_to_scene,
    )

    add_ball_to_scene(scene, [0.0, 0.0, 0.02], BALL_RADIUS_M)
    assert model.nbody == nbody and model.ngeom == 0
    assert float(model.body_subtreemass[0]) == mass


# --- MeshCat helper ----------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def set_sphere(self, path: str, position: Any, radius: float, rgba: Any) -> None:
        self.calls.append(("sphere", path, position, radius))

    def set_visible(self, path: str, visible: bool) -> None:
        self.calls.append(("visible", path, visible))


def test_meshcat_ball_sets_sphere_once_and_toggles_visibility() -> None:
    sink = _Recorder()
    ball = _ball.MeshcatBall(sink, _placement(0.5))
    shown = [ball.update(t) for t in (0.0, 0.5, 0.6)]
    assert shown == [True, True, False]
    assert sum(c[0] == "sphere" for c in sink.calls) == 1
    assert sink.calls[0] == ("sphere", "/ball", (0.5, 0.1, 0.02), BALL_RADIUS_M)
    assert [c[2] for c in sink.calls if c[0] == "visible"] == [True, True, False]


def test_drake_sink_uses_meshcat_object_and_visible_property() -> None:
    real_module("pydrake")
    meshcat = SimpleNamespace(
        SetObject=lambda *a: calls.append(("object", a[0])),
        SetTransform=lambda *a: calls.append(("transform", a[0])),
        SetProperty=lambda *a: calls.append(("property", a)),
    )
    calls: list[tuple[Any, ...]] = []
    sink = _ball.DrakeBallSink(meshcat)
    sink.set_sphere("/ball", (0.0, 0.0, 0.02), 0.02, (1, 1, 1, 1))
    sink.set_visible("/ball", False)
    assert calls == [
        ("object", "/ball"),
        ("transform", "/ball"),
        ("property", ("/ball", "visible", False)),
    ]

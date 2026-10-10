"""Decorative address ball for the MeshCat native exports (GCV-13, #11719)."""

from __future__ import annotations

from dataclasses import replace
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.model_appearance.ball import (
    BALL_RADIUS_M,
    ball_position_at_address,
)
from src.shared.python.model_appearance.club_assembly import (
    assembly_from_spec,
    club_body_name,
    clubface_centre,
    clubface_vector,
)
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export.ball import (
    MAX_ADDRESS_HEIGHT_M,
    resolve_address_ball,
)
from src.tools.native_viewer_export.core import SwingInput

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
SWING_Q_PATH = ROOT / "tests/fixtures/club_face/swing_q_driver.npz"


def _real_driver_swing() -> SwingInput:
    """A real capture-A driver swing; ``q[0]`` is its address frame."""
    pytest.importorskip("mujoco")
    spec_bytes = SPEC_PATH.read_bytes()
    order = tuple(json.loads(spec_bytes)["coordinate_order"])
    q = np.load(SWING_Q_PATH)["q"].astype(float)
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=order,
        dt_s=0.002,
        q0=q[0],
        v0=np.zeros_like(q[0]),
        efforts=np.zeros((len(q) - 1, len(order))),
        reference_q=q,
        reference_v=np.zeros_like(q),
        reference_engine="mujoco",
    )
    return SwingInput(bundle, q, "driver_cap_a", "Driver", "mujoco")


def _patch_mujoco_overlay_source(
    monkeypatch: pytest.MonkeyPatch, clubhead_pose: np.ndarray
) -> None:
    """Replace the native export's MuJoCo FK source with a fixed ``Clubhead`` pose."""

    class _Source:
        def __init__(self, _spec_bytes: bytes) -> None:
            pass

        def frame_poses(self, _coordinates: dict) -> dict:
            return {"Clubhead": clubhead_pose}

    mod = types.ModuleType("overlay_source_stub")
    mod.MujocoOverlaySource = _Source  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "src.engines.physics_engines.mujoco.python.overlay_source",
        mod,
    )


def _stubbed_club_swing() -> Any:
    """A minimal synthetic spec with just a driver club body, no full skeleton."""
    spec = {
        "gravity_m_s2": [0.0, 0.0, -9.81],
        "bodies": [
            {
                "name": "Right Hand/Clubface Vector",
                "solids": [
                    {
                        "name": "Grip",
                        "placement": [
                            [1, 0, 0, 0],
                            [0, 1, 0, -1.0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        "com_m": [0.0, 0.0, 0.0],
                        "mass_kg": 0.05,
                    },
                ],
            }
        ],
        "joints": [
            {
                "name": "club_joint",
                "parent": "world",
                "child": "Right Hand/Clubface Vector",
                "child_to_follower": np.eye(4).tolist(),
                "parent_to_base": np.eye(4).tolist(),
            }
        ],
        "coordinate_order": [],
    }
    bundle = types.SimpleNamespace(
        coordinate_order=(), spec_bytes=json.dumps(spec).encode("utf-8")
    )
    return types.SimpleNamespace(bundle=bundle, q=np.zeros((1, 0)))


def _patch_height_dependent_clubhead(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clubhead frame whose world height equals the single ``h`` coordinate."""

    class _Source:
        def __init__(self, _spec_bytes: bytes) -> None:
            pass

        def frame_poses(self, coordinates: dict) -> dict:
            pose = np.eye(4)
            pose[2, 3] = coordinates["h"]
            return {"Clubhead": pose}

    mod = types.ModuleType("overlay_source_stub_height")
    mod.MujocoOverlaySource = _Source  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "src.engines.physics_engines.mujoco.python.overlay_source",
        mod,
    )


def _grounded_then_elevated_swing(
    n: int = 50, dt_s: float = 0.01, elevated_from: int = 20, elevated_h: float = 1.9
) -> SwingInput:
    """A real (loadable, multi-state) swing bundle: frame 0 is grounded, and
    every frame from ``elevated_from`` on is ``elevated_h`` metres up.

    Shares the club body of :func:`_stubbed_club_swing`, but as a genuine
    ``InputBundle`` so the production clip/windowing pipeline
    (``core._resampled``, ``video_timing.FrameSchedule``) can slice it --
    the GCV-13 impact-window bug (#11719) only reproduces through that real
    windowing, not a hand-built single-state swing.
    """
    spec = {
        "gravity_m_s2": [0.0, 0.0, -9.81],
        "coordinate_order": ["h"],
        "bodies": [
            {
                "name": "Right Hand/Clubface Vector",
                "solids": [
                    {
                        "name": "Grip",
                        "placement": [
                            [1, 0, 0, 0],
                            [0, 1, 0, -1.0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        "com_m": [0.0, 0.0, 0.0],
                        "mass_kg": 0.05,
                    },
                ],
            }
        ],
        "joints": [
            {
                "name": "club_joint",
                "parent": "world",
                "child": "Right Hand/Clubface Vector",
                "child_to_follower": np.eye(4).tolist(),
                "parent_to_base": np.eye(4).tolist(),
            }
        ],
    }
    spec_bytes = json.dumps(spec).encode("utf-8")
    q = np.zeros((n, 1))
    q[elevated_from:, 0] = elevated_h
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=("h",),
        dt_s=dt_s,
        q0=q[0],
        v0=np.zeros(1),
        efforts=np.zeros((n - 1, 1)),
        reference_q=q,
        reference_v=np.zeros_like(q),
        reference_engine="mujoco",
    )
    return SwingInput(bundle, q, "evidence", "Driver", "mujoco")


def test_resolves_the_ball_from_the_real_driver_address_frame() -> None:
    """Wiring check: the resolver composes club_assembly with the Clubhead FK pose."""
    swing = _real_driver_swing()
    resolved = resolve_address_ball(swing)
    assert resolved.reason is None
    assert resolved.source == "address_geometry"
    assert resolved.radius_m == pytest.approx(BALL_RADIUS_M)

    spec = json.loads(swing.bundle.spec_bytes)
    club = assembly_from_spec(spec)
    assert club is not None
    from src.engines.physics_engines.mujoco.python.overlay_source import (
        MujocoOverlaySource,
    )

    source = MujocoOverlaySource(swing.bundle.spec_bytes)
    names = swing.bundle.coordinate_order
    coords = dict(zip(names, map(float, swing.q[0]), strict=True))
    frame = source.frame_poses(coords)["Clubhead"]
    centre = frame[:3, :3] @ clubface_centre(club) + frame[:3, 3]
    normal = frame[:3, :3] @ clubface_vector(club)
    expected = ball_position_at_address(centre, normal, ground_height_m=0.0)
    assert resolved.position_m == pytest.approx(expected)
    # The address face rests on the ground within the grounding tolerance.
    assert abs(float(centre[2])) <= MAX_ADDRESS_HEIGHT_M


def test_ball_rests_on_the_ground_and_touches_the_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grounded, controlled address frame resolves to the shared placement rule."""
    pose = np.eye(4)
    _patch_mujoco_overlay_source(monkeypatch, pose)
    swing = _stubbed_club_swing()
    spec = json.loads(swing.bundle.spec_bytes)
    club = assembly_from_spec(spec)
    assert club is not None
    centre, normal = clubface_centre(club), clubface_vector(club)

    resolved = resolve_address_ball(swing)

    assert resolved.reason is None
    expected = ball_position_at_address(centre, normal, ground_height_m=0.0)
    assert resolved.position_m == pytest.approx(expected)
    assert resolved.position_m[2] == pytest.approx(BALL_RADIUS_M)


def test_unavailable_when_the_address_clubhead_is_off_the_ground(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pose = np.eye(4)
    pose[2, 3] = 0.5  # far above the ground: not a plausible address frame
    _patch_mujoco_overlay_source(monkeypatch, pose)
    swing = _stubbed_club_swing()

    resolved = resolve_address_ball(swing)

    assert resolved.position_m is None
    assert "not grounded" in resolved.reason


def test_unavailable_when_the_spec_has_no_club_body() -> None:
    swing = _real_driver_swing()
    spec = json.loads(swing.bundle.spec_bytes)
    spec["bodies"] = [b for b in spec["bodies"] if "Clubface" not in b["name"]]
    stripped_bundle = replace(swing.bundle, spec_bytes=json.dumps(spec).encode("utf-8"))
    swing = replace(swing, bundle=stripped_bundle)

    resolved = resolve_address_ball(swing)

    assert resolved.position_m is None
    assert resolved.reason == "spec has no club body"


def test_resolving_the_ball_does_not_mutate_swing_state() -> None:
    """The ball resolver is read-only: it never changes the swing's physics state."""
    swing = _real_driver_swing()
    before = swing.q.copy()

    resolve_address_ball(swing)

    assert np.array_equal(swing.q, before)


def test_address_ball_requires_exactly_one_of_position_and_reason() -> None:
    from src.tools.native_viewer_export.ball import AddressBall

    with pytest.raises(ValueError, match="exactly one"):
        AddressBall(np.zeros(3), BALL_RADIUS_M, "address_geometry", reason="x")
    with pytest.raises(ValueError, match="exactly one"):
        AddressBall(None, BALL_RADIUS_M, None, reason=None)


def test_export_settings_draws_the_ball_by_default() -> None:
    from src.tools.native_viewer_export.core import ExportSettings

    assert ExportSettings().ball is True


def test_no_ball_cli_flag_disables_the_decorative_ball() -> None:
    from src.tools.native_viewer_export import cli

    args = cli.build_parser().parse_args(
        ["--bundle", "b.npz", "--out", "o", "--swing", "s", "--no-ball"]
    )
    settings = cli.build_settings(args)
    assert settings.ball is False

    default_args = cli.build_parser().parse_args(
        ["--bundle", "b.npz", "--out", "o", "--swing", "s"]
    )
    assert cli.build_settings(default_args).ball is True


def _real_module(name: str) -> types.ModuleType:
    """Import ``name`` or skip; a mock another test left in ``sys.modules`` skips too.

    Mirrors ``tests/unit/model_appearance/test_clubface_square_at_address.py``'s
    helper of the same name.
    """
    module = pytest.importorskip(name)
    if not isinstance(module, types.ModuleType) or not hasattr(module, "__file__"):
        pytest.skip(f"{name} in sys.modules is a test double, not the real package")
    return module


def test_drake_meshcat_scene_has_a_ball_at_the_resolved_address_position() -> None:
    _real_module("pydrake")
    from pydrake.geometry import Sphere

    from src.tools.native_viewer_export.backends.drake_meshcat import (
        DrakeMeshcatBackend,
    )

    swing = _real_driver_swing()
    resolved = resolve_address_ball(swing)
    assert resolved.position_m is not None

    plant, diagram, _ctx, _pctx, _meshcat, _starts = DrakeMeshcatBackend()._build(
        swing, ball=resolved
    )
    scene_graph = diagram.GetSubsystemByName("scene_graph")
    inspector = scene_graph.model_inspector()
    ball_ids = [
        gid
        for gid in inspector.GetAllGeometryIds()
        if inspector.GetName(gid).endswith("visual_ball")
    ]
    assert len(ball_ids) == 1
    shape = inspector.GetShape(ball_ids[0])
    assert isinstance(shape, Sphere)
    assert shape.radius() == pytest.approx(resolved.radius_m)
    pose = inspector.GetPoseInFrame(ball_ids[0])
    assert np.allclose(pose.translation(), resolved.position_m, atol=1e-6)


def test_drake_meshcat_scene_skips_the_ball_when_disabled() -> None:
    _real_module("pydrake")
    from src.tools.native_viewer_export.backends.drake_meshcat import (
        DrakeMeshcatBackend,
    )

    swing = _real_driver_swing()
    plant, diagram, _ctx, _pctx, _meshcat, _starts = DrakeMeshcatBackend()._build(
        swing, ball=None
    )
    scene_graph = diagram.GetSubsystemByName("scene_graph")
    inspector = scene_graph.model_inspector()
    assert not any(
        inspector.GetName(gid).endswith("visual_ball")
        for gid in inspector.GetAllGeometryIds()
    )


def test_pinocchio_meshcat_scene_has_a_ball_at_the_resolved_address_position() -> None:
    _real_module("pinocchio")
    coal = pytest.importorskip("coal")
    pytest.importorskip("meshcat")
    from src.tools.native_viewer_export.backends.pinocchio_meshcat import (
        PinocchioMeshcatBackend,
    )

    swing = _real_driver_swing()
    resolved = resolve_address_ball(swing)
    assert resolved.position_m is not None

    _adapter, viz, _meshcat = PinocchioMeshcatBackend()._build(swing, ball=resolved)
    balls = [o for o in viz.visual_model.geometryObjects if o.name == "visual_ball"]
    assert len(balls) == 1
    obj = balls[0]
    assert isinstance(obj.geometry, coal.Sphere)
    assert obj.geometry.radius == pytest.approx(resolved.radius_m)
    assert np.allclose(obj.placement.translation, resolved.position_m, atol=1e-6)


class _RecordingBackend:
    """Fake backend recording the ``swing``/``ball`` each clip render saw."""

    engine = "drake"

    def __init__(self) -> None:
        self.calls: list[tuple[SwingInput, Any]] = []

    def unavailable_reason(self) -> str | None:
        return None

    def render(self, swing, settings, indices, overlay, ball=None):
        self.calls.append((swing, ball))
        for _ in indices:
            yield {
                v: np.zeros((settings.height, settings.width, 3), np.uint8)
                for v in settings.views
            }


class _Writer:
    def __init__(self, path: Path, fps: int) -> None:
        self.path = path

    def append_data(self, frame: object) -> None:
        pass

    def close(self) -> None:
        self.path.write_bytes(b"x")


def test_impact_window_clip_gets_the_ball_from_the_true_address_not_the_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GCV-13 bug (#11719): an impact-window clip's own first frame is
    mid-swing (elevated here), not the address; the ball must still resolve
    from the swing's true frame 0, at the same position a full-speed clip
    would use, instead of being wrongly reported as not grounded."""
    from src.tools.native_viewer_export.core import ExportSettings
    from src.tools.native_viewer_export.runner import ExportJob, run_export

    _patch_height_dependent_clubhead(monkeypatch)
    swing = _grounded_then_elevated_swing()
    bpath = tmp_path / "b.npz"
    swing.bundle.save(bpath)
    expected = resolve_address_ball(swing)
    assert expected.position_m is not None  # the true address frame is grounded

    backend = _RecordingBackend()
    job = ExportJob(bpath, tmp_path / "out", "evidence", "Driver", ("drake",))
    settings = ExportSettings(
        width=32,
        height=32,
        fps=20,
        speeds=(),
        views=("face_on",),
        multiview=False,
        impact_time_s=0.3,
        impact_window_s=0.1,
    )
    run_export(
        job,
        settings,
        lambda engine: backend,
        lambda s, engine: (None, (0.5, 0.0, 0.9)),
        _Writer,
    )

    assert len(backend.calls) == 1  # only the impact-window clip (speeds=())
    windowed_swing, ball = backend.calls[0]
    # The clip's own first frame is mid-swing (elevated), not the address...
    assert windowed_swing.q[0, 0] == pytest.approx(1.9)
    # ...but the ball still resolves, at the true address position.
    assert ball is not None and ball.position_m is not None
    assert ball.position_m == pytest.approx(expected.position_m)

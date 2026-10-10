"""Decorative address ball in the MuJoCo, OpenSim and MyoSuite native-export
backends (GCV-13 slice 3, #11719).

Mirrors ``test_address_ball.py``'s fake/monkeypatch idiom: no engine SDK is
required for the backend-level tests because each backend's own
``render_in_worker`` call is monkeypatched, so these exercise the plumbing
(the already-resolved ``AddressBall`` threaded through, a warning logged when
it is unresolved) without needing mujoco, opensim or myosuite installed. The
MJCF geom helper (``backends._ball``) is tested directly against a minimal
parsed document; the physics-identity and OpenSim ground-attach tests need
the real ``mujoco``/``opensim`` packages and skip cleanly without them.
"""

from __future__ import annotations

import logging

import defusedxml.ElementTree as ET

import numpy as np
import pytest

from src.shared.python.model_appearance.ball import BALL_RADIUS_M
from src.tools.native_viewer_export.ball import AddressBall

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _resolved_ball() -> AddressBall:
    return AddressBall(
        np.array([1.0, 0.0, BALL_RADIUS_M]), BALL_RADIUS_M, "address_geometry"
    )


def _unavailable_ball() -> AddressBall:
    return AddressBall(None, BALL_RADIUS_M, None, reason="spec has no club body")


# --- backend-level plumbing: MuJoCo, MyoSuite, OpenSim ---------------------


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        (
            "src.tools.native_viewer_export.backends.mujoco_native",
            "MuJoCoRendererBackend",
        ),
        (
            "src.tools.native_viewer_export.backends.myosuite_arena",
            "MyoSuiteArenaBackend",
        ),
        (
            "src.tools.native_viewer_export.backends.opensim_simbody",
            "OpenSimSimbodyBackend",
        ),
    ],
)
def test_backend_threads_the_resolved_ball_to_the_worker(
    module_name: str, class_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    backend_cls = getattr(module, class_name)
    calls: list[AddressBall | None] = []

    def fake_render_in_worker(*args, **kwargs):
        ball = kwargs.get("ball", args[6] if len(args) > 6 else None)
        calls.append(ball)
        return iter(())

    monkeypatch.setattr(module, "render_in_worker", fake_render_in_worker)
    if class_name == "MyoSuiteArenaBackend":
        monkeypatch.setattr(module, "myosuite_python", lambda: "python")

    ball = _resolved_ball()
    list(backend_cls().render(object(), object(), [0], None, ball))

    # Identity, not equality: AddressBall.position_m is a numpy array, so
    # dataclass equality would raise on the elementwise comparison.
    assert len(calls) == 1
    assert calls[0] is ball


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        (
            "src.tools.native_viewer_export.backends.mujoco_native",
            "MuJoCoRendererBackend",
        ),
        (
            "src.tools.native_viewer_export.backends.myosuite_arena",
            "MyoSuiteArenaBackend",
        ),
        (
            "src.tools.native_viewer_export.backends.opensim_simbody",
            "OpenSimSimbodyBackend",
        ),
    ],
)
def test_backend_logs_a_warning_and_still_renders_when_the_ball_is_unresolved(
    module_name: str,
    class_name: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    backend_cls = getattr(module, class_name)
    calls: list[AddressBall | None] = []

    def fake_render_in_worker(*args, **kwargs):
        ball = kwargs.get("ball", args[6] if len(args) > 6 else None)
        calls.append(ball)
        return iter(())

    monkeypatch.setattr(module, "render_in_worker", fake_render_in_worker)
    if class_name == "MyoSuiteArenaBackend":
        monkeypatch.setattr(module, "myosuite_python", lambda: "python")

    ball = _unavailable_ball()
    with caplog.at_level(logging.WARNING):
        list(backend_cls().render(object(), object(), [0], None, ball))

    # the worker, not the backend, decides to skip the geom
    assert len(calls) == 1
    assert calls[0] is ball
    assert "spec has no club body" in caplog.text


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        (
            "src.tools.native_viewer_export.backends.mujoco_native",
            "MuJoCoRendererBackend",
        ),
        (
            "src.tools.native_viewer_export.backends.myosuite_arena",
            "MyoSuiteArenaBackend",
        ),
        (
            "src.tools.native_viewer_export.backends.opensim_simbody",
            "OpenSimSimbodyBackend",
        ),
    ],
)
def test_backend_passes_no_ball_through_without_warning_when_the_feature_is_off(
    module_name: str,
    class_name: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    backend_cls = getattr(module, class_name)
    calls: list[AddressBall | None] = []

    def fake_render_in_worker(*args, **kwargs):
        ball = kwargs.get("ball", args[6] if len(args) > 6 else None)
        calls.append(ball)
        return iter(())

    monkeypatch.setattr(module, "render_in_worker", fake_render_in_worker)
    if class_name == "MyoSuiteArenaBackend":
        monkeypatch.setattr(module, "myosuite_python", lambda: "python")

    with caplog.at_level(logging.WARNING):
        list(backend_cls().render(object(), object(), [0], None, None))

    assert calls == [None]
    assert "skipping decorative ball" not in caplog.text


# --- _ball.py: the shared MJCF geom helper ----------------------------------


def _root() -> ET.Element:
    return ET.fromstring("<mujoco><worldbody/></mujoco>")


def test_set_decorative_ball_adds_a_massless_noncolliding_sphere() -> None:
    from src.tools.native_viewer_export.backends._ball import set_decorative_ball

    root = _root()
    set_decorative_ball(root, (1.0, 0.0, 0.05))

    world = root.find("worldbody")
    assert world is not None
    geoms = [g for g in world.findall("geom") if g.get("name") == "visual_ball"]
    assert len(geoms) == 1
    geom = geoms[0]
    assert geom.get("type") == "sphere"
    assert geom.get("contype") == "0"
    assert geom.get("conaffinity") == "0"
    assert geom.get("mass") == "0"
    size = geom.get("size")
    assert size is not None
    assert float(size) == pytest.approx(BALL_RADIUS_M)
    pos = geom.get("pos")
    assert pos is not None
    assert [float(v) for v in pos.split()] == pytest.approx([1.0, 0.0, 0.05])


def test_set_decorative_ball_removes_any_existing_ball_when_position_is_none() -> None:
    from src.tools.native_viewer_export.backends._ball import set_decorative_ball

    root = _root()
    set_decorative_ball(root, (1.0, 0.0, 0.05))
    set_decorative_ball(root, None)

    world = root.find("worldbody")
    assert world is not None
    assert not any(g.get("name") == "visual_ball" for g in world.findall("geom"))


def test_set_decorative_ball_replaces_rather_than_duplicates() -> None:
    from src.tools.native_viewer_export.backends._ball import set_decorative_ball

    root = _root()
    set_decorative_ball(root, (1.0, 0.0, 0.05))
    set_decorative_ball(root, (2.0, 0.0, 0.05))

    world = root.find("worldbody")
    assert world is not None
    geoms = [g for g in world.findall("geom") if g.get("name") == "visual_ball"]
    assert len(geoms) == 1
    pos = geoms[0].get("pos")
    assert pos is not None
    assert float(pos.split()[0]) == pytest.approx(2.0)


def test_set_decorative_ball_requires_a_worldbody() -> None:
    from src.tools.native_viewer_export.backends._ball import set_decorative_ball

    root = ET.fromstring("<mujoco/>")
    with pytest.raises(ValueError, match="worldbody"):
        set_decorative_ball(root, (0.0, 0.0, 0.0))


# --- _subprocess.stage_job: ball position encoding --------------------------


def _minimal_swing():
    from src.shared.python.motion_matching.same_input import InputBundle
    from src.tools.native_viewer_export.core import SwingInput

    q = np.zeros((1, 1))
    bundle = InputBundle(
        spec_bytes=b'{"coordinate_order": ["a"]}',
        coordinate_order=("a",),
        dt_s=0.01,
        q0=q[0],
        v0=np.zeros(1),
        efforts=np.zeros((0, 1)),
        reference_q=q,
        reference_v=np.zeros_like(q),
        reference_engine="mujoco",
    )
    return SwingInput(bundle, q, "s", "Driver", "mujoco")


def test_stage_job_encodes_the_resolved_ball_position(tmp_path) -> None:
    from src.tools.native_viewer_export.backends._subprocess import stage_job
    from src.tools.native_viewer_export.core import ExportSettings

    job = stage_job(
        _minimal_swing(), ExportSettings(), [0], None, tmp_path, _resolved_ball()
    )

    assert job.ball_position_m == pytest.approx([1.0, 0.0, BALL_RADIUS_M])


@pytest.mark.parametrize("ball", [None, _unavailable_ball()])
def test_stage_job_omits_the_ball_position_when_unresolved_or_absent(
    ball: AddressBall | None, tmp_path
) -> None:
    from src.tools.native_viewer_export.backends._subprocess import stage_job
    from src.tools.native_viewer_export.core import ExportSettings

    job = stage_job(_minimal_swing(), ExportSettings(), [0], None, tmp_path, ball)

    assert job.ball_position_m is None


# --- MuJoCo physics identity: the real mujoco package ----------------------


def test_mujoco_ball_geom_is_visual_only_and_does_not_change_the_model() -> None:
    mujoco = pytest.importorskip("mujoco")
    from defusedxml import ElementTree as DET
    from pathlib import Path

    from src.tools.native_viewer_export.backends._ball import set_decorative_ball
    from src.tools.native_viewer_export.backends.mujoco_worker import build_scene_xml

    root_dir = Path(__file__).resolve().parents[4]
    spec_bytes = (
        root_dir / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    ).read_bytes()

    xml, _ = build_scene_xml(spec_bytes)
    model_without = mujoco.MjModel.from_xml_string(xml)

    root = DET.fromstring(xml)
    set_decorative_ball(root, (1.0, 0.0, BALL_RADIUS_M))
    xml_with_ball = DET.tostring(root, encoding="unicode")
    model_with = mujoco.MjModel.from_xml_string(xml_with_ball)

    assert model_with.nbody == model_without.nbody  # no new body: world-fixed geom
    assert model_with.ngeom == model_without.ngeom + 1
    ball_id = mujoco.mj_name2id(model_with, mujoco.mjtObj.mjOBJ_GEOM, "visual_ball")
    assert ball_id >= 0
    assert model_with.geom_contype[ball_id] == 0
    assert model_with.geom_conaffinity[ball_id] == 0


# --- OpenSim ground attach: the real opensim package ------------------------


def test_opensim_ball_attaches_to_ground_without_changing_body_count() -> None:
    osim = pytest.importorskip("opensim")
    from pathlib import Path

    from src.tools.native_viewer_export.backends.opensim_worker import build_model

    root_dir = Path(__file__).resolve().parents[4]
    spec_bytes = (
        root_dir / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    ).read_bytes()

    model_without, _ = build_model(osim, spec_bytes)
    n_bodies_without = model_without.getBodySet().getSize()
    n_components_without = len(tuple(model_without.getComponentsList()))

    model_with, _ = build_model(osim, spec_bytes, (1.0, 0.0, BALL_RADIUS_M))
    n_bodies_with = model_with.getBodySet().getSize()
    n_components_with = len(tuple(model_with.getComponentsList()))

    assert n_bodies_with == n_bodies_without  # visual only, no new body
    assert n_components_with == n_components_without + 1

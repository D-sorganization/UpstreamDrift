"""Segment shading beyond MuJoCo: name mapping, coverage and per-backend steps (#13146)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.force_overlay.contracts import ForceTorqueFrame
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export.backends._ball import update_meshcat_frame
from src.tools.native_viewer_export.backends.registry import SHADING_ENGINES
from src.tools.native_viewer_export.core import (
    ENGINES,
    ExportSettings,
    OverlayFeed,
)
from src.tools.native_viewer_export.runner import ExportJob, run_export
from src.tools.native_viewer_export.shading import (
    MeshcatShading,
    build_meshcat_shading,
    log_coverage,
    mujoco_scene_coverage,
    pinocchio_color_bindings,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

SCALE = ForceColorScale(
    enabled=True, opacity=1.0, blend_mode="replace", tension_limit_n=100.0
)
BASE = (0.5, 0.5, 0.5, 0.8)


def _item(name: str, body: str, rgba=BASE):
    return SimpleNamespace(name=name, body=body, rgba=rgba)


# --- name mapping --------------------------------------------------------


def test_pinocchio_bindings_key_by_spec_body_and_skip_world() -> None:
    items = [
        _item("000_a", "solid_reference:GolfSwing3D_Kinetic/LUpperArm"),
        _item("001_b", "solid_reference:GolfSwing3D_Kinetic/LUpperArm"),
        _item("floor", ""),
        _item("002_c", "femur_l"),
    ]
    got = pinocchio_color_bindings(items, "golfer")
    assert set(got) == {"solid_reference:GolfSwing3D_Kinetic/LUpperArm", "femur_l"}
    assert set(got["femur_l"]) == {"golfer/visuals/002_c/<object>"}
    assert len(got["solid_reference:GolfSwing3D_Kinetic/LUpperArm"]) == 2


# --- MeshCat shading -----------------------------------------------------


def _shading(bindings=None):
    calls: list[tuple[str, str, list[float]]] = []
    bindings = bindings or pinocchio_color_bindings(
        [_item("a", "seg_a"), _item("b", "seg_b")], "g"
    )
    shading = MeshcatShading(
        "pinocchio", lambda p, k, v: calls.append((p, k, v)), bindings, SCALE
    )
    return shading, calls


def test_meshcat_shading_uses_shared_blend_and_preserves_alpha() -> None:
    shading, calls = _shading()
    shading.update(AxialLoadFrame(0.0, {"seg_a": 100.0}, "t"))
    path, prop, value = calls[0]
    assert (path, prop) == ("g/visuals/a/<object>", "color")
    assert value[:3] == list(SCALE.shade_rgb(100.0, BASE[:3]))
    assert value[3] == BASE[3]
    assert len(calls) == 1  # seg_b has no load: stays unshaded
    assert shading.coverage() == (1, 2)


def test_meshcat_shading_restores_base_when_load_disappears() -> None:
    shading, calls = _shading()
    shading.update(AxialLoadFrame(0.0, {"seg_a": -50.0}, "t"))
    shading.update(None)
    assert calls[-1] == ("g/visuals/a/<object>", "color", list(BASE))


def test_log_summary_reports_shaded_over_total(caplog) -> None:
    shading, _ = _shading()
    shading.update(AxialLoadFrame(0.0, {"seg_a": 1.0, "seg_b": -1.0}, "t"))
    with caplog.at_level(logging.DEBUG):
        line = shading.log_summary()
    assert "2/2" in line and "pinocchio" in caplog.text
    with pytest.raises(ValueError):
        log_coverage("x", 3, 2)


def test_build_meshcat_shading_is_none_when_off_or_unbound() -> None:
    b = pinocchio_color_bindings([_item("a", "s")])
    noop = lambda p, k, v: None  # noqa: E731
    assert build_meshcat_shading("drake", None, noop, b) is None
    assert (
        build_meshcat_shading("drake", ForceColorScale(enabled=False), noop, b) is None
    )
    assert build_meshcat_shading("drake", SCALE, noop, {}) is None
    assert isinstance(build_meshcat_shading("drake", SCALE, noop, b), MeshcatShading)
    with pytest.raises(TypeError):
        MeshcatShading("drake", noop, b, "red")  # type: ignore[arg-type]


def test_update_meshcat_frame_feeds_the_axial_loads() -> None:
    shading, calls = _shading()
    frame = AxialLoadFrame(0.0, {"seg_b": 100.0}, "t")
    overlay = SimpleNamespace(axial_at=lambda i: frame, glyphs_at=lambda i: None)
    update_meshcat_frame(None, None, overlay, SimpleNamespace(), 3, shading)
    assert calls and calls[0][0] == "g/visuals/b/<object>"
    update_meshcat_frame(None, None, None, SimpleNamespace(), 3, shading)  # no-op


# --- MuJoCo-family scene step -------------------------------------------

XML = """<mujoco><worldbody>
<geom name="floor" type="plane" size="1 1 .1" rgba=".2 .2 .2 1"/>
<body name="seg_a" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" rgba=".5 .5 .5 1"/></body>
<body name="seg_b" pos="1 0 1"><freejoint/><geom type="sphere" size=".1" rgba=".5 .5 .5 1"/></body>
</worldbody></mujoco>"""


def _scene():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=50)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    return mujoco, model, scene


def test_scene_coverage_counts_geometry_bodies() -> None:
    _, model, _ = _scene()
    frames = [None, AxialLoadFrame(0.0, {"seg_a": 5.0, "seg_b": None, "zz": 1.0}, "t")]
    assert mujoco_scene_coverage(model, frames) == (1, 2)


def test_apply_scene_overlays_shades_then_adds_glyphs() -> None:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (
        force_glyphs,
    )
    from src.tools.native_viewer_export.backends._scene_overlays import (
        apply_scene_overlays,
    )

    mujoco, model, scene = _scene()
    frames = [AxialLoadFrame(0.0, {"seg_a": 100.0}, "t")]
    before = scene.ngeom
    ids = {
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g.objid]
        ): i
        for i, g in enumerate(scene.geoms[:before])
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM
    }
    expected = SCALE.shade_rgb(100.0, (0.5, 0.5, 0.5))
    apply_scene_overlays(model, scene, 0, frames, SCALE, None)
    assert tuple(scene.geoms[ids["seg_a"]].rgba[:3]) == pytest.approx(expected)
    assert tuple(scene.geoms[ids["seg_b"]].rgba[:3]) == pytest.approx((0.5,) * 3)
    apply_scene_overlays(model, scene, 0, None, None, None)  # nothing staged
    assert scene.ngeom == before
    assert callable(force_glyphs.add_glyphs_to_scene)


# --- capability and runner warning ---------------------------------------


def test_shading_capability_set() -> None:
    assert set(ENGINES) >= SHADING_ENGINES
    assert sorted(SHADING_ENGINES) == ["drake", "mujoco", "myosuite", "pinocchio"]
    assert "opensim" not in SHADING_ENGINES


class _Backend:
    def __init__(self, engine: str) -> None:
        self.engine = engine

    def unavailable_reason(self):
        return None

    def render(self, swing, settings, indices, overlay):
        yield {v: np.zeros((8, 8, 3), np.uint8) for v in settings.views}


class _Writer:
    def __init__(self, path: Path, fps: int) -> None:
        self.path = path

    def append_data(self, frame) -> None:
        pass

    def close(self) -> None:
        self.path.write_bytes(b"x")


def _warnings_for(tmp_path: Path, engine: str, caplog) -> list[str]:
    q = np.zeros((21, 2))
    bundle = InputBundle(
        spec_bytes=json.dumps({"coordinate_order": ["A", "B"]}).encode(),
        coordinate_order=("A", "B"),
        dt_s=0.001,
        q0=q[0],
        v0=q[0],
        efforts=np.zeros((20, 2)),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
    )
    path = tmp_path / "b.npz"
    bundle.save(path)
    feed = OverlayFeed(lambda i: ForceTorqueFrame(time_s=0.0, engine="mujoco"))
    with caplog.at_level(logging.WARNING):
        run_export(
            ExportJob(path, tmp_path / "o", "d", "Driver", (engine,)),
            ExportSettings(
                width=32,
                height=32,
                fps=20,
                speeds=(1.0,),
                shading=ForceColorScale(enabled=True),
                impact_time_s=0.0,
            ),
            lambda e: _Backend(e),  # type: ignore[arg-type,return-value]
            lambda swing, e: (feed, (0.0, 0.0, 1.0)),
            _Writer,
        )
    return [r.getMessage() for r in caplog.records if "shading" in r.getMessage()]


@pytest.mark.parametrize("engine", sorted(SHADING_ENGINES))
def test_supported_engines_do_not_warn(tmp_path: Path, caplog, engine: str) -> None:
    assert _warnings_for(tmp_path, engine, caplog) == []


def test_unsupported_engine_warns_with_supported_list(tmp_path: Path, caplog) -> None:
    (msg,) = _warnings_for(tmp_path, "opensim", caplog)
    assert "opensim" in msg and "drake" in msg and "unshaded" in msg

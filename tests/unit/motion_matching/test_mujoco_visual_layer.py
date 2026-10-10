"""MuJoCo visual layer: capsule skeleton, floor and lights without physics change."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
FULL_BODY = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"

_HINGE_XML = (
    "<mujoco><worldbody>"
    '<body name="b"><joint name="hinge1" type="hinge" axis="0 1 0"/>'
    '<geom type="sphere" size="0.05"/></body>'
    "</worldbody></mujoco>"
)


def _spec_bytes() -> bytes:
    return json.dumps({"contact": {"ground_height_m": 0.0}}).encode()


class _FakeRenderer:
    """Renderer stub: no offscreen GL needed, records frames drawn."""

    def __init__(self, model: Any, height: int, width: int) -> None:
        self.calls = 0

    def update_scene(self, data: Any, camera: Any = None) -> None:
        self.calls += 1

    def render(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)


def _patch_playback_rendering(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the mujoco render + GIF write so FrameSchedule sampling is isolated."""
    import imageio
    import mujoco

    from src.engines.physics_engines.mujoco.python import full_body_mjcf

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        full_body_mjcf,
        "export_full_body_mjcf",
        lambda spec_bytes, visual=False: (_HINGE_XML, {}),
    )
    monkeypatch.setattr(mujoco, "Renderer", _FakeRenderer)

    def fake_mimsave(path: Path, frames: list, **kwargs: Any) -> None:
        captured["frames"] = frames
        captured["kwargs"] = kwargs

    monkeypatch.setattr(imageio, "mimsave", fake_mimsave)
    return captured


def test_visual_export_adds_geoms_but_keeps_physics_identical() -> None:
    mujoco = pytest.importorskip("mujoco")
    raw = FULL_BODY.read_bytes()
    plain_xml, plain_meta = exporter.export_full_body_mjcf(raw)
    visual_xml, visual_meta = exporter.export_full_body_mjcf(raw, visual=True)
    plain = mujoco.MjModel.from_xml_string(plain_xml)
    visual = mujoco.MjModel.from_xml_string(visual_xml)
    assert visual.nq == plain.nq == 41 and visual.nv == plain.nv == 41
    assert visual.nbody == plain.nbody
    assert visual.ngeom > plain.ngeom + 20  # one capsule per body plus floor
    assert visual.nlight >= 1
    assert visual_meta["visual_layer"]["capsules"] >= plain.nbody - 1
    assert visual_meta["visual_layer"]["floor"] is True
    assert "visual_layer" not in plain_meta
    # Visual geoms never collide and never contribute inertia.
    for i in range(visual.ngeom):
        if visual.geom_group[i] == 1:
            assert visual.geom_contype[i] == 0 and visual.geom_conaffinity[i] == 0
    np.testing.assert_allclose(visual.body_mass, plain.body_mass)
    np.testing.assert_allclose(visual.body_inertia, plain.body_inertia)
    d_plain, d_visual = mujoco.MjData(plain), mujoco.MjData(visual)
    rng = np.random.default_rng(0)
    q = rng.normal(scale=0.2, size=plain.nq)
    d_plain.qpos[:] = q
    d_visual.qpos[:] = q
    mujoco.mj_forward(plain, d_plain)
    mujoco.mj_forward(visual, d_visual)
    np.testing.assert_allclose(d_visual.xpos, d_plain.xpos, atol=1e-12)
    np.testing.assert_allclose(d_visual.qacc, d_plain.qacc, atol=1e-9)


@pytest.mark.requires_gl
def test_visual_export_renders_offscreen(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    xml, _ = exporter.export_full_body_mjcf(FULL_BODY.read_bytes(), visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, 120, 160)
    except (RuntimeError, ValueError, OSError) as error:  # headless CI without GL
        pytest.skip(f"no offscreen GL: {error}")
    renderer.update_scene(data)
    image = renderer.render()
    assert image.shape == (120, 160, 3)
    assert image.mean() > 5.0  # not a black frame: floor and lit capsules visible


def test_whole_body_com_and_scene_markers(tmp_path: Path) -> None:
    import mujoco

    from src.engines.physics_engines.mujoco.python import visual_layer

    xml, _ = exporter.export_full_body_mjcf(FULL_BODY.read_bytes(), visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    com = visual_layer.whole_body_com(model, data)
    masses = model.body_mass[1:]
    expected = (masses[:, None] * data.xipos[1:]).sum(axis=0) / masses.sum()
    np.testing.assert_allclose(com, expected, atol=1e-9)
    scene = mujoco.MjvScene(model, maxgeom=4000)
    option = mujoco.MjvOption()
    mujoco.mjv_updateScene(
        model, data, option, None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    before = scene.ngeom
    visual_layer.add_com_markers(scene, model, data, 0.0)
    assert scene.ngeom == before + 2
    with pytest.raises(ValueError):
        visual_layer.add_scene_marker(scene, com, -0.01, (1, 0, 0, 1))


def test_render_playback_samples_via_frame_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GCV-14 (#11720): frame count follows FrameSchedule, not a fixed stride."""
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python import visual_layer
    from src.shared.python.video_timing.frame_schedule import FrameSchedule

    captured = _patch_playback_rendering(monkeypatch)
    q = np.zeros((101, 1))  # 1.0 s of swing at rate_hz=100
    visual_layer.render_playback(
        _spec_bytes(),
        ["hinge1"],
        q,
        np.zeros(3),
        tmp_path / "out.gif",
        show_com=False,
        rate_hz=100.0,
        fps=60.0,
    )
    expected = FrameSchedule(np.arange(101) / 100.0, 60.0, 1.0).n_frames
    assert len(captured["frames"]) == expected
    assert captured["kwargs"]["duration"] == pytest.approx(1000.0 / 60.0)


def test_render_playback_half_speed_doubles_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python import visual_layer

    q = np.zeros((101, 1))
    captured_full = _patch_playback_rendering(monkeypatch)
    visual_layer.render_playback(
        _spec_bytes(),
        ["hinge1"],
        q,
        np.zeros(3),
        tmp_path / "a.gif",
        show_com=False,
        rate_hz=100.0,
        fps=60.0,
        speed=1.0,
    )
    full = len(captured_full["frames"])

    captured_half = _patch_playback_rendering(monkeypatch)
    visual_layer.render_playback(
        _spec_bytes(),
        ["hinge1"],
        q,
        np.zeros(3),
        tmp_path / "b.gif",
        show_com=False,
        rate_hz=100.0,
        fps=60.0,
        speed=0.5,
    )
    half = len(captured_half["frames"])
    assert abs(half - 2 * full) <= 1


def test_render_playback_stride_alias_warns_and_converts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deprecated ``playback_stride`` converts to an equivalent speed (mirrors
    ``ExportSettings.stride`` in native_viewer_export/core.py)."""
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python import visual_layer
    from src.shared.python.video_timing.frame_schedule import FrameSchedule

    captured = _patch_playback_rendering(monkeypatch)
    q = np.zeros((101, 1))
    with pytest.warns(DeprecationWarning, match="playback_stride"):
        visual_layer.render_playback(
            _spec_bytes(),
            ["hinge1"],
            q,
            np.zeros(3),
            tmp_path / "c.gif",
            show_com=False,
            rate_hz=120.0,
            playback_stride=4,
            fps=30.0,
        )
    # stride=4 at rate_hz=120, fps=30 -> speed = 4 * 30 / 120 = 1.0 (real time)
    expected = FrameSchedule(np.arange(101) / 120.0, 30.0, 1.0).n_frames
    assert len(captured["frames"]) == expected


@pytest.mark.parametrize("kwargs", [{"fps": 0.0}, {"speed": -1.0}, {"rate_hz": 0.0}])
def test_render_playback_invalid_timing_raises(kwargs: dict) -> None:
    from src.engines.physics_engines.mujoco.python import visual_layer

    with pytest.raises(ValueError):
        visual_layer.render_playback(
            b'{"contact": {}}',
            ["j"],
            np.zeros((2, 1)),
            np.zeros(3),
            Path("x.gif"),
            **kwargs,
        )


def test_render_playback_invalid_stride_raises() -> None:
    from src.engines.physics_engines.mujoco.python import visual_layer

    with pytest.raises(ValueError, match="playback_stride"):
        visual_layer.render_playback(
            b'{"contact": {}}',
            ["j"],
            np.zeros((2, 1)),
            np.zeros(3),
            Path("x.gif"),
            playback_stride=0,
        )

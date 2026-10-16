"""The MuJoCo viewport must not go black when its renderer is replaced (UD #12884).

Root cause
----------
``mujoco.Renderer.close()`` (mujoco 3.x, GLFW backend) destroys its own GL
window *first* and only then frees its ``MjrContext``. At that moment the GL
context that is current is whichever renderer rendered or was created last,
so ``mjr_freeContext`` deletes *that* renderer's framebuffer and textures
(object names are per-context counters, so they collide). In the launcher the
dashboard loads a model twice (startup plus the Physics-tab selection), the
second ``_finalize_model_load`` drops the first renderer while the new one is
current, and every later frame of the live renderer reads back pure black.
The standalone dashboard only loads once, which is why it looked fine.

These tests pin:

* the pure frame-content check used as the post-condition (no GL needed);
* :func:`release_renderer` frees the ``MjrContext`` against its *own* context
  before closing (fakes, no GL needed);
* ``MuJoCoSimWidget`` releases the old renderer before it constructs a new
  one (fake renderer, no GL needed);
* with a real GL context: a second model load still renders a non-black
  frame. These skip cleanly when no OpenGL context can be created.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

mujoco = pytest.importorskip("mujoco", reason="MuJoCo required for the sim widget")

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._frame_check import (  # noqa: E402
    FrameStats,
    frame_stats,
    is_blank_frame,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._renderer_lifecycle import (  # noqa: E402
    release_renderer,
)

_SCENE_XML = """
<mujoco>
  <visual><global offwidth="320" offheight="240"/></visual>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 0.1" rgba="0.3 0.6 0.3 1"/>
    <body pos="0 0 1"><freejoint/>
      <geom type="box" size="0.2 0.2 0.2" rgba="0.9 0.2 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


# --------------------------------------------------------------------------
# Pure frame-content check (no GL)
# --------------------------------------------------------------------------


def _gradient_frame(h: int = 48, w: int = 64) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    rgb = np.stack([xs * 4 % 256, ys * 5 % 256, (xs + ys) % 256], axis=-1)
    return rgb.astype(np.uint8)


@pytest.mark.unit
def test_all_black_frame_is_blank() -> None:
    assert is_blank_frame(np.zeros((48, 64, 3), dtype=np.uint8))


@pytest.mark.unit
def test_black_frame_with_small_overlay_is_still_blank() -> None:
    """The audit symptom: black viewport plus a cyan 'left_foot' overlay."""
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[200:210, 300:330] = (0, 255, 255)
    assert is_blank_frame(rgb)


@pytest.mark.unit
def test_uniform_grey_frame_is_blank() -> None:
    """Bright but featureless is not a rendered scene either."""
    assert is_blank_frame(np.full((48, 64, 3), 128, dtype=np.uint8))


@pytest.mark.unit
def test_textured_frame_is_not_blank() -> None:
    assert not is_blank_frame(_gradient_frame())


@pytest.mark.unit
def test_frame_stats_reports_luminance_and_colours() -> None:
    stats = frame_stats(_gradient_frame())
    assert isinstance(stats, FrameStats)
    assert stats.mean_luminance > 5.0
    assert stats.distinct_colors > 8


@pytest.mark.unit
def test_frame_stats_accepts_rgba() -> None:
    rgba = np.concatenate(
        [_gradient_frame(), np.full((48, 64, 1), 255, dtype=np.uint8)], axis=-1
    )
    assert frame_stats(rgba) == frame_stats(_gradient_frame())


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad",
    [np.zeros((48, 64), dtype=np.uint8), np.zeros((0, 64, 3), dtype=np.uint8)],
)
def test_frame_stats_rejects_non_image_shapes(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        frame_stats(bad)


@pytest.mark.unit
def test_frame_stats_rejects_non_uint8() -> None:
    with pytest.raises(TypeError):
        frame_stats(np.zeros((4, 4, 3), dtype=np.float32))


# --------------------------------------------------------------------------
# release_renderer ordering (fakes, no GL)
# --------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []


class _FakeGLContext:
    def __init__(self, rec: _Recorder) -> None:
        self._rec = rec

    def make_current(self) -> None:
        self._rec.calls.append("make_current")


class _FakeMjrContext:
    def __init__(self, rec: _Recorder) -> None:
        self._rec = rec

    def free(self) -> None:
        self._rec.calls.append("mjr.free")


class _FakeRenderer:
    def __init__(self, rec: _Recorder) -> None:
        self._rec = rec
        self._gl_context: Any = _FakeGLContext(rec)
        self._mjr_context: Any = _FakeMjrContext(rec)

    def close(self) -> None:
        self._rec.calls.append("close")


@pytest.mark.unit
def test_release_frees_mjr_context_under_its_own_gl_context_first() -> None:
    rec = _Recorder()
    release_renderer(_FakeRenderer(rec))
    assert rec.calls == ["make_current", "mjr.free", "close"]


@pytest.mark.unit
def test_release_of_none_is_a_no_op() -> None:
    release_renderer(None)


@pytest.mark.unit
def test_release_of_already_closed_renderer_only_closes() -> None:
    rec = _Recorder()
    renderer = _FakeRenderer(rec)
    renderer._gl_context = None
    renderer._mjr_context = None
    release_renderer(renderer)
    assert rec.calls == ["close"]


@pytest.mark.unit
def test_release_rejects_objects_that_cannot_close() -> None:
    with pytest.raises(TypeError):
        release_renderer(object())


# --------------------------------------------------------------------------
# MuJoCoSimWidget lifecycle (fake renderer, no GL)
# --------------------------------------------------------------------------


@pytest.fixture
def qapp() -> Any:
    pytest.importorskip("PyQt6", reason="PyQt6 required for the sim widget")
    from src.shared.python.ui.qt.utils import get_qapp

    try:
        return get_qapp()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qt initialisation failed (headless environment?): {exc}")


@pytest.fixture
def sim_widget_module() -> Any:
    pytest.importorskip("PyQt6", reason="PyQt6 required for the sim widget")
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (
        sim_widget,
    )

    return sim_widget


@pytest.fixture
def widget(qapp: Any, sim_widget_module: Any) -> Any:
    made = sim_widget_module.MuJoCoSimWidget(width=320, height=240, fps=30)
    try:
        yield made
    finally:
        made.stop_simulation()
        made.deleteLater()


@pytest.mark.unit
def test_old_renderer_is_released_before_the_new_one_is_built(
    widget: Any, sim_widget_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class _OrderedRenderer:
        count = 0

        def __init__(self, *_a: Any, **_k: Any) -> None:
            _OrderedRenderer.count += 1
            self.name = f"r{_OrderedRenderer.count}"
            events.append(f"create {self.name}")

    monkeypatch.setattr(sim_widget_module.mujoco, "Renderer", _OrderedRenderer)
    monkeypatch.setattr(
        sim_widget_module,
        "release_renderer",
        lambda r: events.append(f"release {getattr(r, 'name', r)}"),
    )
    monkeypatch.setattr(widget, "_render_once", lambda: None)

    widget.load_model_from_xml(_SCENE_XML)
    widget.load_model_from_xml(_SCENE_XML)

    assert events == [
        "release None",
        "create r1",
        "release r1",
        "create r2",
    ]
    assert widget.renderer.name == "r2"


# --------------------------------------------------------------------------
# Real GL context (skip cleanly when none is available)
# --------------------------------------------------------------------------


@pytest.fixture
def gl_model() -> Any:
    model = mujoco.MjModel.from_xml_string(_SCENE_XML)
    try:
        probe = mujoco.Renderer(model, 32, 32)
    except Exception as exc:  # noqa: BLE001 - any backend failure means "no GL"
        pytest.skip(f"No OpenGL context available for mujoco.Renderer: {exc}")
    release_renderer(probe)
    return model


def _render(model: Any, renderer: Any) -> np.ndarray:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    renderer.update_scene(data, camera=camera)
    return renderer.render()


@pytest.mark.unit
@pytest.mark.requires_gl
def test_released_renderer_does_not_blank_the_live_one(gl_model: Any) -> None:
    first = mujoco.Renderer(gl_model, 240, 320)
    second = mujoco.Renderer(gl_model, 240, 320)
    release_renderer(first)
    frame = _render(gl_model, second)
    release_renderer(second)
    assert not is_blank_frame(frame), frame_stats(frame)


@pytest.mark.unit
@pytest.mark.requires_gl
def test_sim_widget_viewport_survives_a_second_model_load(
    gl_model: Any, widget: Any
) -> None:
    """The launcher path: two model loads, then the label must show a scene."""
    widget.load_model_from_xml(_SCENE_XML)
    widget.load_model_from_xml(_SCENE_XML)
    widget._render_once()

    image = widget.label.pixmap().toImage()
    from PyQt6 import QtGui

    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes())
    rows = np.frombuffer(ptr, np.uint8).reshape(image.height(), image.bytesPerLine())
    frame = rows[:, : image.width() * 3].reshape(image.height(), image.width(), 3)
    release_renderer(widget.renderer)
    widget.renderer = None
    assert not is_blank_frame(frame), frame_stats(frame)

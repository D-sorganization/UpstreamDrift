"""Regression: glyph slots must be initialised with mjv_initGeom (NV-2, #11675).

``add_glyphs_to_scene`` used to fill ``scene.geoms`` slots without
``mjv_initGeom``; stale ``matid``/``dataid`` values then made
``Renderer.render()`` segfault whenever arrows or arcs were present.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

mujoco = pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (  # noqa: E402
    add_glyphs_to_scene,
)

from tests.unit.engines.mujoco.test_force_glyphs import (  # noqa: E402
    _make_dummy_arc,
    _make_dummy_arrow,
    _make_glyph_set,
)

pytestmark = pytest.mark.unit

_POISON = 2000000000
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _poison_free_slots(scene: mujoco.MjvScene) -> None:
    """Fill every slot past ngeom with out-of-range ids (what stale memory holds)."""
    for i in range(scene.ngeom, scene.maxgeom):
        scene.geoms[i].matid = _POISON
        scene.geoms[i].dataid = _POISON


def test_written_slots_are_initialised() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    scene = mujoco.MjvScene(model, maxgeom=64)
    _poison_free_slots(scene)
    glyphs = _make_glyph_set(
        arrows=(_make_dummy_arrow(),), torque_arcs=(_make_dummy_arc(n_segments=4),)
    )
    receipt = add_glyphs_to_scene(scene, glyphs)
    assert receipt.added == 1 + 4 + 1
    for i in range(receipt.added):
        geom = scene.geoms[i]
        assert geom.matid == -1, f"slot {i} matid not reset by mjv_initGeom"
        assert geom.dataid == -1, f"slot {i} dataid not reset by mjv_initGeom"


_RENDER_SCRIPT = textwrap.dedent(
    """
    import os, sys
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from tests.unit.engines.mujoco.test_force_glyphs import (
        _make_dummy_arc, _make_dummy_arrow, _make_glyph_set)
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (
        add_glyphs_to_scene)
    xml = '''<mujoco><visual><global offwidth="64" offheight="64"/></visual>
    <worldbody><light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.2 0.2 0.2 1"/>
    <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.5 1"/></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, 64, 64)
    except Exception as exc:  # no GL context available
        print("SKIP", exc)
        sys.exit(0)
    for _ in range(3):
        renderer.update_scene(data, "cam")
        scene = renderer.scene
        for i in range(scene.ngeom, scene.maxgeom):
            scene.geoms[i].matid = 2000000000
            scene.geoms[i].dataid = 2000000000
        glyphs = _make_glyph_set(
            arrows=(_make_dummy_arrow(tail_m=(-0.5, 0.0, 0.2), tip_m=(0.5, 0.0, 0.2),
                                      rgba=(1.0, 0.0, 0.0, 1.0)),),
            torque_arcs=(_make_dummy_arc(radius_m=0.3, n_segments=16),))
        add_glyphs_to_scene(scene, glyphs)
        img = renderer.render()
        assert img.shape == (64, 64, 3)
    print("RENDERED")
    """
)


@pytest.mark.requires_gl
def test_render_with_arrows_and_arcs_does_not_crash() -> None:
    """Run the render in a child process so a segfault fails instead of killing pytest."""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(_REPO_ROOT), str(_REPO_ROOT / "src")]),
    }
    proc = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, "-X", "faulthandler", "-c", _RENDER_SCRIPT],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
        env=env,
        cwd=_REPO_ROOT,
    )
    if proc.returncode == 0 and "SKIP" in proc.stdout:
        pytest.skip(f"offscreen GL unavailable: {proc.stdout.strip()}")
    assert proc.returncode == 0, (
        f"render crashed (rc={proc.returncode}): {proc.stderr[proc.stderr.find('Fatal') :][:400]}"
    )
    assert "RENDERED" in proc.stdout

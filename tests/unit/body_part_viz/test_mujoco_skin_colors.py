"""Per-segment skin pieces take the shared axial-load colours (#15150)."""

import numpy as np
import pytest

from src.shared.python.body_part_viz import AxialLoadFrame, ForceColorScale

mujoco = pytest.importorskip("mujoco")

pytestmark = pytest.mark.unit

_SKIN = (
    '<skin name="{name}" material="{material}" vertex="0 0 0 1 0 0 0 1 0"'
    ' face="0 1 2"><bone body="{body}" bindpos="0 0 0" bindquat="1 0 0 0"'
    ' vertid="0 1 2" vertweight="1 1 1"/></skin>'
)
_XML = (
    "<mujoco><asset>"
    '<material name="m_thigh" rgba="0.2 0.4 0.6 1"/>'
    '<material name="m_shank" rgba="0.5 0.5 0.5 0.8"/>'
    "</asset><worldbody>"
    '<body name="thigh"><joint/><geom size="0.1"/>'
    '<body name="shank" pos="0 0 -1"><joint/><geom size="0.1"/></body></body>'
    "</worldbody><deformable>"
    + _SKIN.format(name="skin_thigh", material="m_thigh", body="thigh")
    + _SKIN.format(name="skin_shank", material="m_shank", body="shank")
    + "</deformable></mujoco>"
)
_PIECES = {"skin_thigh": "thigh", "skin_shank": "shank"}


def _module():
    from src.shared.python.body_part_viz import mujoco_skin_colors

    return mujoco_skin_colors


def _model():
    return mujoco.MjModel.from_xml_string(_XML)


def _rgba(model, name):
    return model.mat_rgba[model.material(name).id].copy()


def test_bindings_record_each_piece_body_material_and_base_colour():
    pieces = _module().skin_piece_bindings(_model(), _PIECES)
    assert [(p.material, p.body) for p in pieces] == [
        ("m_thigh", "thigh"),
        ("m_shank", "shank"),
    ]
    assert pieces[0].base_rgba == pytest.approx((0.2, 0.4, 0.6, 1.0))


def test_loaded_piece_takes_the_scale_colour_and_keeps_its_alpha():
    mod, model = _module(), _model()
    pieces = mod.skin_piece_bindings(model, _PIECES)
    scale = ForceColorScale(enabled=True)
    frame = AxialLoadFrame(0.0, {"thigh": -1000.0, "shank": None}, "test")
    assert mod.apply_mujoco_skin_colors(model, pieces, frame, scale) == 1
    expected = scale.shade_rgb(-1000.0, (0.2, 0.4, 0.6))
    assert _rgba(model, "m_thigh")[:3] == pytest.approx(expected, abs=1e-6)
    assert _rgba(model, "m_thigh")[3] == pytest.approx(1.0)
    # an unavailable load keeps the base colour: it is never drawn as zero
    assert _rgba(model, "m_shank") == pytest.approx((0.5, 0.5, 0.5, 0.8))


def test_base_colours_return_when_the_load_or_the_scale_goes_away():
    mod, model = _module(), _model()
    pieces = mod.skin_piece_bindings(model, _PIECES)
    on = ForceColorScale(enabled=True)
    mod.apply_mujoco_skin_colors(
        model, pieces, AxialLoadFrame(0.0, {"thigh": 500.0}, "test"), on
    )
    assert mod.apply_mujoco_skin_colors(model, pieces, None, on) == 0
    assert _rgba(model, "m_thigh") == pytest.approx((0.2, 0.4, 0.6, 1.0))
    mod.apply_mujoco_skin_colors(
        model, pieces, AxialLoadFrame(0.0, {"thigh": 500.0}, "test"), on
    )
    off = ForceColorScale()
    frame = AxialLoadFrame(0.0, {"thigh": 500.0}, "test")
    assert mod.apply_mujoco_skin_colors(model, pieces, frame, off) == 0
    assert _rgba(model, "m_thigh") == pytest.approx((0.2, 0.4, 0.6, 1.0))


def test_skin_alpha_scales_the_base_opacity_with_or_without_shading():
    mod, model = _module(), _model()
    pieces = mod.skin_piece_bindings(model, _PIECES)
    mod.apply_mujoco_skin_colors(model, pieces, None, ForceColorScale(), skin_alpha=0.5)
    assert _rgba(model, "m_thigh")[3] == pytest.approx(0.5)
    assert _rgba(model, "m_shank")[3] == pytest.approx(0.4)
    assert np.allclose(_rgba(model, "m_thigh")[:3], (0.2, 0.4, 0.6))


def test_bad_inputs_are_rejected():
    mod, model = _module(), _model()
    pieces = mod.skin_piece_bindings(model, _PIECES)
    with pytest.raises(ValueError, match="unknown skin"):
        mod.skin_piece_bindings(model, {"nope": "thigh"})
    with pytest.raises(ValueError, match="unknown body"):
        mod.skin_piece_bindings(model, {"skin_thigh": "nope"})
    with pytest.raises(ValueError, match="skin_alpha"):
        mod.apply_mujoco_skin_colors(
            model, pieces, None, ForceColorScale(), skin_alpha=1.5
        )
    with pytest.raises(TypeError):
        mod.apply_mujoco_skin_colors(model, pieces, None, "scale")
    with pytest.raises(TypeError):
        mod.apply_mujoco_skin_colors(object(), pieces, None, ForceColorScale())


def test_one_material_shared_by_two_bodies_is_rejected():
    xml = _XML.replace('material="m_shank" vertex', 'material="m_thigh" vertex')
    model = mujoco.MjModel.from_xml_string(xml)
    with pytest.raises(ValueError, match="shared"):
        _module().skin_piece_bindings(model, _PIECES)

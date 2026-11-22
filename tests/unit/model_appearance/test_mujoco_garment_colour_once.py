"""A textured garment's colour must be applied once, not twice (#15150).

MuJoCo multiplies a material ``rgba`` by its texture. A garment whose rgba and
procedural texture both carry the garment colour therefore renders as the
colour squared (darker). The product of the two must equal the requested
colour.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405  # nosec B405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # test-only construction of MJCF, never parses input

import numpy as np
import pytest

from src.shared.python.model_appearance import (
    document_from_dict,
    library,
)
from src.shared.python.model_appearance.appearance_selection import (
    AppearanceOverrides,
    apply_overrides,
)
from src.shared.python.model_appearance.defaults import default_document_dict

pytestmark = pytest.mark.unit

TOLERANCE = 1.0 / 255.0
TOP_OVERRIDE = "#2f5fa8"
_TOP_RGB = (0x2F / 255.0, 0x5F / 255.0, 0xA8 / 255.0)


def _rendered_rgb(doc_dict: dict, part_label: str) -> tuple[np.ndarray, np.ndarray]:
    """(requested rgb, compiled rgba * median texture rgb) of the torso garment."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python import appearance_layer as layer
    from src.shared.python.model_appearance.appearance_selection import (
        _garment_name,
    )

    doc = document_from_dict(doc_dict)
    name = _garment_name(doc.clothing, "torso", part_label)
    root = ET.Element("mujoco")
    assets = layer._Assets(root, doc)
    mat_name = assets.material(name)
    ET.SubElement(
        ET.SubElement(root, "worldbody"),
        "geom",
        type="box",
        size="0.1 0.1 0.1",
        material=mat_name,
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
    rgba = np.asarray(model.mat_rgba[mat_id], dtype=float)
    tex_id = int(model.mat_texid[mat_id].max())
    assert tex_id >= 0, "garment must be textured for this check"
    adr = int(model.tex_adr[tex_id])
    size = int(model.tex_width[tex_id] * model.tex_height[tex_id]) * int(
        model.tex_nchannel[tex_id]
    )
    pixels = np.asarray(model.tex_data[adr : adr + size], dtype=float)
    tex_rgb = np.median(pixels.reshape(-1, int(model.tex_nchannel[tex_id])), axis=0)
    tex_rgb = tex_rgb[:3] / 255.0
    requested = np.asarray(library.library_materials(doc)[name].rgba[:3])
    return requested, rgba[:3] * tex_rgb


def test_default_top_colour_applied_once() -> None:
    requested, rendered = _rendered_rgb(default_document_dict(), "top")
    assert np.all(np.abs(rendered - requested) <= TOLERANCE), (requested, rendered)


def test_overridden_top_colour_applied_once() -> None:
    doc = apply_overrides(
        default_document_dict(), AppearanceOverrides(top_color=TOP_OVERRIDE)
    )
    requested, rendered = _rendered_rgb(doc, "top")
    assert np.allclose(requested, _TOP_RGB, atol=TOLERANCE)
    assert np.all(np.abs(rendered - requested) <= TOLERANCE), (requested, rendered)


def test_textured_garment_keeps_alpha() -> None:
    from src.engines.physics_engines.mujoco.python import appearance_layer as layer

    material = library.MATERIALS["polo_navy"]
    rgba = layer._material_attrs(material, plane=False)["rgba"].split()
    assert float(rgba[3]) == pytest.approx(material.rgba[3])


def test_untextured_skin_material_keeps_its_own_colour() -> None:
    """The skinned body drops the skin texture; the colour must then be rgba."""
    pytest.importorskip("mujoco")
    from pathlib import Path

    import json

    from defusedxml import ElementTree as DET

    from src.engines.physics_engines.mujoco.python import full_body_mjcf
    from src.shared.python.model_appearance import defaults

    spec_path = (
        Path(__file__).resolve().parents[3]
        / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["subject"]["body_visual_backend"] = "skinned_cc0"
    doc = defaults.default_appearance()
    xml, _ = full_body_mjcf.export_full_body_mjcf(
        json.dumps(spec).encode(), appearance=doc
    )
    root = DET.fromstring(xml)
    skin = next(s for s in root.iter("skin") if s.get("name") == "body_skin_skin")
    material = next(
        m for m in root.iter("material") if m.get("name") == skin.get("material")
    )
    assert material.get("texture") is None
    rgba = [float(v) for v in material.get("rgba", "").split()]
    expected = library.library_materials(doc)[
        skin.get("material", "")[len("mat_") :]
    ].rgba
    assert np.allclose(rgba, expected, atol=TOLERANCE), (rgba, expected)

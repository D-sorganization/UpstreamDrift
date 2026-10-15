"""R2 (#12681): the pelvis body wears the active outfit's pants in the MJCF."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.shared.python.model_appearance import CLOTHING, document_from_dict
from src.shared.python.model_appearance.library import pants_material

pytestmark = pytest.mark.unit
MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
SPECS = [MODELS / f"full_body_spec_anthro_{c}.json" for c in ("driver", "iron7")]
PELVIS = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/LowerTorso"
DRESSED = sorted(p for p in CLOTHING if pants_material(p) is not None)


def _pelvis_materials(spec: Path, doc_dict: dict) -> set[str]:
    xml, _ = exporter.export_full_body_mjcf(
        spec.read_bytes(), appearance=document_from_dict(doc_dict)
    )
    body = next(b for b in ET.fromstring(xml).iter("body") if b.get("name") == PELVIS)
    return {
        g.get("material", "")
        for g in body.findall("geom")
        if g.get("name", "").startswith("visual_") and "_skin" not in g.get("name", "")
    }


@pytest.mark.parametrize("spec", SPECS, ids=lambda p: p.stem)
@pytest.mark.parametrize("preset", [None, *DRESSED])
def test_pelvis_visual_geoms_use_the_active_pants_material(
    spec: Path, preset: str | None
) -> None:
    doc_dict = {"schema_version": "appearance-v1"}
    if preset is not None:
        doc_dict["clothing"] = preset
    active = document_from_dict(doc_dict).clothing  # None -> the default outfit
    pants = pants_material(active)
    assert pants is not None
    used = _pelvis_materials(spec, doc_dict)
    assert used == {f"mat_{pants}"}, used
    assert f"mat_{CLOTHING[active]['torso']}" not in used

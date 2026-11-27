"""Contact spheres are hidden in the default drawing, never in the physics."""

from pathlib import Path

import pytest
from defusedxml import ElementTree as ET

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.shared.python.model_appearance import document_from_dict

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
FULL_BODY = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
DOC = document_from_dict({"schema_version": "appearance-v1"})


def _contact_alpha(xml: str) -> float:
    root = ET.fromstring(xml)
    for cls in root.iter("default"):
        if cls.get("class") == "contact":
            rgba = cls.find("geom").get("rgba")
            return 1.0 if rgba is None else float(rgba.split()[3])
    raise AssertionError("no contact default class")


def _export(**kw):
    return exporter.export_full_body_mjcf(FULL_BODY.read_bytes(), **kw)[0]


def test_default_appearance_export_hides_contact_spheres() -> None:
    assert _contact_alpha(_export(appearance=DOC)) == 0.0


def test_setting_shows_contact_spheres_again() -> None:
    xml = _export(appearance=DOC, show_contact_spheres=True)
    assert _contact_alpha(xml) > 0.0


def test_hidden_spheres_stay_in_the_model() -> None:
    mujoco = pytest.importorskip("mujoco")
    shown = mujoco.MjModel.from_xml_string(
        _export(appearance=DOC, show_contact_spheres=True)
    )
    hidden = mujoco.MjModel.from_xml_string(_export(appearance=DOC))
    assert hidden.ngeom == shown.ngeom
    assert hidden.nsite == shown.nsite


def test_plain_physics_export_is_unchanged() -> None:
    assert _contact_alpha(_export()) == 1.0

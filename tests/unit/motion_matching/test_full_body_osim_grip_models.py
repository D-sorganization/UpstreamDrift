"""``grip_model`` option of the OpenSim full-body exporter (#11739, OSV-7)."""

from __future__ import annotations

import json
from pathlib import Path

import defusedxml.ElementTree as ET
import pytest

from src.engines.physics_engines.opensim.python.full_body_osim import (
    export_full_body_osim,
)
from src.shared.python.grip_contact import GripInterface, default_bushing

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def _root(xml: str):
    return ET.fromstring(xml.replace("::", "__"))


def test_default_is_weld_and_unchanged(spec: dict) -> None:
    xml_default, meta_default = export_full_body_osim(spec)
    xml_weld, _ = export_full_body_osim(spec, grip_model="weld")
    assert xml_default == xml_weld
    assert meta_default["grip_model"] == "weld"
    root = _root(xml_default)
    assert root.find(".//WeldConstraint") is not None
    assert root.find(".//BushingForce") is None
    assert root.find(".//FreeJoint") is None


def test_contact_needs_its_configuration(spec: dict) -> None:
    with pytest.raises(ValueError, match="grip_contact"):
        export_full_body_osim(spec, grip_model="contact")


def test_contact_configuration_only_valid_with_contact_model(
    spec: dict, tmp_path: Path
) -> None:
    from src.engines.physics_engines.opensim.python.full_body_grip_contact import (
        ContactGripConfig,
    )
    from src.shared.python.grip_contact.pad_contact import build_pad_model

    cfg = ContactGripConfig(
        build_pad_model(GripInterface.from_spec(spec), 1100.0), tmp_path
    )
    with pytest.raises(ValueError, match="only valid"):
        export_full_body_osim(spec, grip_model="bushing", grip_contact=cfg)


def test_unknown_grip_model_rejected(spec: dict) -> None:
    with pytest.raises(ValueError, match="grip_model"):
        export_full_body_osim(spec, grip_model="glue")


def test_bushing_topology(spec: dict) -> None:
    xml, meta = export_full_body_osim(spec, grip_model="bushing")
    root = _root(xml)
    assert meta["grip_model"] == "bushing"
    assert root.find(".//WeldConstraint") is None
    assert root.find(".//FreeJoint") is not None
    names = [e.get("name") for e in root.findall(".//BushingForce")]
    assert names == ["grip_bushing_left", "grip_bushing_right"]
    assert meta["coordinate_count"] == 44 + 6
    assert meta["coordinate_order"][:44] == spec["coordinate_order"]
    # club coordinates are passive: no coordinate actuator on them
    actuators = {e.get("name") for e in root.findall(".//CoordinateActuator")}
    assert not any(a.startswith("tau_ClubFree") for a in actuators)
    assert "tau_LWInputX" in actuators


def test_bushing_conserves_total_mass_and_club_mass(spec: dict) -> None:
    xml_w, _ = export_full_body_osim(spec)
    xml_b, _ = export_full_body_osim(spec, grip_model="bushing")

    def masses(xml: str) -> dict[str, float]:
        return {
            b.get("name"): float(b.find("mass").text)
            for b in _root(xml).findall(".//BodySet/objects/Body")
        }

    mw, mb = masses(xml_w), masses(xml_b)
    assert sum(mb.values()) == pytest.approx(sum(mw.values()), rel=1e-12)
    assert mb["Clubhead"] == pytest.approx(spec["club"]["total_mass_kg"])
    for name, m in mw.items():
        if name not in ("Clubhead", "Grip"):
            assert mb[name] == pytest.approx(m)


def test_bushing_parameters_are_emitted(spec: dict) -> None:
    bp = default_bushing().scaled(0.5)
    gi = GripInterface.from_spec(spec, bushing=bp)
    xml, _ = export_full_body_osim(spec, grip_model="bushing", grip_interface=gi)
    force = _root(xml).find(".//BushingForce")
    got = [float(x) for x in force.find("translational_stiffness").text.split()]
    assert got == pytest.approx(list(bp.translational_stiffness_n_m))


def test_bushing_spec_requires_hand_solids(spec: dict) -> None:
    broken = json.loads(json.dumps(spec))
    club = next(b for b in broken["bodies"] if "Clubface" in b["name"])
    club["solids"] = [s for s in club["solids"] if not s["name"].endswith("/RHand")]
    with pytest.raises(ValueError, match="hand solids"):
        export_full_body_osim(broken, grip_model="bushing")

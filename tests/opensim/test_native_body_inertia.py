"""Actual OpenSim source/body inertia admission and provenance negatives."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_body_inertia import (
    audit_native_body_inertia,
)

pytestmark = pytest.mark.integration


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(path: Path, *, mass: float = 1.0, welded: bool = False) -> None:
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    body = osim.Body(
        "segment", mass, osim.Vec3(0.1, -0.2, 0.3), osim.Inertia(0.03, 0.04, 0.05)
    )
    model.addBody(body)
    model.addJoint(
        (osim.WeldJoint if welded else osim.PinJoint)(
            "hinge",
            model.getGround(),
            osim.Vec3(0),
            osim.Vec3(0),
            body,
            osim.Vec3(0),
            osim.Vec3(0),
        )
    )
    model.initSystem()
    model.printToXML(str(path))


def _change_body(path: Path, field: str, value: str) -> None:
    tree = ElementTree.parse(path)
    node = tree.find(f"./Model/BodySet/objects/Body/{field}")
    assert node is not None
    node.text = value
    tree.write(path, encoding="utf-8", xml_declaration=True)


def test_physical_source_and_native_readback_are_bound(tmp_path: Path) -> None:
    source = tmp_path / "body.osim"
    _model(source)

    audit = audit_native_body_inertia(source, _digest(source))

    assert audit.admitted
    assert audit.body_count == 1
    assert audit.bodies[0].name == "segment"
    assert audit.bodies[0].source_native_exact
    assert audit.source_sha256 == _digest(source)
    audit.require_admitted()


def test_rotated_unphysical_tensor_rejected_by_principal_moments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rotated.osim"
    _model(source)
    cosine = np.sqrt(0.5)
    rotation = np.array([[cosine, 0, cosine], [0, 1, 0], [-cosine, 0, cosine]])
    tensor = rotation @ np.diag([0.02, 0.02, 0.08]) @ rotation.T
    _change_body(
        source,
        "inertia",
        " ".join(
            str(value)
            for value in (
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[0, 1],
                tensor[0, 2],
                tensor[1, 2],
            )
        ),
    )

    audit = audit_native_body_inertia(source, _digest(source))

    assert not audit.admitted
    assert audit.bodies[0].name == "segment"
    assert "triangle inequality" in audit.bodies[0].failure_reason
    with pytest.raises(ValueError, match="segment"):
        audit.require_admitted()


def test_source_mutation_or_massless_nonzero_tensor_fails_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "changed.osim"
    _model(source)
    original_digest = _digest(source)
    _change_body(source, "mass", "0")
    with pytest.raises(ValueError, match="source.*hash"):
        audit_native_body_inertia(source, original_digest)
    with pytest.raises(ValueError, match="massless.*inertia"):
        audit_native_body_inertia(source, _digest(source))


def test_exact_zero_mass_and_zero_inertia_route_is_explicit(tmp_path: Path) -> None:
    source = tmp_path / "carrier.osim"
    _model(source, welded=True)
    _change_body(source, "mass", "0")
    _change_body(source, "inertia", "0 0 0 0 0 0")

    audit = audit_native_body_inertia(source, _digest(source))

    assert audit.admitted
    assert audit.bodies[0].mass_kg == 0
    assert audit.bodies[0].inertia_kg_m2 == (0, 0, 0, 0, 0, 0)


def test_native_mass_readback_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    osim = pytest.importorskip("opensim")
    source = tmp_path / "body.osim"
    _model(source)
    native_model = osim.Model

    def altered_load(path: str) -> object:
        model = native_model(path)
        model.updBodySet().get("segment").setMass(2.0)
        return model

    monkeypatch.setattr(osim, "Model", altered_load)
    with pytest.raises(ValueError, match="source/native body mass"):
        audit_native_body_inertia(source, _digest(source))


def test_malformed_body_center_is_rejected_before_native_loading(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad-center.osim"
    _model(source)
    _change_body(source, "mass_center", "0 NaN 0")
    with pytest.raises(ValueError, match="malformed source body mass, COM"):
        audit_native_body_inertia(source, _digest(source))


def test_exact_557_source_reports_four_invalid_body_tensors() -> None:
    path_text = os.environ.get("OPEN_SIM_EXACT_557_SOURCE")
    if path_text is None:
        pytest.skip("external exact 557-muscle source is unavailable")
    source = Path(path_text)

    audit = audit_native_body_inertia(source, _digest(source))

    assert not audit.admitted
    assert audit.body_count == 40
    assert {body.name for body in audit.bodies if body.failure_reason} == {
        "clavicle_r",
        "clavicle_l",
        "scapula_r",
        "scapula_l",
    }

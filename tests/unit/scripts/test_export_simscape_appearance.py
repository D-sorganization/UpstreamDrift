"""R9c (#12690): the GS3DX Simscape appearance comes from the shared spec.

Engine-free: the exporter resolves ``model_appearance`` and writes one
versioned JSON that MATLAB applies. These tests pin that every colour and
trunk size in it is the shared spec's (one source of truth), that the trunk
is wider than it is deep, and that the file is deterministic, versioned and
validated.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = REPO_ROOT / "scripts" / "export_simscape_appearance.py"
_MATLAB_ROOT = (
    REPO_ROOT / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab"
    "/exploratory_gs3dx"
)


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("export_simscape_appearance", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter() -> Any:
    return _load()


@pytest.fixture(scope="module")
def payload(exporter: Any) -> dict[str, Any]:
    return exporter.export()


def _visual(payload: dict[str, Any], name: str) -> dict[str, Any]:
    return next(v for v in payload["visuals"] if v["name"] == name)


def test_colours_are_the_shared_library_materials(payload: dict[str, Any]) -> None:
    from src.shared.python.model_appearance import library

    clothing = library.CLOTHING[payload["source"]["clothing"]]
    expect = {
        "top": library.MATERIALS[clothing["torso"]].rgba[:3],
        "pants": library.MATERIALS[clothing["pelvis"]].rgba[:3],
        "shoe": library.MATERIALS[clothing["foot"]].rgba[:3],
        "glove": library.MATERIALS[clothing["hand"]].rgba[:3],
        "skin": library.MATERIALS[payload["source"]["skin_tone"]].rgba[:3],
    }
    for role, rgb in expect.items():
        assert payload["colours"][role] == pytest.approx(rgb, abs=1e-6), role


def test_pelvis_wears_the_trousers_and_legs_match(payload: dict[str, Any]) -> None:
    pants = payload["colours"]["pants"]
    assert _visual(payload, "Pelvis")["rgb"] == pants
    legs = [
        r for r in payload["recolour"] if "Thigh" in r["block"] or "Shank" in r["block"]
    ]
    assert len(legs) == 4 and all(r["rgb"] == pants for r in legs)
    # The retired GS3DX literals (grey pants, blue shirt) must be gone.
    assert pants != [0.30, 0.30, 0.34]
    assert payload["colours"]["top"] != [0.22, 0.33, 0.62]


def test_trunk_is_sized_from_the_spec_with_ml_over_ap(
    payload: dict[str, Any], exporter: Any
) -> None:
    from src.shared.python.model_appearance import library

    thickness = library.GARMENT_THICKNESS
    spec = json.loads(exporter.DEFAULT_SPEC.read_text(encoding="utf-8"))
    chest = exporter.section_size_for(spec, "COMRod")
    ml, ap, _ = _visual(payload, "Chest")["radii_m"]
    assert ml == pytest.approx(chest.radius_m * thickness, abs=1e-5)
    assert ap == pytest.approx(chest.radius_m * chest.aspect * thickness, abs=1e-5)
    assert chest.radius_m == pytest.approx(0.158, abs=1e-3)  # thorax ML, #12790
    for name in ("Pelvis", "Abdomen", "Chest", "Trapezius"):
        ml, ap, _z = _visual(payload, name)["radii_m"]
        assert ml > ap > 0, name


def test_hip_and_shoulder_lines_are_documented_rotations(
    payload: dict[str, Any],
) -> None:
    assert _visual(payload, "Pelvis")["rotation_z_deg"] == pytest.approx(46.3)
    assert _visual(payload, "Chest")["rotation_z_deg"] == pytest.approx(17.0)


def test_output_is_deterministic_and_matches_the_committed_file(
    exporter: Any, payload: dict[str, Any]
) -> None:
    text = exporter.render_json(payload)
    assert text == exporter.render_json(exporter.export())
    committed = exporter.DEFAULT_OUTPUT.read_bytes().decode("utf-8")
    assert committed.replace("\r\n", "\n") == text, (
        "regenerate: PYTHONPATH=.:vendor/ud-tools/src "
        "python3 scripts/export_simscape_appearance.py"
    )


def test_schema_is_versioned_and_unknown_versions_raise(
    exporter: Any, payload: dict[str, Any]
) -> None:
    assert payload["schema_version"] == "simscape-appearance/v1"
    bad = copy.deepcopy(payload)
    bad["schema_version"] = "simscape-appearance/v2"
    with pytest.raises(ValueError, match="schema_version"):
        exporter.validate_payload(bad)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda p: p["colours"].update(skin=[1.2, 0.5, 0.5]), "rgb"),
        (lambda p: p["visuals"][0].update(radii_m=[0.1, 0.0, 0.1]), "positive"),
        (lambda p: p["visuals"][0].update(radii_m=[0.05, 0.1, 0.1]), "exceed"),
        (lambda p: p.pop("neck"), "missing"),
    ],
)
def test_invalid_payloads_are_rejected(
    exporter: Any, payload: dict[str, Any], mutate: Any, match: str
) -> None:
    bad = copy.deepcopy(payload)
    mutate(bad)
    with pytest.raises(ValueError, match=match):
        exporter.validate_payload(bad)


def test_unknown_spec_body_raises(exporter: Any) -> None:
    spec = json.loads(exporter.DEFAULT_SPEC.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="no body named"):
        exporter.section_size_for(spec, "NoSuchBody")


def test_matlab_builder_has_no_appearance_literals() -> None:
    """No skin/shirt/pants colour or trunk-ellipsoid radius survives in MATLAB."""
    src = (_MATLAB_ROOT / "tools" / "gs3dx_build_human.m").read_text(encoding="utf-8")
    body = src[src.index("function vis = local_body_visuals") :]
    body = body[: body.index("\nfunction vis = local_club")]
    assert not re.search(r"\b(skin|shirt|pants)\s*=\s*\[", body)
    assert "gs3dx_read_appearance" in body
    for literal in ("[0.17 0.115 0.12]", "[0.145 0.105 0.15]", "[0.16 0.11 0.155]"):
        assert literal not in src

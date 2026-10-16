"""Golden-output characterization tests for the model converter (#12841).

Pins the exact exported MJCF/URDF text and the schema validator's parsed
model / error messages so refactors under the architecture budget cannot
change observable behaviour.  Regenerate goldens only deliberately with
``MODEL_CONVERTER_WRITE_GOLDEN=1``.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_converter.mjcf_exporter import export_mjcf  # noqa: E402
from tools.model_converter.schema_validator import (  # noqa: E402
    ValidationError,
    validate_canonical_model,
)
from tools.model_converter.urdf_exporter import export_urdf  # noqa: E402

GOLDEN_DIR = Path(__file__).parent / "golden"
CANONICAL_YAML = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "models"
    / "spec"
    / "golfer_canonical.yaml"
)
WRITE = os.environ.get("MODEL_CONVERTER_WRITE_GOLDEN") == "1"


def _inertia() -> dict[str, float]:
    return {"ixx": 0.22, "iyy": 0.20, "izz": 0.10}


def _geom() -> dict[str, Any]:
    return {"type": "capsule", "size": [0.1, 0.3], "visual_rgba": [0.7, 0.5, 0.3, 1.0]}


def _seg(name: str, parent: str, joint: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "parent": parent,
        "joint": joint,
        "origin": {"xyz": [0.0, 0.0, 0.2], "rpy": [0.0, 0.1, 0.0]},
        "mass": 2.0,
        "inertia": _inertia(),
        "geometry": _geom(),
    }


def synthetic_dict() -> dict[str, Any]:
    """Model exercising every joint type plus site-bearing segment names."""
    dof = {"axis": [0, 0, 1], "limits": [-1.0, 1.0]}
    dof_y = {"axis": [0, 1, 0], "limits": [-0.5, 0.5]}
    dof_x = {"axis": [1, 0, 0]}
    return {
        "description": "Synthetic golden model",
        "units": {"length": "meters"},
        "coordinate_system": {"z": "up"},
        "root": {
            "name": "pelvis",
            "position": [0.0, 0.0, 0.9],
            "orientation": [0.0, 0.0, 0.1],
            "mass": 10.0,
            "inertia": _inertia(),
            "geometry": _geom(),
        },
        "segments": [
            _seg(
                "thorax",
                "pelvis",
                {
                    "type": "revolute",
                    "axis": [0, 0, 1],
                    "limits": [-1, 1],
                    "damping": 1.5,
                },
            ),
            _seg(
                "neck",
                "thorax",
                {"type": "universal", "dofs": [dof, dof_y], "damping": 0.5},
            ),
            _seg("shoulder", "thorax", {"type": "gimbal", "dofs": [dof, dof_y, dof_x]}),
            _seg("hand_left", "shoulder", {"type": "fixed"}),
            _seg("hand_right", "shoulder", {"type": "revolute", "axis": [1, 0, 0]}),
            _seg("club_shaft", "hand_right", {"type": "fixed"}),
            _seg("club_head", "club_shaft", {"type": "fixed"}),
            _seg("slider", "pelvis", {"type": "prismatic"}),
        ],
        "constraints": [{"kind": "weld"}],
        "contacts": [{"name": "c"}],
        "named_frames": [{"name": "f"}],
        "scaling": {"height": 1.8},
    }


def _model(case: str):
    if case == "canonical":
        return validate_canonical_model(CANONICAL_YAML)
    return validate_canonical_model(synthetic_dict())


def _check(name: str, text: str) -> None:
    path = GOLDEN_DIR / name
    if WRITE:
        GOLDEN_DIR.mkdir(exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    assert text.encode("utf-8") == path.read_bytes()


@pytest.mark.unit
@pytest.mark.parametrize("case", ["canonical", "synthetic"])
def test_mjcf_output_is_byte_identical(case: str) -> None:
    _check(f"{case}.mjcf.xml", export_mjcf(_model(case)))


@pytest.mark.unit
@pytest.mark.parametrize("case", ["canonical", "synthetic"])
def test_urdf_output_is_byte_identical(case: str) -> None:
    _check(f"{case}.urdf.xml", export_urdf(_model(case)))


@pytest.mark.unit
@pytest.mark.parametrize("case", ["mjcf", "urdf"])
def test_exporters_write_same_text_to_disk(case: str, tmp_path: Path) -> None:
    model = _model("synthetic")
    export = export_mjcf if case == "mjcf" else export_urdf
    out = tmp_path / "nested" / f"m.{case}"
    text = export(model, out)
    assert out.read_text(encoding="utf-8") == text


@pytest.mark.unit
@pytest.mark.parametrize("case", ["canonical", "synthetic"])
def test_validated_model_repr_is_stable(case: str) -> None:
    _check(f"{case}.model.repr.txt", repr(_model(case)))


def _seg_of(d: dict[str, Any], i: int = 0) -> dict[str, Any]:
    return d["segments"][i]


def _mutations() -> dict[str, Callable[[dict[str, Any]], Any]]:
    def revolute(**kw: Any) -> dict[str, Any]:
        return {"type": "revolute", **kw}

    def universal(*dofs: dict[str, Any]) -> dict[str, Any]:
        return {"type": "universal", "dofs": list(dofs)}

    z = {"axis": [0, 0, 1]}

    def cycle(d: dict[str, Any]) -> None:
        _seg_of(d, 0).update(parent="neck")

    return {
        "not_dict_toplevel": lambda d: [1, 2],
        "missing_root": lambda d: d.pop("root"),
        "missing_segments": lambda d: d.pop("segments"),
        "units_not_dict": lambda d: d.update(units=[1]),
        "coord_not_dict": lambda d: d.update(coordinate_system="x"),
        "root_not_dict": lambda d: d.update(root=[1]),
        "root_no_name": lambda d: d["root"].pop("name"),
        "root_name_not_str": lambda d: d["root"].update(name=5),
        "root_mass_zero": lambda d: d["root"].update(mass=0),
        "root_inertia_missing": lambda d: d["root"]["inertia"].pop("iyy"),
        "root_inertia_negative": lambda d: d["root"]["inertia"].update(izz=-1),
        "root_inertia_triangle": lambda d: d["root"]["inertia"].update(
            ixx=0.01, iyy=0.01, izz=0.5
        ),
        "root_inertia_not_pd": lambda d: d["root"]["inertia"].update(ixy=5.0),
        "segments_empty": lambda d: d.update(segments=[]),
        "segments_not_list": lambda d: d.update(segments={}),
        "segment_not_dict": lambda d: d["segments"].append(3),
        "segment_no_name": lambda d: _seg_of(d).pop("name"),
        "segment_duplicate": lambda d: _seg_of(d, 1).update(name="thorax"),
        "segment_root_name_dup": lambda d: _seg_of(d).update(name="pelvis"),
        "segment_no_parent": lambda d: _seg_of(d).pop("parent"),
        "segment_mass_zero": lambda d: _seg_of(d).update(mass=0),
        "segment_bad_parent": lambda d: _seg_of(d).update(parent="ghost"),
        "joint_bad_type": lambda d: _seg_of(d).update(joint={"type": "wobbly"}),
        "joint_neg_damping": lambda d: _seg_of(d).update(
            joint=revolute(axis=[0, 0, 1], damping=-1)
        ),
        "revolute_no_axis": lambda d: _seg_of(d).update(joint=revolute()),
        "revolute_zero_axis": lambda d: _seg_of(d).update(
            joint=revolute(axis=[0, 0, 0])
        ),
        "revolute_bad_limits": lambda d: _seg_of(d).update(
            joint=revolute(axis=[0, 0, 1], limits=[1, -1])
        ),
        "universal_wrong_dofs": lambda d: _seg_of(d).update(joint=universal(z)),
        "gimbal_wrong_dofs": lambda d: _seg_of(d).update(
            joint={"type": "gimbal", "dofs": []}
        ),
        "dof_no_axis": lambda d: _seg_of(d).update(joint=universal(z, {})),
        "dof_zero_axis": lambda d: _seg_of(d).update(
            joint=universal(z, {"axis": [0, 0, 0]})
        ),
        "dof_bad_limits": lambda d: _seg_of(d).update(
            joint=universal(z, {"axis": [1, 0, 0], "limits": [2, 1]})
        ),
        "segment_inertia_missing": lambda d: _seg_of(d).update(inertia={}),
        "cycle": cycle,
        "self_cycle": lambda d: _seg_of(d, 0).update(parent="thorax"),
    }


@pytest.mark.unit
def test_validator_error_messages_are_stable() -> None:
    results: dict[str, str] = {}
    for name, mutate in _mutations().items():
        data = copy.deepcopy(synthetic_dict())
        replaced = mutate(data)
        if name == "not_dict_toplevel":
            data = replaced
        try:
            validate_canonical_model(data)
        except Exception as exc:  # noqa: BLE001 - pin exact exception types
            results[name] = f"{type(exc).__name__}: {exc}"
        else:
            results[name] = "OK"
    text = json.dumps(results, indent=2, sort_keys=True) + "\n"
    _check("validator_errors.json", text)


@pytest.mark.unit
def test_validator_rejects_wrong_source_type() -> None:
    with pytest.raises(ValidationError, match="Expected dict, Path, or str"):
        validate_canonical_model(42)  # type: ignore[arg-type]


@pytest.mark.unit
def test_validator_accepts_str_path() -> None:
    model = validate_canonical_model(str(CANONICAL_YAML))
    assert model.root.name == "pelvis"

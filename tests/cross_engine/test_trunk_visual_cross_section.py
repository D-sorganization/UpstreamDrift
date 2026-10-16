"""Cross-engine guard: every engine draws the trunk wider (ML) than deep (AP).

R1 (#12680, epic #12678). The MuJoCo appearance layer once lofted the trunk
with the full radius on the anterior axis, so the chest rendered deeper than
wide while Drake, Pinocchio, OpenSim and MyoSuite drew the spec's own
``visual_hints`` ellipsoids the right way round. This guard checks each
engine's trunk visual source in the canonical spec body frame for both
anthropometric captures. Simscape (GS3DX) is covered by R9c.

The ML direction is the spec landmark axis resolved by
``model_appearance.cross_section`` (hips for the pelvis, shoulders for the
trunk); the AP direction is ML x up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.shared.python.model_appearance import document_from_dict
from src.shared.python.model_appearance.cross_section import body_sections
from src.shared.python.model_appearance.library import classify_body
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton

pytestmark = [pytest.mark.unit, pytest.mark.cross_engine]

ROOT = Path(__file__).resolve().parents[2]
SPECS = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json",
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_iron7.json",
)
TRUNK_PARTS = ("torso", "pelvis")
# Backends that draw ``skeleton.shapes`` (the spec hint ellipsoids) as the trunk.
HINT_SHAPE_BACKENDS = {
    "opensim": "src/tools/native_viewer_export/backends/opensim_worker.py",
}
# Backends that render the shared appearance document (same trunk lofts as the
# MuJoCo appearance layer, #12688) via the common MeshCat scene builder.
APPEARANCE_BACKENDS = {
    "drake": "src/tools/native_viewer_export/backends/drake_meshcat.py",
    "pinocchio": "src/tools/native_viewer_export/backends/pinocchio_meshcat.py",
}


def _ellipsoid_semi_axis(half: np.ndarray, direction: np.ndarray) -> float:
    """Radius of an axis-aligned ellipsoid along a unit ``direction``."""
    d = direction / np.linalg.norm(direction)
    return float(1.0 / np.sqrt(np.sum((d / half) ** 2)))


def _trunk_hint_extents(spec: dict[str, Any]) -> list[tuple[str, float, float]]:
    """(body, ML semi-axis, AP semi-axis) of every trunk hint shape."""
    sections = body_sections(spec)
    out = []
    for shape in derive_visual_skeleton(spec).shapes:
        if classify_body(shape.body) not in TRUNK_PARTS:
            continue
        half = np.asarray(shape.half_size_m)
        sec = sections[shape.body]
        out.append(
            (
                shape.body,
                _ellipsoid_semi_axis(half, sec.ml_axis),
                _ellipsoid_semi_axis(half, sec.depth_axis),
            )
        )
    return out


def _mjcf_trunk_extents(
    spec: dict[str, Any], raw: bytes, *, appearance: bool
) -> list[tuple[str, float, float]]:
    """(body, ML extent, AP extent) of the trunk visual geoms in an MJCF export.

    With ``appearance`` the skin meshes are measured; without it (plain
    ``visual=True`` export, as MyoSuite uses) the hint ellipsoid geoms are.
    Both are mapped back from the MJCF body (joint follower) frame.
    """
    if appearance:
        xml, _ = exporter.export_full_body_mjcf(
            raw, appearance=document_from_dict({"schema_version": "appearance-v1"})
        )
    else:
        xml, _ = exporter.export_full_body_mjcf(raw, visual=True)
    root = ET.fromstring(xml)
    meshes = {
        m.get("name"): np.array(m.get("vertex", "").split(), float).reshape(-1, 3)
        for m in root.iter("mesh")
    }
    follower = {j["child"]: np.asarray(j["child_to_follower"]) for j in spec["joints"]}
    sections = body_sections(spec)
    out = []
    for body in root.iter("body"):
        name = body.get("name") or ""
        if name not in sections or classify_body(name) not in TRUNK_PARTS:
            continue
        offset = np.linalg.inv(follower[name])
        rot = offset[:3, :3]
        sec = sections[name]
        for geom in body.findall("geom"):
            if appearance and "_skin" in (geom.get("mesh") or ""):
                local = (meshes[geom.get("mesh")] - offset[:3, 3]) @ rot
                out.append(
                    (
                        name,
                        float(np.ptp(local @ sec.ml_axis)),
                        float(np.ptp(local @ sec.depth_axis)),
                    )
                )
            elif not appearance and geom.get("type") == "ellipsoid":
                half = np.array(geom.get("size", "").split(), float)
                assert "quat" not in geom.attrib  # axis-aligned in the MJCF frame
                out.append(
                    (
                        name,
                        _ellipsoid_semi_axis(half, rot @ sec.ml_axis),
                        _ellipsoid_semi_axis(half, rot @ sec.depth_axis),
                    )
                )
    return out


def _extents(engine: str, spec_path: Path) -> list[tuple[str, float, float]]:
    raw = spec_path.read_bytes()
    spec = json.loads(raw)
    if engine == "mujoco_appearance":
        return _mjcf_trunk_extents(spec, raw, appearance=True)
    if engine == "myosuite":  # visual=True MJCF with the hint ellipsoids
        return _mjcf_trunk_extents(spec, raw, appearance=False)
    if engine in APPEARANCE_BACKENDS:
        source = (ROOT / APPEARANCE_BACKENDS[engine]).read_text(encoding="utf-8")
        assert "default_appearance" in source, f"{engine} no longer uses the appearance"
        return _mjcf_trunk_extents(spec, raw, appearance=True)
    source = (ROOT / HINT_SHAPE_BACKENDS[engine]).read_text(encoding="utf-8")
    assert "skeleton.shapes" in source, f"{engine} no longer draws the hint shapes"
    return _trunk_hint_extents(spec)


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.stem)
@pytest.mark.parametrize(
    "engine", ["mujoco_appearance", "myosuite", "drake", "pinocchio", "opensim"]
)
def test_trunk_is_wider_than_deep_in_every_engine(engine, spec_path) -> None:
    extents = _extents(engine, spec_path)
    bodies = {body for body, _, _ in extents}
    assert any(b.endswith("LowerTorso") for b in bodies), (engine, bodies)
    assert any(b.endswith("COMRod") for b in bodies), (engine, bodies)
    for body, ml, ap in extents:
        assert ml > ap, (engine, body, ml, ap)

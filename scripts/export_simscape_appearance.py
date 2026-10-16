"""Export the shared appearance spec to the GS3DX Simscape golfer (R9c, #12690).

The GS3DX golfer draws its body with massless ellipsoids built by
``gs3dx_build_human.m``. Until R9c those ellipsoids carried their own
hard-coded colours and radii, a third appearance source next to MuJoCo and the
shared spec. This script is the only bridge: it resolves the engine-neutral
``model_appearance`` spec (:func:`resolve_appearance`, :func:`section_size`)
and writes one versioned JSON file that MATLAB reads and applies. MATLAB holds
no colour or trunk-size literal and fails closed when the file is absent.

Frame contract (DbC)
    The spec sizes the trunk by a mediolateral (ML) semi-axis and an AP/ML
    aspect, independent of any body frame. The GS3DX trunk frames have z down
    the spine, so every ellipsoid is emitted as ``radii_m = [ML, AP, z]``
    with its x axis turned onto the hip or shoulder line by
    ``rotation_z_deg``: x is the mediolateral axis, y the anteroposterior
    axis. ``ML > AP`` is checked here and again, in the world frame at
    address, by ``exploratory_gs3dx/tests/test_gs3dx_appearance.m``.

The placement of each ellipsoid in the GS3DX frames (``offset_m``, the
spine-axis half length ``z_m``, the hip and shoulder line angles) is GS3DX
layout, not appearance, and is the only per-model data kept here. Visual only:
nothing in the file touches mass, inertia or joints.

Usage, from the repository root::

    PYTHONPATH=.:vendor/ud-tools/src python scripts/export_simscape_appearance.py
    PYTHONPATH=.:vendor/ud-tools/src python scripts/export_simscape_appearance.py --check
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "simscape-appearance/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = (
    REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
DEFAULT_APPEARANCE = (
    REPO_ROOT / "docs/development/full_body_models/full_body_spec_v1.appearance.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab"
    "/exploratory_gs3dx/appearance/gs3dx_appearance.v1.json"
)

ROLES = ("skin", "top", "pants", "shoe", "glove")
# role -> (anatomical part, mesh label prefix) whose resolved material it is.
_ROLE_SOURCE = {
    "skin": ("forearm", "skin"),
    "top": ("torso", "garment"),
    "pants": ("pelvis", "garment"),
    "shoe": ("foot", "garment"),
    "glove": ("hand", "garment"),
}
_TRUNK_PARTS = {"LowerTorso": "pelvis", "COMRod": "torso"}

# GS3DX frame layout. Hip line 46.3 deg from x toward y on the lower trunk;
# shoulder line 17 deg on the upper trunk (docs/HUMAN.md). Trunk frames have z
# down the spine.
HIP_LINE_DEG = 46.3
SHOULDER_LINE_DEG = 17.0
_LOWER = "Hips and Torso Inputs/LowerTorso"
_UPPER = "Hips and Torso Inputs/UpperTorsoTop"


@dataclass(frozen=True)
class _Trunk:
    """Trunk ellipsoid: ML/AP from the spec section of ``spec_body``."""

    name: str
    parent: str
    spec_body: str
    role: str
    ml_frac: float
    ap_frac: float
    z_m: float
    offset_m: tuple[float, float, float]
    rotation_z_deg: float


@dataclass(frozen=True)
class _Limb:
    """Round ellipsoid: radius = library part radius x ``radial_scale``."""

    name: str
    parent: str
    part: str
    role: str
    radial_scale: float
    z_m: float


_TRUNK_LAYOUT = (
    _Trunk("Pelvis", _LOWER, "LowerTorso", "pants", 1, 1, 0.12,
           (-0.027, 0.027, 0.174), HIP_LINE_DEG),
    _Trunk("Abdomen", _LOWER, "LowerTorso", "top", 1, 1, 0.15,
           (-0.0135, 0.0135, -0.03), HIP_LINE_DEG),
    _Trunk("Chest", _UPPER, "COMRod", "top", 1, 1, 0.155,
           (0.0, 0.0, 0.03), SHOULDER_LINE_DEG),
    _Trunk("Trapezius", _UPPER, "COMRod", "top", 0.75, 0.75, 0.045,
           (0.005, -0.015, -0.125), SHOULDER_LINE_DEG),
)  # fmt: skip
_LIMB_LAYOUT = (
    _Limb("L Shoulder", "HubtoLS", "shoulder", "top", 1.0, 0.095),
    _Limb("R Shoulder", "HubtoRS", "shoulder", "top", 1.0, 0.095),
    _Limb("L Upper Arm", "LUpperArm", "upper_arm", "top", 1.0, 0.165),
    _Limb("R Upper Arm", "RUpperArm", "upper_arm", "top", 1.0, 0.165),
    _Limb("L Forearm Upper", "Left Forearm/LUpperForearm", "forearm", "skin", 1.17, 0.08),
    _Limb("L Forearm Lower", "Left Forearm/LLowerForearm", "forearm", "skin", 0.86, 0.08),
    _Limb("R Forearm Upper", "Right Forearm/RUpperForearm", "forearm", "skin", 1.17, 0.08),
    _Limb("R Forearm Lower", "Right Forearm/RLowerForearm", "forearm", "skin", 0.86, 0.08),
)  # fmt: skip
# Blocks that already exist in the model and are only recoloured.
_RECOLOUR = (
    ("Hips and Torso Inputs/Head", "skin"),
    ("Grip/LHand", "glove"),
    ("Grip/RHand", "glove"),
    ("Lower Body/L Thigh", "pants"),
    ("Lower Body/R Thigh", "pants"),
    ("Lower Body/L Shank", "pants"),
    ("Lower Body/R Shank", "pants"),
)
# Neck: no spec source for its width. Radius is layout; its length is the
# model's NeckLength, applied by MATLAB.
NECK_RADIUS_M = 0.052
NECK_BELOW_C7_M = 0.05


def _short(body: str) -> str:
    return body.rsplit("/", 1)[-1]


def _role_rgb(resolved: Any, library: Any) -> dict[str, list[float]]:
    """RGB of each role, read from the resolved meshes (the only colour source)."""
    out: dict[str, list[float]] = {}
    for role, (part, label) in _ROLE_SOURCE.items():
        names = {
            m.material_name
            for m in resolved.meshes
            if library.classify_body(m.body) == part and m.label.startswith(label)
        }
        if len(names) != 1:
            raise ValueError(
                f"role {role!r} needs exactly one {part}/{label} material, got "
                f"{sorted(names)}"
            )
        rgba = resolved.rgba(next(iter(names)))
        out[role] = [round(c, 6) for c in rgba[:3]]
    return out


def _find_body(spec: Mapping[str, Any], short: str) -> str:
    from src.shared.python.motion_matching.visual_skeleton import (
        derive_visual_skeleton,
    )

    names = {c.body for c in derive_visual_skeleton(spec).capsules}
    found = sorted(n for n in names if _short(n) == short)
    if not found:
        raise ValueError(f"spec has no body named {short!r}")
    return found[0]


def section_size_for(spec: Mapping[str, Any], short_body: str) -> Any:
    """Shared-spec trunk section of the body whose last path segment matches."""
    from src.shared.python.model_appearance.body_meshes import section_size

    return section_size(spec, _find_body(spec, short_body))


def _trunk_visuals(
    spec: Mapping[str, Any], rgb: Mapping[str, list[float]], thickness: float
) -> list[dict[str, Any]]:
    out = []
    for t in _TRUNK_LAYOUT:
        size = section_size_for(spec, t.spec_body)
        ml = size.radius_m * t.ml_frac * thickness
        ap = size.radius_m * size.aspect * t.ap_frac * thickness
        out.append(
            {
                "name": t.name,
                "parent": t.parent,
                "role": t.role,
                "rgb": rgb[t.role],
                "radii_m": [round(ml, 6), round(ap, 6), t.z_m],
                "offset_m": list(t.offset_m),
                "rotation_z_deg": t.rotation_z_deg,
                "sizing_source": size.source,
            }
        )
    return out


def _limb_visuals(
    rgb: Mapping[str, list[float]], library: Any, thickness: float
) -> list[dict[str, Any]]:
    out = []
    for limb in _LIMB_LAYOUT:
        clothed = limb.role != "skin"
        r = library.PART_RADIUS_M[limb.part] * limb.radial_scale
        r *= thickness if clothed else 1.0
        out.append(
            {
                "name": limb.name,
                "parent": limb.parent,
                "role": limb.role,
                "rgb": rgb[limb.role],
                "radii_m": [round(r, 6), round(r, 6), limb.z_m],
                "offset_m": [0.0, 0.0, 0.0],
                "rotation_z_deg": 0.0,
                "sizing_source": "library",
            }
        )
    return out


def build_payload(
    spec: Mapping[str, Any], doc: Any, spec_name: str = "spec"
) -> dict[str, Any]:
    """The versioned GS3DX appearance payload for ``spec`` and ``doc`` (pure)."""
    from src.shared.python.model_appearance import library
    from src.shared.python.model_appearance.body_meshes import resolve_appearance

    resolved = resolve_appearance(spec, doc)
    rgb = _role_rgb(resolved, library)
    thickness = library.GARMENT_THICKNESS
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "spec": spec_name,
            "appearance": doc.name,
            "clothing": doc.clothing,
            "skin_tone": doc.skin_tone,
        },
        "colours": rgb,
        "visuals": _trunk_visuals(spec, rgb, thickness)
        + _limb_visuals(rgb, library, thickness),
        "recolour": [{"block": b, "rgb": rgb[r], "role": r} for b, r in _RECOLOUR],
        "neck": {
            "radius_m": NECK_RADIUS_M,
            "below_c7_m": NECK_BELOW_C7_M,
            "rgb": rgb["skin"],
        },
        "shoe_rgb": rgb["shoe"],
    }
    validate_payload(payload)
    return payload


def _check_rgb(rgb: Sequence[float], what: str) -> None:
    ok = len(rgb) == 3 and all(
        isinstance(c, (int, float)) and math.isfinite(c) and 0.0 <= c <= 1.0
        for c in rgb
    )
    if not ok:
        raise ValueError(f"{what}: rgb must be 3 floats in [0, 1], got {list(rgb)}")


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Precondition check shared by the writer and the tests (DbC).

    Raises ``ValueError`` on an unknown schema version, a bad colour, a
    non-positive radius, or a trunk ellipsoid that is deeper than it is wide.
    """
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unknown schema_version {version!r}, want {SCHEMA_VERSION}")
    for key in ("colours", "visuals", "recolour", "neck", "shoe_rgb"):
        if key not in payload:
            raise ValueError(f"payload is missing {key!r}")
    for role in ROLES:
        _check_rgb(payload["colours"][role], f"colours.{role}")
    for v in payload["visuals"]:
        radii = v["radii_m"]
        if not all(math.isfinite(r) and r > 0 for r in radii):
            raise ValueError(f"{v['name']}: radii must be positive, got {radii}")
        _check_rgb(v["rgb"], v["name"])
        if v["name"] in {t.name for t in _TRUNK_LAYOUT} and not radii[0] > radii[1]:
            raise ValueError(f"{v['name']}: ML {radii[0]} must exceed AP {radii[1]}")
    for item in payload["recolour"]:
        _check_rgb(item["rgb"], item["block"])
    _check_rgb(payload["shoe_rgb"], "shoe_rgb")


def render_json(payload: Mapping[str, Any]) -> str:
    """Deterministic text form (sorted by construction, LF, trailing newline)."""
    text = json.dumps(payload, indent=2)
    # Number arrays on one line, as prettier (a commit hook) formats them.
    text = re.sub(
        r"\[\s+([-0-9.e,\s]+?)\s+\]",
        lambda m: "[" + ", ".join(m.group(1).replace(",", " ").split()) + "]",
        text,
    )
    return text + "\n"


def export(
    spec_path: Path = DEFAULT_SPEC,
    appearance_path: Path = DEFAULT_APPEARANCE,
) -> dict[str, Any]:
    from src.shared.python.model_appearance.schema import load_appearance

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    doc = load_appearance(appearance_path)
    try:
        name = Path(spec_path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        name = Path(spec_path).name
    return build_payload(spec, doc, name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--appearance", type=Path, default=DEFAULT_APPEARANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed file is stale"
    )
    args = parser.parse_args(argv)
    text = render_json(export(args.spec, args.appearance))
    if args.check:
        current = (
            args.output.read_bytes().decode("utf-8").replace("\r\n", "\n")
            if args.output.exists()
            else None
        )
        if current != text:
            print(f"{args.output} is stale: rerun without --check", file=sys.stderr)
            return 1
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

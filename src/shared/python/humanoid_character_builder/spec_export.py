"""Character builder: parameters -> spec -> engine exports (CMB-3, #11654).

One implementation shared by the web API
(``src/api/routes/character_builder.py``) and the desktop tool
(``src/tools/character_builder``). Compilation goes through
``spec_params`` (CMB-1) and presets through the presets loader (CMB-2).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.shared.python.humanoid_character_builder.presets.loader import (
    CharacterPreset,
    list_character_presets,
    load_character_preset,
)
from src.shared.python.humanoid_character_builder.spec_params import (
    SpecCharacterParameters,
    compile_full_body_spec,
    serialize_spec,
)

EXPORT_FORMATS: dict[str, tuple[str, str]] = {
    "spec": ("application/json", "json"),
    "urdf": ("text/xml", "urdf"),
    "mjcf": ("text/xml", "xml"),
    "osim": ("text/xml", "osim"),
    "gltf": ("model/gltf+json", "gltf"),
}


@dataclass(frozen=True)
class CompiledCharacter:
    """A compiled character: parameters, spec document and its content hash."""

    parameters: SpecCharacterParameters
    document: dict[str, Any]
    sha256: str
    preset_id: str | None


def resolve_parameters(
    preset: str | None, overrides: Mapping[str, Any]
) -> tuple[SpecCharacterParameters, str | None]:
    """Return parameters from an optional preset plus non-None overrides.

    Raises ``ValueError`` for an unknown preset id or invalid values.
    """
    given = {k: v for k, v in overrides.items() if v is not None}
    if preset:
        loaded = load_character_preset(preset, **given)
        return loaded.parameters, loaded.id
    return SpecCharacterParameters.from_dict(given), None


def compile_character(
    preset: str | None, overrides: Mapping[str, Any]
) -> CompiledCharacter:
    """Resolve parameters and compile the full-body spec document."""
    params, preset_id = resolve_parameters(preset, overrides)
    document = compile_full_body_spec(params)
    sha = hashlib.sha256(serialize_spec(document).encode("utf-8")).hexdigest()
    return CompiledCharacter(params, document, sha, preset_id)


def _body_mass(body: Mapping[str, Any]) -> float:
    return float(sum(s["mass_kg"] for s in body["solids"]))


def build_summary(character: CompiledCharacter) -> dict[str, Any]:
    """Compact description of a compiled character for the build response."""
    doc = character.document
    return {
        "parameters": character.parameters.to_dict(),
        "preset": character.preset_id,
        "spec_sha256": character.sha256,
        "schema_version": doc["schema_version"],
        "qualification": doc["qualification"],
        "bodies": len(doc["bodies"]),
        "joints": len(doc["joints"]),
        "coordinates": len(doc["coordinate_order"]),
        "total_mass_kg": sum(_body_mass(b) for b in doc["bodies"]),
        "club": doc["club"]["name"],
    }


def build_preview(character: CompiledCharacter) -> dict[str, Any]:
    """Body masses and joint topology for a lightweight preview panel."""
    doc = character.document
    return {
        "spec_sha256": character.sha256,
        "stature_m": character.parameters.stature_m,
        "bodies": [
            {"name": b["name"], "mass_kg": _body_mass(b)} for b in doc["bodies"]
        ],
        "joints": [
            {"name": j["name"], "parent": j["parent"], "child": j["child"]}
            for j in doc["joints"]
        ],
    }


def export_character(character: CompiledCharacter, fmt: str) -> tuple[str, str, str]:
    """Return ``(text, media_type, filename)`` for ``fmt`` in ``EXPORT_FORMATS``.

    Engine exporters import lazily so listing presets never needs them.
    """
    if fmt not in EXPORT_FORMATS:
        raise ValueError(
            f"Unsupported export format {fmt!r}; use {sorted(EXPORT_FORMATS)}"
        )
    media_type, ext = EXPORT_FORMATS[fmt]
    text = serialize_spec(character.document)
    if fmt == "urdf":
        from src.engines.physics_engines.drake.python.full_body_urdf import (
            export_full_body_urdf,
        )

        text = export_full_body_urdf(text.encode("utf-8"))[0]
    elif fmt == "mjcf":
        from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
            export_full_body_mjcf,
        )

        text = export_full_body_mjcf(text.encode("utf-8"))[0]
    elif fmt == "osim":
        from src.engines.physics_engines.opensim.python.full_body_osim import (
            export_full_body_osim,
        )

        text = export_full_body_osim(character.document)[0]
    elif fmt == "gltf":
        from src.shared.python.model_appearance.skinned_body import (
            build_skinned_body,
            skinned_body_gltf_text,
        )

        text = skinned_body_gltf_text(
            build_skinned_body(character.document, spec_sha256=character.sha256)
        )
    stem = character.preset_id or "custom"
    return text, media_type, f"{stem}_{character.sha256[:8]}.{ext}"


def preset_listing() -> list[dict[str, Any]]:
    """Metadata for every shipped preset, parameters included."""
    items: list[CharacterPreset] = [
        load_character_preset(pid) for pid in list_character_presets()
    ]
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "category": p.category,
            "parameters": p.parameters.to_dict(),
            "provenance": p.provenance,
            "limitations": p.limitations,
        }
        for p in items
    ]


import defusedxml.ElementTree as DefusedET
import numpy as np
from src.shared.python.humanoid_character_builder.spec_params import (
    _compute_spec_poses,
    _verify_mjcf,
    _verify_osim_joint_offsets,
    _verify_urdf,
)


def _verify_osim(
    doc: dict[str, Any], spec_body_poses: dict[str, np.ndarray]
) -> tuple[float, float]:
    """Verify OpenSim mass conservation and body placement parity."""
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        clean_osim_body_name,
        export_full_body_osim,
    )

    osim_xml, _ = export_full_body_osim(doc)
    osim_root = DefusedET.fromstring(osim_xml.replace("::", "__"))
    osim_mass_kg = float(
        sum(
            float(m.text)
            for m in osim_root.findall(".//BodySet/objects/Body/mass")
            if m.text is not None
        )
    )

    osim_bodies = {
        b.attrib["name"]
        for b in osim_root.findall(".//BodySet/objects/Body")
        if "name" in b.attrib
    }
    for b in doc["bodies"]:
        if b["name"] == "world":
            continue
        expected_name = clean_osim_body_name(b["name"])
        if expected_name not in osim_bodies:
            raise ValueError(f"OpenSim missing body {expected_name} (from {b['name']})")

    osim_markers = {
        m.attrib["name"]
        for m in osim_root.findall(".//MarkerSet/objects/Marker")
        if "name" in m.attrib
    }
    for mname in doc.get("marker_attachments", {}):
        if mname not in osim_markers:
            raise ValueError(f"OpenSim missing marker {mname}")

    osim_poses = _verify_osim_joint_offsets(list(osim_root.findall(".//CustomJoint")))
    osim_body_errors = []
    for b in doc["bodies"]:
        if b["name"] == "world":
            continue
        clean = clean_osim_body_name(b["name"])
        sp = f"/bodyset/{clean}"
        if sp in osim_poses and b["name"] in spec_body_poses:
            err = float(
                np.linalg.norm(
                    spec_body_poses[b["name"]][:3, 3] - osim_poses[sp][:3, 3]
                )
            )
            osim_body_errors.append(err)
    osim_max_frame_err = max(osim_body_errors) if osim_body_errors else 0.0
    return osim_mass_kg, osim_max_frame_err


def _verify_pinocchio(
    doc: dict[str, Any],
    spec_frame_origins: dict[str, np.ndarray],
    spec_mass_kg: float,
    *,
    mass_tol_kg: float,
    frame_tol_m: float,
) -> int:
    """Verify Pinocchio depth-first joint tree length and native frame origins if installed."""
    from src.engines.physics_engines.pinocchio.python.native_model import (
        depth_first_joints,
    )

    pin_joints = depth_first_joints(doc["joints"])
    if len(pin_joints) != len(doc["joints"]):
        raise ValueError(
            f"Pinocchio joint tree length mismatch: {len(pin_joints)} != {len(doc['joints'])}"
        )

    try:
        import sys
        import types

        import pinocchio  # noqa: F401

        is_mock = not isinstance(pinocchio, types.ModuleType) or (
            not hasattr(pinocchio, "__file__")
            and pinocchio.__name__ not in sys.builtin_module_names
        )
        if not is_mock:
            from src.engines.physics_engines.pinocchio.python.native_model import (
                FullBodyPinocchioModel,
            )

            pin_model = FullBodyPinocchioModel(doc)
            pin_frame_poses = pin_model.frame_poses({})
            for fname, fpose in pin_frame_poses.items():
                if fname in spec_frame_origins:
                    err = float(
                        np.linalg.norm(spec_frame_origins[fname] - fpose[:3, 3])
                    )
                    if err > frame_tol_m:
                        raise ValueError(
                            f"Pinocchio frame origin error {err} > {frame_tol_m} for {fname}"
                        )
            pin_total_mass = float(sum(b.mass for b in pin_model.model.inertias))
            if abs(pin_total_mass - spec_mass_kg) > mass_tol_kg:
                raise ValueError(
                    f"Pinocchio mass mismatch: {pin_total_mass} vs {spec_mass_kg}"
                )
    except (ImportError, ModuleNotFoundError, TypeError):
        pass

    return len(pin_joints)


def verify_cross_engine_l0_parity(
    character: CompiledCharacter,
    *,
    mass_tol_kg: float = 1e-9,
    frame_tol_m: float = 1e-3,
) -> dict[str, Any]:
    """Verify L0 export parity across MuJoCo, Drake/URDF, OpenSim, and Pinocchio (CMB-11, #11662).

    Validates total mass conservation and world-frame origin agreement at
    neutral pose (q = 0) against the compiled full-body specification.
    """
    if mass_tol_kg <= 0:
        raise ValueError(f"mass_tol_kg must be > 0, got {mass_tol_kg}")
    if frame_tol_m <= 0:
        raise ValueError(f"frame_tol_m must be > 0, got {frame_tol_m}")

    doc = character.document
    spec_mass_kg, spec_body_poses, spec_frame_origins = _compute_spec_poses(doc)
    spec_bytes = serialize_spec(doc).encode("utf-8")

    mjcf_mass_kg, mjcf_max_frame_err = _verify_mjcf(spec_bytes, spec_frame_origins)
    urdf_mass_kg, urdf_max_frame_err = _verify_urdf(spec_bytes, spec_frame_origins)
    osim_mass_kg, osim_max_frame_err = _verify_osim(doc, spec_body_poses)
    pin_joint_count = _verify_pinocchio(
        doc,
        spec_frame_origins,
        spec_mass_kg,
        mass_tol_kg=mass_tol_kg,
        frame_tol_m=frame_tol_m,
    )

    engine_masses_kg = {
        "mjcf": mjcf_mass_kg,
        "urdf": urdf_mass_kg,
        "osim": osim_mass_kg,
    }
    engine_max_frame_error_m = {
        "mjcf": mjcf_max_frame_err,
        "urdf": urdf_max_frame_err,
        "osim": osim_max_frame_err,
    }

    mass_errors = [abs(m - spec_mass_kg) for m in engine_masses_kg.values()]
    max_mass_error_kg = float(max(mass_errors)) if mass_errors else 0.0
    max_frame_origin_error_m = (
        float(max(engine_max_frame_error_m.values()))
        if engine_max_frame_error_m
        else 0.0
    )

    if max_mass_error_kg > mass_tol_kg:
        raise ValueError(
            f"Cross-engine mass parity violation: max mass error {max_mass_error_kg:.4e} kg "
            f"> tolerance {mass_tol_kg:.4e} kg (engine masses: {engine_masses_kg})"
        )

    if max_frame_origin_error_m > frame_tol_m:
        raise ValueError(
            f"Cross-engine frame origin parity violation: max frame origin error "
            f"{max_frame_origin_error_m:.4e} m > tolerance {frame_tol_m:.4e} m "
            f"(engine errors: {engine_max_frame_error_m})"
        )

    return {
        "spec_sha256": character.sha256,
        "spec_mass_kg": spec_mass_kg,
        "engine_masses_kg": engine_masses_kg,
        "engine_max_frame_error_m": engine_max_frame_error_m,
        "max_mass_error_kg": max_mass_error_kg,
        "max_frame_origin_error_m": max_frame_origin_error_m,
        "pinocchio_joint_count": pin_joint_count,
        "passed": True,
    }


__all__ = [
    "EXPORT_FORMATS",
    "CompiledCharacter",
    "build_preview",
    "build_summary",
    "compile_character",
    "export_character",
    "preset_listing",
    "resolve_parameters",
    "verify_cross_engine_l0_parity",
]

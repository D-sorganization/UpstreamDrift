"""Validate and import immutable sized catalog models into Necromatcher."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.shared.python.model_appearance.player_models import (
    player_ids,
    player_profile,
)
from src.shared.python.model_appearance.schema import (
    document_from_dict,
    document_to_dict,
    physics_spec_sha256,
)
from src.shared.python.motion_matching.pipeline.plant import (
    ScalarCoordinateUnits,
    get_plant,
)

from .necromatcher import NecromatcherLibrary


@dataclass(frozen=True)
class CatalogModelImport:
    """Stable identifiers and hashes for one imported catalog candidate."""

    player_id: str
    session_id: str
    model_id: str
    bundle_asset_id: str
    model_sha256: str
    visual_sha256: str
    source_manifest_sha256: str
    physics_spec_sha256: str
    coordinate_units: tuple[str, ...]


def _read_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return raw, value


def import_catalog_model(
    library: NecromatcherLibrary,
    player_id: str,
    source_directory: Path | str,
) -> CatalogModelImport:
    """Validate a player export completely, then add it atomically to a library.

    The export directory must contain the exact artifacts written by
    ``player_models.export_player``. The function never creates a library,
    changes an existing identity, or overwrites an asset. Exact reimports are
    idempotent; any collision or incomplete prior registration is rejected.
    """
    if player_id not in player_ids():
        raise ValueError(f"Unknown catalog player {player_id!r}")
    source_root = Path(source_directory).expanduser().resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("Model source must be an export directory")

    profile = player_profile(player_id)
    receipt_bytes, receipt = _read_json(source_root / "receipt.json", "receipt.json")
    spec_bytes, spec = _read_json(source_root / "model.json", "model.json")
    visual_bytes, visual = _read_json(
        source_root / "model.visual.json", "model.visual.json"
    )
    _, appearance_record = _read_json(
        source_root / "model.appearance.json", "model.appearance.json"
    )
    xml_path = source_root / "model.xml"
    if xml_path.is_symlink() or not xml_path.is_file():
        raise ValueError("model.xml must be a regular file")
    xml_bytes = xml_path.read_bytes()
    if not xml_bytes:
        raise ValueError("model.xml must not be empty")

    if receipt.get("player") != player_id or receipt.get("mode") != "sized_candidate":
        raise ValueError("Source receipt is not a sized export for the selected player")
    if receipt.get("profile") != profile:
        raise ValueError(
            "Source profile metadata does not match the current catalog profile"
        )
    if receipt.get("visual_definition") != "model.visual.json":
        raise ValueError("Source receipt does not identify model.visual.json")
    if physics_spec_sha256(spec) != receipt.get("physics_spec_sha256"):
        raise ValueError("Physics spec hash does not match the source receipt")
    if physics_spec_sha256(visual) != receipt.get("physics_spec_sha256"):
        raise ValueError(
            "Visual definition physics hash does not match the source receipt"
        )
    if (
        not isinstance(spec.get("coordinate_order"), list)
        or not spec["coordinate_order"]
    ):
        raise ValueError("Physics spec requires an ordered coordinate list")
    if receipt.get("coordinate_order") != spec["coordinate_order"]:
        raise ValueError("Receipt coordinate order differs from the physics spec")
    if receipt.get("export", {}).get("coordinate_order") != spec["coordinate_order"]:
        raise ValueError("MJCF receipt coordinate order differs from the physics spec")

    visual_physics = dict(visual)
    embedded_appearance = visual_physics.pop("appearance", None)
    if visual_physics != spec or embedded_appearance != appearance_record:
        raise ValueError(
            "Visual definition does not combine the exact physics and appearance files"
        )
    appearance = document_from_dict(appearance_record)
    if document_to_dict(appearance) != appearance_record:
        raise ValueError("Appearance file is not in canonical validated form")

    generated_xml, export_metadata = export_full_body_mjcf(visual_bytes)
    if generated_xml.encode("utf-8") != xml_bytes:
        raise ValueError("Stored MJCF does not match its visual model definition")
    if receipt.get("export", {}).get("mjcf_sha256") != export_metadata["mjcf_sha256"]:
        raise ValueError("MJCF hash differs from the source receipt")

    plant = get_plant("mujoco", visual_bytes)
    if not isinstance(plant, ScalarCoordinateUnits):
        raise ValueError("Compiled MuJoCo model does not report coordinate units")
    dofs = tuple(spec["coordinate_order"])
    units = tuple(plant.coordinate_units)
    if tuple(plant.coordinate_order) != dofs or len(units) != len(dofs):
        raise ValueError(
            "Compiled MuJoCo coordinate order differs from the source spec"
        )
    if any(unit not in {"m", "rad"} for unit in units):
        raise ValueError("Compiled MuJoCo model has unsupported coordinate units")

    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    profile_sha = sha(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    )
    xml_sha = sha(xml_bytes)
    visual_sha = sha(visual_bytes)
    receipt_sha = sha(receipt_bytes)
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_STORED) as archive:
        for name, payload in (
            ("model.visual.json", visual_bytes),
            ("receipt.json", receipt_bytes),
        ):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_STORED
            archive.writestr(info, payload)
    bundle = buffer.getvalue()
    package_sha = sha(b"\0".join((xml_bytes, spec_bytes, bundle)))
    session_id = f"{player_id}-catalog-{package_sha[:16]}"
    model_id = f"{player_id}-mujoco-{xml_sha[:16]}"
    bundle_id = f"{model_id}-bundle"
    model_version = str(receipt.get("model_version", "catalog-export-v1"))
    library.add_catalog_model_bundle(
        player_id=player_id,
        display_name=str(profile["display_name"]),
        session_id=session_id,
        model_id=model_id,
        bundle_asset_id=bundle_id,
        model_xml=xml_bytes,
        bundle=bundle,
        visual_sha256=visual_sha,
        source_manifest_sha256=receipt_sha,
        dofs=dofs,
        coordinate_units=units,
        source_profile_sha256=profile_sha,
        physics_spec_sha256=str(receipt["physics_spec_sha256"]),
        model_version=model_version,
    )
    return CatalogModelImport(
        player_id,
        session_id,
        model_id,
        bundle_id,
        "sha256:" + xml_sha,
        "sha256:" + visual_sha,
        "sha256:" + receipt_sha,
        str(receipt["physics_spec_sha256"]),
        units,
    )

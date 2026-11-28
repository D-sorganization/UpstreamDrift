"""Validate and import immutable sized catalog models into Necromatcher."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from src.shared.python.model_appearance.golfer_model_versions import (
    current_golfer_model,
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

from .necromatcher import CatalogModelBundle, NecromatcherLibrary
from .necromatcher_ranges import export_bound_native_xml


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
    mode: str
    model_version: str
    reference_baseline_id: str | None = None
    reference_baseline_sha256: str | None = None


def _read_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    raw = path.read_bytes()
    return _read_json_bytes(raw, label)


def _read_json_bytes(raw: bytes, label: str) -> tuple[bytes, dict[str, Any]]:
    """Parse an exact JSON object while retaining the original bytes."""
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return raw, value


def _trusted_reference_baseline(
    source: Path | bytes | None, source_id: str | None
) -> tuple[bytes, str]:
    """Read an explicitly supplied baseline and retain its caller provenance."""
    if source is None or not isinstance(source_id, str) or not source_id.strip():
        raise ValueError(
            "Appearance-only import requires an explicit trusted baseline and source id"
        )
    if isinstance(source, bytes):
        payload = source
    else:
        path = Path(source).expanduser()
        if path.is_symlink() or not path.is_file():
            raise ValueError("Reference baseline must be a regular file")
        payload = path.read_bytes()
    try:
        if not isinstance(json.loads(payload), dict):
            raise ValueError("Trusted reference-skin baseline must be an object")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Trusted reference-skin baseline is malformed") from exc
    return payload, source_id


def import_catalog_model(
    library: NecromatcherLibrary,
    player_id: str,
    source_directory: Path | str,
    *,
    reference_baseline: Path | bytes | None = None,
    reference_baseline_id: str | None = None,
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

    mode = receipt.get("mode")
    if receipt.get("player") != player_id or mode not in {
        "sized_candidate",
        "appearance_only",
    }:
        raise ValueError("Source receipt mode or player is not importable")
    if receipt.get("profile") != profile:
        raise ValueError(
            "Source profile metadata does not match the current catalog profile"
        )
    if receipt.get("visual_definition") != "model.visual.json":
        raise ValueError("Source receipt does not identify model.visual.json")
    export_record = receipt.get("export")
    if not isinstance(export_record, dict):
        raise ValueError("Source receipt export metadata is malformed")
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
    if export_record.get("coordinate_order") != spec["coordinate_order"]:
        raise ValueError("MJCF receipt coordinate order differs from the physics spec")

    baseline_bytes: bytes | None = None
    baseline_id: str | None = None
    baseline_sha: str | None = None
    if mode == "appearance_only":
        baseline_bytes, baseline_id = _trusted_reference_baseline(
            reference_baseline, reference_baseline_id
        )
        _, baseline_spec = _read_json_bytes(
            baseline_bytes, "trusted reference-skin baseline"
        )
        if (
            physics_spec_sha256(spec) != physics_spec_sha256(baseline_spec)
            or spec != baseline_spec
        ):
            raise ValueError(
                "Appearance-only physics differs from the trusted reference baseline"
            )
        baseline_sha = hashlib.sha256(baseline_bytes).hexdigest()

    visual_physics = dict(visual)
    embedded_appearance = visual_physics.pop("appearance", None)
    if visual_physics != spec or embedded_appearance != appearance_record:
        raise ValueError(
            "Visual definition does not combine the exact physics and appearance files"
        )
    appearance = document_from_dict(appearance_record)
    if document_to_dict(appearance) != appearance_record:
        raise ValueError("Appearance file is not in canonical validated form")

    generated_xml = export_bound_native_xml(visual_bytes)
    if generated_xml.encode("utf-8") != xml_bytes:
        raise ValueError("Stored MJCF does not match its visual model definition")
    if export_record.get("mjcf_sha256") != hashlib.sha256(
        generated_xml.encode("utf-8")
    ).hexdigest():
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
    bundle_members = [
        ("model.visual.json", visual_bytes),
        ("receipt.json", receipt_bytes),
    ]
    if baseline_bytes is not None:
        bundle_members.append(("reference-baseline.json", baseline_bytes))
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_STORED) as archive:
        for name, payload in bundle_members:
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_STORED
            archive.writestr(info, payload)
    bundle = buffer.getvalue()
    mode_id = "sized" if mode == "sized_candidate" else "reference-skin"
    package_sha = sha(
        b"\0".join(
            (
                mode.encode(),
                (baseline_id or "").encode(),
                (baseline_sha or "").encode(),
                xml_bytes,
                spec_bytes,
                bundle,
            )
        )
    )
    session_id = f"{player_id}-{mode_id}-catalog-{package_sha[:16]}"
    model_id = f"{player_id}-{mode_id}-mujoco-{xml_sha[:16]}"
    bundle_id = f"{model_id}-bundle"
    model_version = receipt.get("model_version", current_golfer_model().version)
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("Source export must identify a model version")
    library.add_catalog_model_bundle(
        CatalogModelBundle(
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
            candidate_mode=mode,
            reference_baseline_id=baseline_id,
            reference_baseline_sha256=baseline_sha,
        )
    )
    return CatalogModelImport(
        player_id=player_id,
        session_id=session_id,
        model_id=model_id,
        bundle_asset_id=bundle_id,
        model_sha256="sha256:" + xml_sha,
        visual_sha256="sha256:" + visual_sha,
        source_manifest_sha256="sha256:" + receipt_sha,
        physics_spec_sha256=str(receipt["physics_spec_sha256"]),
        coordinate_units=units,
        mode=mode,
        model_version=model_version,
        reference_baseline_id=baseline_id,
        reference_baseline_sha256=baseline_sha,
    )

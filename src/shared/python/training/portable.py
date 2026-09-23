"""Portable export/import of training job configs, model cards, and weights (NM-11, #10626).

Enforces cryptographic SHA-256 integrity verification, defends against zip-slip
path traversal, and provides trusted typed reconstruction for cross-tool mobility.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

from .config import TrainingConfig, TrainingFramework

MANIFEST_FILENAME = "manifest.json"
CONFIG_FILENAME = "job_config.json"
MODEL_CARD_FILENAME = "model_card.json"
PACKAGE_SCHEMA = "portable-job-package/1.0.0"


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export_job_package(
    output_path: Path,
    config: TrainingConfig,
    model_card: dict[str, Any],
    weights_path: Path | None = None,
) -> Path:
    """Bundle job config, model card, and optional weights into a verified zip archive.

    Args:
        output_path: Target archive path (.zip).
        config: TrainingConfig specification.
        model_card: Model card metadata dictionary.
        weights_path: Optional path to serialized weights file.

    Returns:
        The written output_path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "framework": config.framework.value,
        "entry_point": config.entry_point,
        "dataset_id": config.dataset_id,
        "output_dir": str(config.output_dir),
        "hyperparameters": dict(config.hyperparameters)
        if config.hyperparameters is not None
        else {},
    }
    config_bytes = json.dumps(config_dict, indent=2, sort_keys=True).encode("utf-8")
    card_bytes = json.dumps(model_card, indent=2, sort_keys=True).encode("utf-8")

    manifest_files: dict[str, str] = {
        CONFIG_FILENAME: _compute_sha256(config_bytes),
        MODEL_CARD_FILENAME: _compute_sha256(card_bytes),
    }

    weights_bytes: bytes | None = None
    weights_filename: str | None = None
    if weights_path is not None and weights_path.exists():
        weights_filename = weights_path.name
        weights_bytes = weights_path.read_bytes()
        manifest_files[weights_filename] = _compute_sha256(weights_bytes)

    # Compute composite digest over sorted entries
    hasher = hashlib.sha256()
    for fname in sorted(manifest_files.keys()):
        hasher.update(f"{fname}:{manifest_files[fname]}".encode())
    composite_digest = hasher.hexdigest()

    manifest_data = {
        "schema": PACKAGE_SCHEMA,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": manifest_files,
        "composite_digest": composite_digest,
    }
    manifest_bytes = json.dumps(manifest_data, indent=2, sort_keys=True).encode("utf-8")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(CONFIG_FILENAME, config_bytes)
        zf.writestr(MODEL_CARD_FILENAME, card_bytes)
        if weights_filename is not None and weights_bytes is not None:
            zf.writestr(weights_filename, weights_bytes)
        zf.writestr(MANIFEST_FILENAME, manifest_bytes)

    return output_path


def import_job_package(
    package_path: Path,
    target_dir: Path,
) -> tuple[TrainingConfig, dict[str, Any], Path | None]:
    """Verify package integrity, extract safely, and reconstruct configuration and card.

    Args:
        package_path: Path to .zip package.
        target_dir: Directory where files should be extracted.

    Returns:
        tuple of (reconstructed TrainingConfig, model_card dict, extracted weights Path or None).

    Raises:
        FileNotFoundError: If package_path does not exist.
        ValueError: On missing manifest, zip-slip path traversal attempt, or checksum mismatch.
    """
    if not package_path.exists():
        raise FileNotFoundError(f"Package not found: {package_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir_resolved = target_dir.resolve()

    with zipfile.ZipFile(package_path, "r") as zf:
        namelist = zf.namelist()
        if MANIFEST_FILENAME not in namelist:
            raise ValueError(
                f"Package {package_path.name} invalid: {MANIFEST_FILENAME} missing"
            )

        manifest_raw = zf.read(MANIFEST_FILENAME)
        try:
            manifest = json.loads(manifest_raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise ValueError(f"Corrupt manifest in {package_path.name}") from err

        expected_files = manifest.get("files", {})

        # 1. Defend against path traversal (Zip-Slip)
        for member_name in namelist:
            dest_path = (target_dir / member_name).resolve()
            if not str(dest_path).startswith(str(target_dir_resolved)):
                raise ValueError(
                    f"Illegal path traversal attempt: {member_name!r} resolves outside target directory"
                )

        # 2. Verify checksum integrity for each member in manifest
        for member_name, expected_sha in expected_files.items():
            if member_name not in namelist:
                raise ValueError(
                    f"Manifest declared {member_name!r} but not found in archive"
                )
            actual_bytes = zf.read(member_name)
            actual_sha = _compute_sha256(actual_bytes)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Checksum mismatch for {member_name}: expected {expected_sha}, got {actual_sha}"
                )

        # 3. Extract verified files
        zf.extractall(target_dir)

    # Reconstruct TrainingConfig
    config_file = target_dir / CONFIG_FILENAME
    config_dict = json.loads(config_file.read_text(encoding="utf-8"))
    config = TrainingConfig(
        framework=TrainingFramework(config_dict["framework"]),
        entry_point=config_dict["entry_point"],
        dataset_id=config_dict.get("dataset_id"),
        output_dir=Path(config_dict.get("output_dir", target_dir)),
        hyperparameters=config_dict.get("hyperparameters", {}),
    )

    # Reconstruct model card
    card_file = target_dir / MODEL_CARD_FILENAME
    model_card = json.loads(card_file.read_text(encoding="utf-8"))

    # Locate weights if present
    weights_path: Path | None = None
    for member_name in expected_files:
        if member_name not in (CONFIG_FILENAME, MODEL_CARD_FILENAME, MANIFEST_FILENAME):
            w_candidate = target_dir / member_name
            if w_candidate.exists():
                weights_path = w_candidate
                break

    return config, model_card, weights_path

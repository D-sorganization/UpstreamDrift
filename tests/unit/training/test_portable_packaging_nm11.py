"""Unit tests for NM-11 Portable Job, Model Card, and Weights Export/Import.

Governing Issue: #10626
Parent Epic: #10603
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile
import pytest

from src.shared.python.training import TrainingConfig, TrainingFramework
from src.shared.python.training.portable import (
    export_job_package,
    import_job_package,
)

pytestmark = pytest.mark.unit


def _sample_config(out_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point="neural_motion:train_masked_proposals",
        dataset_id="tour_dataset_v1",
        output_dir=out_dir,
        hyperparameters={
            "model_id": "driven_double_pendulum",
            "epochs": 10,
            "lr": 0.001,
        },
    )


def test_export_and_import_roundtrip(tmp_path: Path) -> None:
    """export_job_package and import_job_package faithfully preserve config, card, and weights."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    cfg = _sample_config(src_dir)

    model_card = {
        "model_id": "driven_double_pendulum",
        "schema_version": "1.0.0",
        "parameters_count": 12840,
        "validation_status": "QUALIFIED",
    }
    weights_path = src_dir / "weights.bin"
    weights_path.write_bytes(b"\x00\x01\x02\x03\x04\x05DEEP_NEURAL_WEIGHTS")

    package_zip = tmp_path / "package.zip"
    exported_path = export_job_package(
        output_path=package_zip,
        config=cfg,
        model_card=model_card,
        weights_path=weights_path,
    )
    assert exported_path == package_zip
    assert package_zip.exists()

    # Import into target directory
    dest_dir = tmp_path / "dest"
    imported_cfg, imported_card, imported_weights = import_job_package(
        package_path=package_zip,
        target_dir=dest_dir,
    )

    assert imported_cfg.entry_point == cfg.entry_point
    assert imported_cfg.framework == cfg.framework
    assert imported_cfg.dataset_id == cfg.dataset_id
    assert imported_cfg.hyperparameters == cfg.hyperparameters

    assert imported_card["model_id"] == model_card["model_id"]
    assert imported_card["parameters_count"] == model_card["parameters_count"]

    assert imported_weights is not None
    assert imported_weights.exists()
    assert imported_weights.read_bytes() == weights_path.read_bytes()


def test_tamper_detection_checksum_mismatch(tmp_path: Path) -> None:
    """import_job_package detects file corruption or tampering via SHA-256 mismatch."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    cfg = _sample_config(src_dir)
    weights_path = src_dir / "weights.bin"
    weights_path.write_bytes(b"ORIGINAL_DATA")

    package_zip = tmp_path / "package.zip"
    export_job_package(
        output_path=package_zip,
        config=cfg,
        model_card={"model_id": "test"},
        weights_path=weights_path,
    )

    # Tamper with weights inside zip
    tampered_zip = tmp_path / "tampered.zip"
    with zipfile.ZipFile(package_zip, "r") as zin:
        with zipfile.ZipFile(tampered_zip, "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "weights.bin":
                    data = b"TAMPERED_MALICIOUS_DATA"
                zout.writestr(item, data)

    dest_dir = tmp_path / "dest"
    with pytest.raises(ValueError, match="Checksum mismatch|Integrity error"):
        import_job_package(tampered_zip, target_dir=dest_dir)


def test_path_traversal_zip_slip_rejected(tmp_path: Path) -> None:
    """import_job_package rejects archives containing directory-traversal paths."""
    malicious_zip = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious_zip, "w") as zf:
        zf.writestr("../../evil.txt", "exploit")
        manifest = {
            "schema": "portable-job-package/1.0.0",
            "files": {"../../evil.txt": "dummy_sha"},
        }
        zf.writestr("manifest.json", json.dumps(manifest))

    dest_dir = tmp_path / "dest"
    with pytest.raises(ValueError, match="Illegal path traversal|outside target"):
        import_job_package(malicious_zip, target_dir=dest_dir)


def test_missing_manifest_rejected(tmp_path: Path) -> None:
    """import_job_package fails closed when manifest.json is absent."""
    empty_zip = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip, "w") as zf:
        zf.writestr("other.txt", "data")

    dest_dir = tmp_path / "dest"
    with pytest.raises(ValueError, match="manifest.json missing"):
        import_job_package(empty_zip, target_dir=dest_dir)

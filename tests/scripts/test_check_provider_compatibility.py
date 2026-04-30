from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.check_provider_compatibility import main


def test_main_reports_success_for_compatible_provider_manifest(
    tmp_path: Path, capsys
) -> None:
    provider_root = tmp_path / "provider"
    models_dir = provider_root / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "swing.xml").write_text("<mujoco />", encoding="utf-8")
    manifest_path = provider_root / "model_pack.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "manifest_version": "1.0.0",
                "pack_id": "mujoco-pack",
                "pack_name": "MuJoCo Pack",
                "provider": "mujoco_models",
                "models": [
                    {
                        "id": "mujoco_swing",
                        "name": "MuJoCo Swing",
                        "description": "Provider swing model",
                        "type": "mjcf",
                        "path": "models/swing.xml",
                        "engine_type": "mujoco",
                        "capabilities": ["ik"],
                        "identity": {
                            "canonical_id": "golf.swing.main",
                            "motion_family": "golf-swing",
                            "exercise": "driver-full-swing",
                            "humanoid": "golf-athlete",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--provider-root",
            str(provider_root),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["is_compatible"] is True
    assert payload["provider"] == "mujoco_models"


def test_main_reports_machine_readable_failures_for_invalid_manifest(
    tmp_path: Path, capsys
) -> None:
    provider_root = tmp_path / "provider"
    provider_root.mkdir()
    manifest_path = provider_root / "model_pack.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "manifest_version": "1.0.0",
                "pack_id": "broken-pack",
                "pack_name": "Broken Pack",
                "provider": "drake_models",
                "models": [
                    {
                        "id": "broken_model",
                        "name": "Broken Model",
                        "description": "Missing artifact and identity",
                        "type": "urdf",
                        "path": "models/missing.urdf",
                        "engine_type": "drake",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--provider-root",
            str(provider_root),
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    issue_codes = {issue["code"] for issue in payload["issues"]}
    assert "missing_artifact_path" in issue_codes
    assert "missing_canonical_identity" in issue_codes

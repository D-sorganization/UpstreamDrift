"""Tests for launcher provider compatibility harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from src.launchers.launcher_provider_compatibility import (
    assert_launcher_provider_compatibility,
    assert_provider_manifest_compatibility,
    evaluate_launcher_model_compatibility,
    is_engine_runtime_available,
    validate_provider_manifest,
)


class LocalModel:
    id = "local_model"
    path = "models/model.urdf"


class ExternalModel:
    id = "external_model"
    provider = "drake_models"
    source_root = "../Drake_Models"
    path = "models/model.urdf"
    working_dir = "python"
    python_paths = ["src", "bindings"]


class CrossEngineModelA:
    id = "mujoco_swing"
    provider = "mujoco_models"
    path = "models/swing.xml"
    engine_type = "mujoco"

    class identity:
        canonical_id = "golf.swing.main"
        motion_family = "golf-swing"
        exercise = "driver-full-swing"
        humanoid = "golf-athlete"


class CrossEngineModelB:
    id = "drake_swing"
    provider = "drake_models"
    path = "models/swing.urdf"
    engine_type = "drake"

    class identity:
        canonical_id = "golf.swing.main"
        motion_family = "golf-swing"
        exercise = "driver-full-swing"
        humanoid = "golf-athlete"


class DrakeProviderModel:
    id = "drake_model"
    provider = "drake_models"
    path = "models/drake.urdf"
    engine_type = "drake"


def test_evaluate_launcher_model_compatibility_for_local_model(tmp_path: Path) -> None:
    model_file = tmp_path / "models" / "model.urdf"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("<robot />", encoding="utf-8")

    results = evaluate_launcher_model_compatibility([LocalModel()], tmp_path)

    assert len(results) == 1
    assert results[0].is_compatible
    assert results[0].provider == "local"
    assert results[0].artifact_path == model_file


def test_evaluate_launcher_model_compatibility_reports_missing_provider_paths(
    tmp_path: Path,
) -> None:
    results = evaluate_launcher_model_compatibility([ExternalModel()], tmp_path)

    assert len(results) == 1
    assert not results[0].is_compatible
    assert any(issue.code == "missing_source_root" for issue in results[0].issues)
    assert any(issue.code == "missing_artifact_path" for issue in results[0].issues)
    assert any(issue.code == "missing_working_directory" for issue in results[0].issues)


def test_assert_launcher_provider_compatibility_raises_on_failures(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="external_model"):
        assert_launcher_provider_compatibility([ExternalModel()], tmp_path)


def test_evaluate_launcher_model_compatibility_preserves_canonical_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mujoco_file = tmp_path / "models" / "swing.xml"
    drake_file = tmp_path / "models" / "swing.urdf"
    mujoco_file.parent.mkdir(parents=True)
    mujoco_file.write_text("<mujoco />", encoding="utf-8")
    drake_file.write_text("<robot />", encoding="utf-8")

    monkeypatch.setattr(
        "src.launchers.launcher_provider_compatibility.is_engine_runtime_available",
        lambda engine_type: True,
    )

    results = evaluate_launcher_model_compatibility(
        [CrossEngineModelA(), CrossEngineModelB()],
        tmp_path,
    )

    assert [result.canonical_id for result in results] == [
        "golf.swing.main",
        "golf.swing.main",
    ]
    assert all(result.is_compatible for result in results)


def test_evaluate_launcher_model_compatibility_distinguishes_runtime_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_file = tmp_path / "models" / "drake.urdf"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("<robot />", encoding="utf-8")

    monkeypatch.setattr(
        "src.launchers.launcher_provider_compatibility.is_engine_runtime_available",
        lambda engine_type: False,
    )

    results = evaluate_launcher_model_compatibility([DrakeProviderModel()], tmp_path)

    assert len(results) == 1
    assert not results[0].is_compatible
    assert any(issue.category == "runtime_unavailable" for issue in results[0].issues)


def test_is_engine_runtime_available_accepts_stubbed_module_without_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pydrake", object())

    assert is_engine_runtime_available("drake") is True


def test_validate_provider_manifest_reports_machine_readable_diagnostics(
    tmp_path: Path,
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
                        "description": "Missing asset path",
                        "type": "urdf",
                        "path": "models/missing.urdf",
                        "engine_type": "drake",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_provider_manifest(manifest_path, provider_root)

    assert report.provider == "drake_models"
    assert not report.is_compatible
    assert any(issue.code == "missing_artifact_path" for issue in report.issues)
    assert all(issue.message for issue in report.issues)
    assert all(isinstance(issue.context, dict) for issue in report.issues)


def test_assert_provider_manifest_compatibility_raises_with_model_and_issue_codes(
    tmp_path: Path,
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
                        "description": "Missing asset path",
                        "type": "urdf",
                        "path": "models/missing.urdf",
                        "engine_type": "drake",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="broken_model") as exc_info:
        assert_provider_manifest_compatibility(manifest_path, provider_root)

    assert "missing_artifact_path" in str(exc_info.value)


def test_validate_provider_manifest_allows_tool_provider_without_canonical_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_root = tmp_path / "Tools"
    utility_file = provider_root / "src" / "pendulum_launcher.py"
    utility_file.parent.mkdir(parents=True)
    utility_file.write_text("print('pendulum')\n", encoding="utf-8")
    manifest_path = provider_root / "model_pack.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "manifest_version": "1.0.0",
                "pack_id": "tools-pack",
                "pack_name": "Tools",
                "provider": "tools",
                "models": [
                    {
                        "id": "pendulum_suite",
                        "name": "Pendulum Suite",
                        "description": "Utility pendulum workflows",
                        "type": "special_app",
                        "path": "src/pendulum_launcher.py",
                        "capabilities": ["pendulum", "simulation"],
                        "launcher": {
                            "category": "tool",
                            "logo": "golf_logo.svg",
                            "status": "utility",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.launchers.launcher_provider_compatibility.is_engine_runtime_available",
        lambda engine_type: True,
    )

    report = validate_provider_manifest(manifest_path, provider_root)

    assert report.is_compatible
    assert report.results[0].is_compatible
    assert report.results[0].canonical_id is None

"""Tests for launcher provider compatibility harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.launchers.launcher_provider_compatibility import (
    assert_launcher_provider_compatibility,
    evaluate_launcher_model_compatibility,
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
    assert any("source root does not exist" in issue for issue in results[0].issues)
    assert any("artifact path does not exist" in issue for issue in results[0].issues)
    assert any(
        "working directory does not exist" in issue for issue in results[0].issues
    )


def test_assert_launcher_provider_compatibility_raises_on_failures(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="external_model"):
        assert_launcher_provider_compatibility([ExternalModel()], tmp_path)


def test_evaluate_launcher_model_compatibility_preserves_canonical_identity(
    tmp_path: Path,
) -> None:
    mujoco_file = tmp_path / "models" / "swing.xml"
    drake_file = tmp_path / "models" / "swing.urdf"
    mujoco_file.parent.mkdir(parents=True)
    mujoco_file.write_text("<mujoco />", encoding="utf-8")
    drake_file.write_text("<robot />", encoding="utf-8")

    results = evaluate_launcher_model_compatibility(
        [CrossEngineModelA(), CrossEngineModelB()],
        tmp_path,
    )

    assert [result.canonical_id for result in results] == [
        "golf.swing.main",
        "golf.swing.main",
    ]
    assert all(result.is_compatible for result in results)

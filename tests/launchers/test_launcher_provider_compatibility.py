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

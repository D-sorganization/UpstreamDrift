"""Regression tests for URDF/MJCF model path containment."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException
from src.api.routes import models as models_module


def _write_urdf(path: Path, robot_name: str) -> None:
    path.write_text(
        (
            f'<robot name="{robot_name}">'
            '<link name="base_link">'
            '<visual><geometry><box size="1 1 1"/></geometry></visual>'
            "</link>"
            "</robot>"
        ),
        encoding="utf-8",
    )


def test_discover_models_skips_escape_symlink_and_keeps_internal_symlink(
    tmp_path, monkeypatch
) -> None:
    """Discovery should keep contained models and reject escaped symlinks."""
    models_dir = tmp_path / "models"
    nested_dir = models_dir / "nested"
    outside_dir = tmp_path / "outside"
    nested_dir.mkdir(parents=True)
    outside_dir.mkdir()

    nested_file = nested_dir / "nested_leg.urdf"
    internal_alias = models_dir / "alias_leg.urdf"
    outside_file = outside_dir / "escape.urdf"
    escape_alias = models_dir / "escape.urdf"

    _write_urdf(nested_file, "nested_leg")
    _write_urdf(outside_file, "escape")

    try:
        internal_alias.symlink_to(nested_file)
        escape_alias.symlink_to(outside_file)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    monkeypatch.setattr(models_module, "_find_project_root", lambda: tmp_path)
    monkeypatch.setattr(models_module, "_MODEL_DIRS", [Path("models")])

    discovered = models_module._discover_models()
    names = {model["name"] for model in discovered}
    paths = {model["path"] for model in discovered}

    assert "nested_leg" in names
    assert "alias_leg" in names
    assert "escape" not in names
    assert "models/escape.urdf" not in paths


def test_get_model_urdf_serves_nested_model(tmp_path, monkeypatch) -> None:
    """A normal nested model should still be discovered and served."""
    models_dir = tmp_path / "models"
    nested_dir = models_dir / "nested"
    nested_dir.mkdir(parents=True)

    nested_file = nested_dir / "nested_leg.urdf"
    _write_urdf(nested_file, "nested_leg")

    monkeypatch.setattr(models_module, "_find_project_root", lambda: tmp_path)
    monkeypatch.setattr(models_module, "_MODEL_DIRS", [Path("models")])

    response = asyncio.run(models_module.get_model_urdf("nested_leg", logger=None))

    assert response.model_name == "nested_leg"
    assert response.root_link == "base_link"
    assert [link.link_name for link in response.links] == ["base_link"]


def test_get_model_urdf_rejects_escape_symlink_on_final_validation(
    tmp_path, monkeypatch
) -> None:
    """Serving should re-check containment immediately before reading."""
    models_dir = tmp_path / "models"
    outside_dir = tmp_path / "outside"
    models_dir.mkdir()
    outside_dir.mkdir()

    outside_file = outside_dir / "escape.urdf"
    escape_alias = models_dir / "escape.urdf"
    _write_urdf(outside_file, "escape")

    try:
        escape_alias.symlink_to(outside_file)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    monkeypatch.setattr(models_module, "_find_project_root", lambda: tmp_path)
    monkeypatch.setattr(models_module, "_MODEL_DIRS", [Path("models")])
    monkeypatch.setattr(
        models_module,
        "_discover_models",
        lambda: [
            {
                "name": "escape",
                "format": "urdf",
                "path": "models/escape.urdf",
            }
        ],
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(models_module.get_model_urdf("escape", logger=None))

    assert excinfo.value.status_code == 404

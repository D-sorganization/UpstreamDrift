"""Regression tests for URDF/MJCF model path containment."""

from __future__ import annotations

import asyncio
from pathlib import Path

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

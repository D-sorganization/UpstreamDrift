"""Tests for the /models/mesh-asset endpoint (issue #8406).

The endpoint serves glTF mesh files referenced by URDF ``mesh_path`` values
for the frontend URDFViewer. It must resolve relative paths against the
allowed model directories only and reject anything escaping them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from src.api.routes import models as models_module

pytestmark = pytest.mark.unit

_GLB_BYTES = b"glTF\x02\x00\x00\x00fake-binary-payload"


@pytest.fixture()
def mesh_tree(tmp_path, monkeypatch) -> Path:
    """Point model discovery at a tmp tree with one .glb asset."""
    meshes_dir = tmp_path / "models" / "meshes"
    meshes_dir.mkdir(parents=True)
    (meshes_dir / "club.glb").write_bytes(_GLB_BYTES)
    (meshes_dir / "scene.gltf").write_text("{}", encoding="utf-8")
    # A file OUTSIDE the allowed model root that traversal must never reach.
    (tmp_path / "secret.glb").write_bytes(b"outside")

    monkeypatch.setattr(models_module, "_find_project_root", lambda: tmp_path)
    monkeypatch.setattr(models_module, "_MODEL_DIRS", [Path("models")])
    return tmp_path


def _get(path: str):
    return asyncio.run(models_module.get_model_mesh_asset(path=path))


class TestMeshAssetServing:
    def test_serves_glb_with_binary_content_type(self, mesh_tree: Path) -> None:
        response = _get("meshes/club.glb")
        assert response.media_type == "model/gltf-binary"
        assert Path(response.path) == mesh_tree / "models" / "meshes" / "club.glb"

    def test_serves_gltf_with_json_content_type(self, mesh_tree: Path) -> None:
        response = _get("meshes/scene.gltf")
        assert response.media_type == "model/gltf+json"

    def test_package_prefix_is_stripped(self, mesh_tree: Path) -> None:
        response = _get("package://meshes/club.glb")
        assert response.media_type == "model/gltf-binary"

    def test_path_with_model_dir_prefix_resolves(self, mesh_tree: Path) -> None:
        response = _get("models/meshes/club.glb")
        assert Path(response.path) == mesh_tree / "models" / "meshes" / "club.glb"


class TestMeshAssetRejection:
    def test_parent_traversal_is_rejected(self, mesh_tree: Path) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _get("../secret.glb")
        assert exc_info.value.status_code == 400

    def test_nested_parent_traversal_is_rejected(self, mesh_tree: Path) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _get("meshes/../../secret.glb")
        assert exc_info.value.status_code == 400

    def test_absolute_path_is_rejected(self, mesh_tree: Path) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _get(str(mesh_tree / "models" / "meshes" / "club.glb"))
        assert exc_info.value.status_code == 400

    def test_unsupported_extension_is_rejected(self, mesh_tree: Path) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _get("meshes/club.stl")
        assert exc_info.value.status_code == 400

    def test_empty_path_is_rejected(self, mesh_tree: Path) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _get("   ")
        assert exc_info.value.status_code == 400

    def test_missing_asset_is_404(self, mesh_tree: Path) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _get("meshes/nonexistent.glb")
        assert exc_info.value.status_code == 404

    def test_symlink_escape_is_not_served(self, mesh_tree: Path) -> None:
        link = mesh_tree / "models" / "meshes" / "evil.glb"
        link.symlink_to(mesh_tree / "secret.glb")
        with pytest.raises(HTTPException) as exc_info:
            _get("meshes/evil.glb")
        assert exc_info.value.status_code == 404

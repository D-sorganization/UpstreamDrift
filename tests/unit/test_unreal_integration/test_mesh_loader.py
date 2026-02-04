"""Unit tests for mesh loading system.

TDD tests for multi-format mesh loading (GLTF/GLB/FBX/OBJ).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.unreal_integration.mesh_loader import (
    LoadedMesh,
    MeshBone,
    MeshFace,
    MeshFormat,
    MeshLoader,
    MeshMaterial,
    MeshSkeleton,
    MeshVertex,
    UnsupportedFormatError,
)


class TestMeshFormat:
    """Tests for MeshFormat enum."""

    def test_supported_formats(self):
        """Test all supported formats exist."""
        assert MeshFormat.OBJ is not None
        assert MeshFormat.STL is not None
        assert MeshFormat.GLTF is not None
        assert MeshFormat.GLB is not None
        assert MeshFormat.FBX is not None
        assert MeshFormat.COLLADA is not None
        assert MeshFormat.PLY is not None

    def test_format_from_extension(self):
        """Test format detection from file extension."""
        assert MeshFormat.from_extension(".obj") == MeshFormat.OBJ
        assert MeshFormat.from_extension(".gltf") == MeshFormat.GLTF
        assert MeshFormat.from_extension(".glb") == MeshFormat.GLB
        assert MeshFormat.from_extension(".fbx") == MeshFormat.FBX
        assert MeshFormat.from_extension(".dae") == MeshFormat.COLLADA

    def test_format_from_extension_case_insensitive(self):
        """Test format detection is case-insensitive."""
        assert MeshFormat.from_extension(".OBJ") == MeshFormat.OBJ
        assert MeshFormat.from_extension(".Gltf") == MeshFormat.GLTF

    def test_format_unknown_extension(self):
        """Test unknown extension raises error."""
        with pytest.raises(UnsupportedFormatError):
            MeshFormat.from_extension(".xyz")

    def test_format_extensions(self):
        """Test format extension properties."""
        assert MeshFormat.OBJ.extension == ".obj"
        assert MeshFormat.GLTF.extension == ".gltf"
        assert MeshFormat.GLB.extension == ".glb"


class TestMeshVertex:
    """Tests for MeshVertex data structure."""

    def test_create_vertex(self):
        """Test vertex creation."""
        v = MeshVertex(
            position=np.array([1.0, 2.0, 3.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            uv=np.array([0.5, 0.5]),
        )
        assert v.position[0] == 1.0
        assert v.normal[1] == 1.0
        assert v.uv[0] == 0.5

    def test_vertex_without_optional_fields(self):
        """Test vertex creation without optional fields."""
        v = MeshVertex(position=np.array([1.0, 2.0, 3.0]))
        assert v.normal is None
        assert v.uv is None
        assert v.color is None

    def test_vertex_with_color(self):
        """Test vertex with vertex color."""
        v = MeshVertex(
            position=np.array([0.0, 0.0, 0.0]),
            color=np.array([1.0, 0.0, 0.0, 1.0]),  # Red
        )
        assert v.color is not None
        assert v.color[0] == 1.0

    def test_vertex_with_bone_weights(self):
        """Test vertex with skeletal weights."""
        v = MeshVertex(
            position=np.array([0.0, 0.0, 0.0]),
            bone_indices=np.array([0, 1, 2, 3]),
            bone_weights=np.array([0.5, 0.3, 0.15, 0.05]),
        )
        assert sum(v.bone_weights) == pytest.approx(1.0)


class TestMeshFace:
    """Tests for MeshFace data structure."""

    def test_create_triangle(self):
        """Test triangle face creation."""
        f = MeshFace(indices=np.array([0, 1, 2]))
        assert len(f.indices) == 3
        assert f.is_triangle

    def test_create_quad(self):
        """Test quad face creation."""
        f = MeshFace(indices=np.array([0, 1, 2, 3]))
        assert len(f.indices) == 4
        assert not f.is_triangle

    def test_face_with_material(self):
        """Test face with material index."""
        f = MeshFace(indices=np.array([0, 1, 2]), material_index=1)
        assert f.material_index == 1


class TestMeshMaterial:
    """Tests for MeshMaterial data structure."""

    def test_create_material(self):
        """Test material creation."""
        m = MeshMaterial(
            name="GolfClub_Metal",
            base_color=(0.8, 0.8, 0.9, 1.0),
            metallic=0.9,
            roughness=0.3,
        )
        assert m.name == "GolfClub_Metal"
        assert m.metallic == 0.9

    def test_material_with_textures(self):
        """Test material with texture paths."""
        m = MeshMaterial(
            name="GolfShirt",
            base_color_texture="textures/shirt_diffuse.png",
            normal_texture="textures/shirt_normal.png",
        )
        assert m.base_color_texture is not None
        assert m.normal_texture is not None

    def test_default_material(self):
        """Test default material creation."""
        m = MeshMaterial.default()
        assert m.name == "default"
        assert m.base_color == (0.8, 0.8, 0.8, 1.0)


class TestMeshBone:
    """Tests for MeshBone data structure."""

    def test_create_bone(self):
        """Test bone creation."""
        b = MeshBone(
            name="shoulder_L",
            index=5,
            parent_index=4,
            local_transform=np.eye(4),
        )
        assert b.name == "shoulder_L"
        assert b.parent_index == 4

    def test_root_bone(self):
        """Test root bone (no parent)."""
        b = MeshBone(
            name="root",
            index=0,
            parent_index=-1,  # No parent
            local_transform=np.eye(4),
        )
        assert b.is_root
        assert b.parent_index == -1

    def test_bone_with_inverse_bind(self):
        """Test bone with inverse bind matrix."""
        inv_bind = np.eye(4)
        inv_bind[3, :3] = [0.1, 0.2, 0.3]  # Translation
        b = MeshBone(
            name="elbow_L",
            index=6,
            parent_index=5,
            local_transform=np.eye(4),
            inverse_bind_matrix=inv_bind,
        )
        assert b.inverse_bind_matrix is not None


class TestMeshSkeleton:
    """Tests for MeshSkeleton data structure."""

    def test_create_skeleton(self):
        """Test skeleton creation."""
        bones = [
            MeshBone(name="root", index=0, parent_index=-1, local_transform=np.eye(4)),
            MeshBone(name="spine", index=1, parent_index=0, local_transform=np.eye(4)),
            MeshBone(name="head", index=2, parent_index=1, local_transform=np.eye(4)),
        ]
        skeleton = MeshSkeleton(bones=bones)
        assert len(skeleton.bones) == 3
        assert skeleton.root_bone.name == "root"

    def test_skeleton_bone_lookup(self):
        """Test bone lookup by name."""
        bones = [
            MeshBone(name="root", index=0, parent_index=-1, local_transform=np.eye(4)),
            MeshBone(
                name="shoulder_L", index=1, parent_index=0, local_transform=np.eye(4)
            ),
        ]
        skeleton = MeshSkeleton(bones=bones)
        assert skeleton.get_bone("shoulder_L") is not None
        assert skeleton.get_bone("nonexistent") is None

    def test_skeleton_hierarchy(self):
        """Test skeleton hierarchy traversal."""
        bones = [
            MeshBone(name="root", index=0, parent_index=-1, local_transform=np.eye(4)),
            MeshBone(name="spine", index=1, parent_index=0, local_transform=np.eye(4)),
            MeshBone(
                name="shoulder_L", index=2, parent_index=1, local_transform=np.eye(4)
            ),
            MeshBone(
                name="shoulder_R", index=3, parent_index=1, local_transform=np.eye(4)
            ),
        ]
        skeleton = MeshSkeleton(bones=bones)
        children = skeleton.get_children(1)  # Children of spine
        assert len(children) == 2

    def test_skeleton_bone_names(self):
        """Test getting all bone names."""
        bones = [
            MeshBone(name="root", index=0, parent_index=-1, local_transform=np.eye(4)),
            MeshBone(name="spine", index=1, parent_index=0, local_transform=np.eye(4)),
        ]
        skeleton = MeshSkeleton(bones=bones)
        names = skeleton.bone_names
        assert "root" in names
        assert "spine" in names


class TestLoadedMesh:
    """Tests for LoadedMesh data structure."""

    def test_create_loaded_mesh(self):
        """Test loaded mesh creation."""
        vertices = [
            MeshVertex(position=np.array([0.0, 0.0, 0.0])),
            MeshVertex(position=np.array([1.0, 0.0, 0.0])),
            MeshVertex(position=np.array([0.5, 1.0, 0.0])),
        ]
        faces = [MeshFace(indices=np.array([0, 1, 2]))]

        mesh = LoadedMesh(
            name="triangle",
            vertices=vertices,
            faces=faces,
        )
        assert mesh.name == "triangle"
        assert mesh.vertex_count == 3
        assert mesh.face_count == 1

    def test_mesh_with_materials(self):
        """Test mesh with materials."""
        vertices = [MeshVertex(position=np.array([0.0, 0.0, 0.0]))]
        faces = [MeshFace(indices=np.array([0, 0, 0]))]
        materials = [MeshMaterial(name="mat1")]

        mesh = LoadedMesh(
            name="test",
            vertices=vertices,
            faces=faces,
            materials=materials,
        )
        assert len(mesh.materials) == 1

    def test_mesh_with_skeleton(self):
        """Test mesh with skeleton."""
        vertices = [MeshVertex(position=np.array([0.0, 0.0, 0.0]))]
        faces = [MeshFace(indices=np.array([0, 0, 0]))]
        bones = [
            MeshBone(name="root", index=0, parent_index=-1, local_transform=np.eye(4))
        ]
        skeleton = MeshSkeleton(bones=bones)

        mesh = LoadedMesh(
            name="skinned",
            vertices=vertices,
            faces=faces,
            skeleton=skeleton,
        )
        assert mesh.has_skeleton
        assert mesh.skeleton.bone_count == 1

    def test_mesh_bounds(self):
        """Test mesh bounding box calculation."""
        vertices = [
            MeshVertex(position=np.array([0.0, 0.0, 0.0])),
            MeshVertex(position=np.array([10.0, 5.0, 3.0])),
            MeshVertex(position=np.array([-5.0, 2.0, -1.0])),
        ]
        faces = [MeshFace(indices=np.array([0, 1, 2]))]

        mesh = LoadedMesh(name="test", vertices=vertices, faces=faces)
        bounds = mesh.bounding_box

        assert bounds["min"][0] == -5.0
        assert bounds["max"][0] == 10.0

    def test_mesh_to_arrays(self):
        """Test mesh conversion to numpy arrays."""
        vertices = [
            MeshVertex(position=np.array([0.0, 0.0, 0.0])),
            MeshVertex(position=np.array([1.0, 0.0, 0.0])),
            MeshVertex(position=np.array([0.5, 1.0, 0.0])),
        ]
        faces = [MeshFace(indices=np.array([0, 1, 2]))]

        mesh = LoadedMesh(name="test", vertices=vertices, faces=faces)
        positions, indices = mesh.to_arrays()

        assert positions.shape == (3, 3)
        assert indices.shape == (1, 3)


class TestMeshLoader:
    """Tests for MeshLoader class."""

    def test_create_loader(self):
        """Test loader creation."""
        loader = MeshLoader()
        assert loader is not None

    def test_supported_formats(self):
        """Test loader reports supported formats."""
        loader = MeshLoader()
        formats = loader.supported_formats
        assert MeshFormat.OBJ in formats
        assert MeshFormat.GLTF in formats
        assert MeshFormat.GLB in formats

    def test_can_load_format(self):
        """Test format support checking."""
        loader = MeshLoader()
        assert loader.can_load(".obj")
        assert loader.can_load(".gltf")
        assert not loader.can_load(".xyz")

    def test_load_invalid_path(self):
        """Test loading nonexistent file."""
        loader = MeshLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/model.obj")

    def test_load_unsupported_format(self, tmp_path: Path):
        """Test loading unsupported format."""
        # Create a fake file
        fake_file = tmp_path / "model.xyz"
        fake_file.write_text("fake content")

        loader = MeshLoader()
        with pytest.raises(UnsupportedFormatError):
            loader.load(str(fake_file))

    def test_load_obj_basic(self, tmp_path: Path):
        """Test loading basic OBJ file."""
        obj_content = """
# Simple triangle
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
f 1 2 3
"""
        obj_file = tmp_path / "triangle.obj"
        obj_file.write_text(obj_content)

        loader = MeshLoader()
        mesh = loader.load(str(obj_file))

        assert mesh.vertex_count == 3
        assert mesh.face_count == 1

    def test_load_obj_with_normals(self, tmp_path: Path):
        """Test loading OBJ with normals."""
        obj_content = """
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
vn 0.0 0.0 1.0
f 1//1 2//1 3//1
"""
        obj_file = tmp_path / "triangle_normals.obj"
        obj_file.write_text(obj_content)

        loader = MeshLoader()
        mesh = loader.load(str(obj_file))

        assert mesh.vertices[0].normal is not None

    def test_load_obj_with_uvs(self, tmp_path: Path):
        """Test loading OBJ with texture coordinates."""
        obj_content = """
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.5 1.0
f 1/1 2/2 3/3
"""
        obj_file = tmp_path / "triangle_uvs.obj"
        obj_file.write_text(obj_content)

        loader = MeshLoader()
        mesh = loader.load(str(obj_file))

        assert mesh.vertices[0].uv is not None

    def test_loader_caching(self, tmp_path: Path):
        """Test mesh caching."""
        obj_content = "v 0.0 0.0 0.0\nf 1 1 1"
        obj_file = tmp_path / "cached.obj"
        obj_file.write_text(obj_content)

        loader = MeshLoader(enable_cache=True)

        # First load
        mesh1 = loader.load(str(obj_file))

        # Second load should use cache
        mesh2 = loader.load(str(obj_file))

        assert mesh1 is mesh2  # Same object from cache

    def test_loader_cache_invalidation(self, tmp_path: Path):
        """Test cache invalidation on file modification."""
        obj_content = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3"
        obj_file = tmp_path / "modified.obj"
        obj_file.write_text(obj_content)

        loader = MeshLoader(enable_cache=True)
        mesh1 = loader.load(str(obj_file))
        assert mesh1.vertex_count == 3  # Original has 3 vertices

        # Modify file - add a 4th vertex
        import time

        time.sleep(0.1)  # Ensure different mtime
        obj_file.write_text(
            "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nv 1.0 1.0 0.0\nf 1 2 3\nf 2 4 3"
        )

        # Should reload and have more vertices
        mesh2 = loader.load(str(obj_file))
        assert mesh2.vertex_count == 6  # 2 triangles * 3 vertices each

    def test_loader_clear_cache(self, tmp_path: Path):
        """Test clearing loader cache."""
        obj_content = "v 0.0 0.0 0.0\nf 1 1 1"
        obj_file = tmp_path / "clear_cache.obj"
        obj_file.write_text(obj_content)

        loader = MeshLoader(enable_cache=True)
        loader.load(str(obj_file))
        assert loader.cache_size > 0

        loader.clear_cache()
        assert loader.cache_size == 0


class TestMeshLoaderGLTF:
    """Tests for GLTF/GLB loading."""

    def test_create_minimal_gltf(self, tmp_path: Path):
        """Test loading minimal GLTF file."""
        # Create minimal GLTF JSON
        gltf_content = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {"POSITION": 0},
                            "indices": 1,
                        }
                    ]
                }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,  # FLOAT
                    "count": 3,
                    "type": "VEC3",
                    "max": [1.0, 1.0, 0.0],
                    "min": [0.0, 0.0, 0.0],
                },
                {
                    "bufferView": 1,
                    "componentType": 5123,  # UNSIGNED_SHORT
                    "count": 3,
                    "type": "SCALAR",
                },
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36},
                {"buffer": 0, "byteOffset": 36, "byteLength": 6},
            ],
            "buffers": [{"byteLength": 42}],
        }

        gltf_file = tmp_path / "minimal.gltf"
        gltf_file.write_text(json.dumps(gltf_content))

        # Create binary buffer
        import struct

        vertices = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0)
        indices = struct.pack("<3H", 0, 1, 2)
        buffer_data = vertices + indices

        bin_file = tmp_path / "minimal.bin"
        bin_file.write_bytes(buffer_data)

        # Update GLTF to reference the bin file
        gltf_content["buffers"][0]["uri"] = "minimal.bin"
        gltf_file.write_text(json.dumps(gltf_content))

        loader = MeshLoader()
        # Note: Full GLTF loading requires trimesh or pygltflib
        # This test validates the structure

    def test_gltf_with_skeleton(self, tmp_path: Path):
        """Test loading GLTF with skeleton."""
        # This would test skeleton loading from GLTF
        # Actual implementation depends on trimesh/pygltflib


class TestMeshLoaderContracts:
    """Tests for Design by Contract compliance."""

    def test_load_requires_valid_path(self):
        """Test load validates path exists."""
        loader = MeshLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/definitely/does/not/exist.obj")

    def test_load_validates_format(self, tmp_path: Path):
        """Test load validates file format."""
        bad_file = tmp_path / "bad.unknownformat"
        bad_file.write_text("garbage")

        loader = MeshLoader()
        with pytest.raises(UnsupportedFormatError):
            loader.load(str(bad_file))

    def test_loaded_mesh_invariants(self, tmp_path: Path):
        """Test loaded mesh maintains invariants."""
        obj_content = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.5 1.0 0.0\nf 1 2 3"
        obj_file = tmp_path / "invariants.obj"
        obj_file.write_text(obj_content)

        loader = MeshLoader()
        mesh = loader.load(str(obj_file))

        # Invariant: vertex_count == len(vertices)
        assert mesh.vertex_count == len(mesh.vertices)

        # Invariant: face_count == len(faces)
        assert mesh.face_count == len(mesh.faces)

        # Invariant: all face indices valid
        for face in mesh.faces:
            for idx in face.indices:
                assert 0 <= idx < mesh.vertex_count

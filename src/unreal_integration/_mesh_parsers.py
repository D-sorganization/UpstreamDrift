from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import numpy as np

from ._mesh_types import (
    LoadedMesh,
    MeshFace,
    MeshLoadError,
    MeshVertex,
)

logger = logging.getLogger(__name__)


def load_obj(path: Path) -> LoadedMesh:  # noqa: C901
    if not (path is not None):
        raise ValueError("path must be provided")
    vertices: list[MeshVertex] = []
    faces: list[MeshFace] = []
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    uvs: list[np.ndarray] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == "v":
                positions.append(
                    np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                )
            elif parts[0] == "vn":
                normals.append(
                    np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                )
            elif parts[0] == "vt":
                uvs.append(np.array([float(parts[1]), float(parts[2])]))
            elif parts[0] == "f":
                face_indices = []
                for i in range(1, len(parts)):
                    vertex_data = parts[i].split("/")
                    v_idx = int(vertex_data[0]) - 1

                    vt_idx = None
                    vn_idx = None
                    if len(vertex_data) > 1 and vertex_data[1]:
                        vt_idx = int(vertex_data[1]) - 1
                    if len(vertex_data) > 2 and vertex_data[2]:
                        vn_idx = int(vertex_data[2]) - 1

                    vertex = MeshVertex(
                        position=positions[v_idx],
                        normal=(
                            normals[vn_idx]
                            if vn_idx is not None and vn_idx < len(normals)
                            else None
                        ),
                        uv=(
                            uvs[vt_idx]
                            if vt_idx is not None and vt_idx < len(uvs)
                            else None
                        ),
                    )
                    vertices.append(vertex)
                    face_indices.append(len(vertices) - 1)

                faces.append(MeshFace(indices=np.array(face_indices)))

    return LoadedMesh(
        name=path.stem,
        vertices=vertices,
        faces=faces,
    )


def load_stl(path: Path) -> LoadedMesh:
    if not (path is not None):
        raise ValueError("path must be provided")
    vertices: list[MeshVertex] = []
    faces: list[MeshFace] = []

    with open(path, "rb") as f:
        header = f.read(80)
        is_binary = not header.strip().startswith(b"solid")

    if is_binary:
        with open(path, "rb") as f:
            f.read(80)
            num_triangles = struct.unpack("<I", f.read(4))[0]

            for _ in range(num_triangles):
                normal = struct.unpack("<3f", f.read(12))
                normal_arr = np.array(normal)

                face_indices = []
                for _ in range(3):
                    vertex = struct.unpack("<3f", f.read(12))
                    vertices.append(
                        MeshVertex(
                            position=np.array(vertex),
                            normal=normal_arr,
                        )
                    )
                    face_indices.append(len(vertices) - 1)

                faces.append(MeshFace(indices=np.array(face_indices)))

                f.read(2)
    else:
        with open(path) as f:
            current_normal = None
            face_vertices: list[int] = []

            for line in f:
                line = line.strip()
                if line.startswith("facet normal"):
                    parts = line.split()
                    current_normal = np.array(
                        [float(parts[2]), float(parts[3]), float(parts[4])]
                    )
                elif line.startswith("vertex"):
                    parts = line.split()
                    vertices.append(
                        MeshVertex(
                            position=np.array(
                                [float(parts[1]), float(parts[2]), float(parts[3])]
                            ),
                            normal=current_normal,
                        )
                    )
                    face_vertices.append(len(vertices) - 1)
                elif line.startswith("endfacet"):
                    if len(face_vertices) == 3:
                        faces.append(MeshFace(indices=np.array(face_vertices)))
                    face_vertices = []

    return LoadedMesh(
        name=path.stem,
        vertices=vertices,
        faces=faces,
    )


def load_gltf(path: Path) -> LoadedMesh:
    try:
        import trimesh

        scene = trimesh.load(str(path))

        if isinstance(scene, trimesh.Scene):
            meshes = list(scene.geometry.values())
            if not meshes:
                raise MeshLoadError("No meshes found in scene", str(path))
            mesh_data = meshes[0]
            for m in meshes[1:]:
                mesh_data = trimesh.util.concatenate([mesh_data, m])
        else:
            mesh_data = scene

        vertices = [
            MeshVertex(
                position=mesh_data.vertices[i],
                normal=(
                    mesh_data.vertex_normals[i]
                    if hasattr(mesh_data, "vertex_normals")
                    else None
                ),
            )
            for i in range(len(mesh_data.vertices))
        ]

        faces = [MeshFace(indices=face) for face in mesh_data.faces]

        skeleton = _extract_gltf_skeleton(path)

        return LoadedMesh(
            name=path.stem,
            vertices=vertices,
            faces=faces,
            skeleton=skeleton,
        )

    except ImportError:
        return _load_gltf_basic(path)


def _extract_gltf_skeleton(path: Path):
    from ._mesh_types import MeshSkeleton, MeshBone

    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic == b"glTF":
                f.read(8)
                chunk_len = struct.unpack("<I", f.read(4))[0]
                chunk_type = f.read(4)
                if chunk_type != b"JSON":
                    return None
                data = json.loads(f.read(chunk_len).decode("utf-8"))
            else:
                with open(path, encoding="utf-8") as text_f:
                    data = json.load(text_f)

        if "nodes" not in data or "skins" not in data:
            return None
        skins = data["skins"]
        if not skins:
            return None

        nodes = data["nodes"]
        skin = skins[0]
        joints = skin.get("joints", [])
        if not joints:
            return None

        parents = {}
        for i, node in enumerate(nodes):
            for child_idx in node.get("children", []):
                parents[child_idx] = i

        skeleton = MeshSkeleton()
        for i, joint_node_idx in enumerate(joints):
            if joint_node_idx >= len(nodes):
                continue
            node = nodes[joint_node_idx]
            name = node.get("name", f"joint_{i}")

            parent_node_idx = parents.get(joint_node_idx, -1)
            parent_index = -1
            if parent_node_idx in joints:
                parent_index = joints.index(parent_node_idx)

            transform = np.eye(4)
            if "matrix" in node:
                transform = np.array(node["matrix"]).reshape(4, 4).T

            bone = MeshBone(
                name=name, index=i, parent_index=parent_index, local_transform=transform
            )
            skeleton.bones.append(bone)

        return skeleton if skeleton.bones else None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to extract GLTF skeleton: {e}")
        return None


def _load_gltf_basic(path: Path) -> LoadedMesh:
    if path.suffix.lower() == ".glb":
        raise MeshLoadError("GLB loading requires trimesh library", str(path))

    with open(path) as f:
        gltf = json.load(f)

    logger.warning("Basic GLTF loading - some features may not be supported")

    if "meshes" not in gltf or not gltf["meshes"]:
        raise MeshLoadError("No meshes found in GLTF", str(path))

    raise MeshLoadError(
        "GLTF accessor/buffer parsing requires the trimesh library",
        str(path),
    )


def load_fbx(path: Path) -> LoadedMesh:
    try:
        import trimesh

        mesh = trimesh.load(str(path))

        if isinstance(mesh, trimesh.Scene):
            meshes = list(mesh.geometry.values())
            if not meshes:
                raise MeshLoadError("No meshes found in FBX", str(path))
            mesh = meshes[0]

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Expected trimesh.Trimesh, got {type(mesh).__name__}")
        vertices = [
            MeshVertex(position=mesh.vertices[i]) for i in range(len(mesh.vertices))
        ]

        faces = [MeshFace(indices=face) for face in mesh.faces]

        return LoadedMesh(
            name=path.stem,
            vertices=vertices,
            faces=faces,
        )

    except ImportError as e:
        raise MeshLoadError(
            "FBX loading requires trimesh library: pip install trimesh[easy]",
            str(path),
        ) from e


def load_collada(path: Path) -> LoadedMesh:
    try:
        import trimesh

        mesh = trimesh.load(str(path))

        if isinstance(mesh, trimesh.Scene):
            meshes = list(mesh.geometry.values())
            if not meshes:
                raise MeshLoadError("No meshes found in COLLADA", str(path))
            mesh = meshes[0]

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Expected trimesh.Trimesh, got {type(mesh).__name__}")
        vertices = [
            MeshVertex(position=mesh.vertices[i]) for i in range(len(mesh.vertices))
        ]

        faces = [MeshFace(indices=face) for face in mesh.faces]

        return LoadedMesh(
            name=path.stem,
            vertices=vertices,
            faces=faces,
        )

    except ImportError as e:
        raise MeshLoadError(
            "COLLADA loading requires trimesh library", str(path)
        ) from e


def load_ply(path: Path) -> LoadedMesh:
    try:
        import trimesh

        mesh = trimesh.load(str(path))
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Expected trimesh.Trimesh, got {type(mesh).__name__}")

        vertices = [
            MeshVertex(position=mesh.vertices[i]) for i in range(len(mesh.vertices))
        ]

        faces = [MeshFace(indices=face) for face in mesh.faces]

        return LoadedMesh(
            name=path.stem,
            vertices=vertices,
            faces=faces,
        )

    except ImportError as e:
        raise MeshLoadError("PLY loading requires trimesh library", str(path)) from e

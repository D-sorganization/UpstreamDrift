from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MeshLoadError(Exception):
    def __init__(
        self, message: str, path: str | None = None, cause: Exception | None = None
    ) -> None:
        if not (message is not None):
            raise ValueError("message must be provided")
        super().__init__(message)
        self.path = path
        self.cause = cause


class UnsupportedFormatError(MeshLoadError):
    def __init__(self, extension: str, path: str | None = None) -> None:
        if not (extension is not None):
            raise ValueError("extension must be provided")
        super().__init__(f"Unsupported mesh format: {extension}", path)
        self.extension = extension


class MeshFormat(Enum):
    OBJ = "obj"
    STL = "stl"
    GLTF = "gltf"
    GLB = "glb"
    FBX = "fbx"
    COLLADA = "dae"
    PLY = "ply"

    @property
    def extension(self) -> str:
        return f".{self.value}"

    @classmethod
    def from_extension(cls, ext: str) -> MeshFormat:
        ext_lower = ext.lower().lstrip(".")
        for fmt in cls:
            if fmt.value == ext_lower:
                return fmt
        raise UnsupportedFormatError(ext)


@dataclass
class MeshVertex:
    position: np.ndarray
    normal: np.ndarray | None = None
    uv: np.ndarray | None = None
    color: np.ndarray | None = None
    bone_indices: np.ndarray | None = None
    bone_weights: np.ndarray | None = None


@dataclass
class MeshFace:
    indices: np.ndarray
    material_index: int = 0

    @property
    def is_triangle(self) -> bool:
        return len(self.indices) == 3


@dataclass
class MeshMaterial:
    name: str = "default"
    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_color_texture: str | None = None
    normal_texture: str | None = None
    metallic_roughness_texture: str | None = None
    occlusion_texture: str | None = None
    emissive_texture: str | None = None

    @classmethod
    def default(cls) -> MeshMaterial:
        return cls(name="default")


@dataclass
class MeshBone:
    name: str
    index: int
    parent_index: int
    local_transform: np.ndarray
    inverse_bind_matrix: np.ndarray | None = None

    @property
    def is_root(self) -> bool:
        return self.parent_index < 0


@dataclass
class MeshSkeleton:
    bones: list[MeshBone] = field(default_factory=list)

    @property
    def bone_count(self) -> int:
        return len(self.bones)

    @property
    def bone_names(self) -> list[str]:
        return [bone.name for bone in self.bones]

    @property
    def root_bone(self) -> MeshBone | None:
        for bone in self.bones:
            if bone.is_root:
                return bone
        return None

    def get_bone(self, name: str) -> MeshBone | None:
        if not (name is not None):
            raise ValueError("name must be provided")
        for bone in self.bones:
            if bone.name == name:
                return bone
        return None

    def get_bone_by_index(self, index: int) -> MeshBone | None:
        if not (index is not None):
            raise ValueError("index must be provided")
        for bone in self.bones:
            if bone.index == index:
                return bone
        return None

    def get_children(self, parent_index: int) -> list[MeshBone]:
        return [bone for bone in self.bones if bone.parent_index == parent_index]


@dataclass
class LoadedMesh:
    name: str
    vertices: list[MeshVertex]
    faces: list[MeshFace]
    materials: list[MeshMaterial] = field(default_factory=list)
    skeleton: MeshSkeleton | None = None
    source_path: str | None = None
    format: MeshFormat | None = None

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def has_skeleton(self) -> bool:
        return self.skeleton is not None and self.skeleton.bone_count > 0

    @property
    def has_normals(self) -> bool:
        return any(v.normal is not None for v in self.vertices)

    @property
    def has_uvs(self) -> bool:
        return any(v.uv is not None for v in self.vertices)

    @property
    def bounding_box(self) -> dict[str, np.ndarray]:
        if not self.vertices:
            return {"min": np.zeros(3), "max": np.zeros(3)}
        positions = np.array([v.position for v in self.vertices])
        return {
            "min": np.min(positions, axis=0),
            "max": np.max(positions, axis=0),
        }

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        positions = np.array([v.position for v in self.vertices])
        indices = np.array([f.indices for f in self.faces])
        return positions, indices

    def get_normals_array(self) -> np.ndarray | None:
        if not self.has_normals:
            return None
        return np.array(
            [v.normal if v.normal is not None else [0, 0, 0] for v in self.vertices]
        )

    def get_uvs_array(self) -> np.ndarray | None:
        if not self.has_uvs:
            return None
        return np.array([v.uv if v.uv is not None else [0, 0] for v in self.vertices])

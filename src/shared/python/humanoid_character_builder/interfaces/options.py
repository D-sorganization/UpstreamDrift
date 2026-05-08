"""Export options for humanoid character models."""

from __future__ import annotations

from dataclasses import dataclass

from humanoid_character_builder.generators.mesh_generator import MeshGeneratorBackend
from humanoid_character_builder.mesh.inertia_calculator import InertiaMode


@dataclass
class ExportOptions:
    """Options for exporting character models."""

    # URDF options
    urdf_filename: str = "humanoid.urdf"
    include_collision: bool = True

    # Mesh options
    generate_meshes: bool = True
    mesh_format: str = "stl"  # stl, obj, dae
    mesh_backend: MeshGeneratorBackend = MeshGeneratorBackend.PRIMITIVE

    # Inertia options
    inertia_mode: InertiaMode = InertiaMode.PRIMITIVE
    density_kg_m3: float = 1050.0

    # Output structure
    create_package_structure: bool = True
    mesh_subdirectory: str = "meshes"
    config_subdirectory: str = "config"

    # Include additional files
    save_config: bool = True
    config_format: str = "yaml"  # yaml or json

#!/usr/bin/env python3
"""Verification script for viewer backends."""

import sys
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unreal_integration.viewer_backends import (
    create_viewer,
    ViewerConfig,
    BackendType,
    ViewerBackend,
)
from src.unreal_integration.mesh_loader import LoadedMesh, MeshVertex, MeshFace
from src.unreal_integration.geometry import Vector3, Quaternion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_mesh() -> LoadedMesh:
    """Create a simple test mesh (triangle)."""
    vertices = [
        MeshVertex(position=np.array([0.0, 0.0, 0.0])),
        MeshVertex(position=np.array([1.0, 0.0, 0.0])),
        MeshVertex(position=np.array([0.0, 1.0, 0.0])),
    ]
    faces = [MeshFace(indices=np.array([0, 1, 2]))]
    return LoadedMesh(name="test_mesh", vertices=vertices, faces=faces)

def verify_backend(backend_type: str):
    """Verify a specific backend."""
    logger.info(f"Verifying {backend_type} backend...")

    try:
        backend = create_viewer(backend_type)
    except NotImplementedError:
        logger.warning(f"{backend_type} not implemented yet.")
        return
    except Exception as e:
        logger.error(f"Failed to create {backend_type} backend: {e}")
        return

    try:
        backend.initialize()
        logger.info(f"{backend_type} initialized.")

        mesh = create_test_mesh()
        name = backend.add_mesh(mesh, name="test_obj")
        logger.info(f"Added mesh: {name}")

        backend.update_transform(
            name,
            position=Vector3(1.0, 2.0, 3.0),
            rotation=Quaternion.identity()
        )
        logger.info("Updated transform.")

        backend.render()
        logger.info("Rendered frame.")

        backend.remove_object(name)
        logger.info("Removed object.")

        backend.shutdown()
        logger.info(f"{backend_type} shutdown complete.")

    except Exception as e:
        logger.error(f"Error during {backend_type} verification: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main verification function."""
    verify_backend("mock")
    print("-" * 40)
    verify_backend("pyvista")
    print("-" * 40)
    verify_backend("unreal_bridge")

if __name__ == "__main__":
    main()

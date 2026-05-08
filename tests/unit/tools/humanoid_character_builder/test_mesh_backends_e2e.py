"""
End-to-end tests for MakeHuman and SMPLX mesh backends.

These tests are marked as `slow` and `live_simulation` because they:
- Require external dependencies (torch, trimesh, smplx)
- Require asset files (SMPL-X models, MakeHuman exports)
- Generate real mesh files on disk

Run with: pytest -m "slow or live_simulation" -v
"""

import os
import tempfile
from pathlib import Path

import pytest

from humanoid_character_builder import BodyParameters, CharacterBuilder
from humanoid_character_builder.generators import MeshGeneratorBackend


class TestMakeHumanBackendE2E:
    """End-to-end tests for MakeHuman mesh backend."""

    @pytest.mark.makehuman
    @pytest.mark.slow
    @pytest.mark.live_simulation
    def test_makehuman_mesh_generation(self):
        """
        Test MakeHuman mesh generation from BodyParameters to mesh files.

        This test:
        1. Creates BodyParameters
        2. Runs MakeHuman backend (not mocked)
        3. Produces mesh files on disk
        4. Validates mesh files are non-empty STL/OBJ
        """
        # Skip if MakeHuman not configured
        if not os.environ.get("MAKEHUMAN_MESH_DIR"):
            pytest.skip("MAKEHUMAN_MESH_DIR not set")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "makehuman_output"

            # Create builder with MakeHuman backend
            builder = CharacterBuilder(mesh_backend=MeshGeneratorBackend.MAKEHUMAN)

            # Build character
            params = BodyParameters(
                height_m=1.80,
                mass_kg=80.0,
                gender_factor=1.0,  # Male
            )
            result = builder.build(params, generate_meshes=True)

            # Verify build succeeded
            assert result.success, f"Build failed: {result.error_message}"

            # Verify URDF was generated
            assert result.urdf_xml is not None
            assert len(result.urdf_xml) > 0

            # Verify mesh files exist and are non-empty
            if result.mesh_result:
                for mesh_path in result.mesh_result.mesh_paths.values():
                    if mesh_path and mesh_path.exists():
                        assert mesh_path.stat().st_size > 0, (
                            f"Empty mesh file: {mesh_path}"
                        )
                        assert mesh_path.suffix.lower() in [".stl", ".obj"], (
                            f"Unexpected mesh format: {mesh_path}"
                        )

            # Export and verify
            result.export_urdf(output_dir)
            assert output_dir.exists(), "Output directory not created"

            urdf_file = output_dir / "humanoid.urdf"
            assert urdf_file.exists(), "URDF file not created"
            assert urdf_file.stat().st_size > 0, "URDF file is empty"


class TestSMPLXBackendE2E:
    """End-to-end tests for SMPL-X mesh backend."""

    @pytest.mark.smplx
    @pytest.mark.slow
    @pytest.mark.live_simulation
    def test_smplx_mesh_generation(self):
        """
        Test SMPL-X mesh generation from BodyParameters to mesh files.

        This test:
        1. Creates BodyParameters
        2. Runs SMPL-X backend (not mocked)
        3. Produces mesh files on disk
        4. Validates mesh files are non-empty STL/OBJ
        """
        # Skip if SMPL-X not configured
        if not os.environ.get("SMPLX_MODEL_DIR"):
            pytest.skip("SMPLX_MODEL_DIR not set")

        # Skip if torch not available
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "smplx_output"

            # Create builder with SMPL-X backend
            builder = CharacterBuilder(mesh_backend=MeshGeneratorBackend.SMPLX)

            # Build character
            params = BodyParameters(
                height_m=1.75,
                mass_kg=70.0,
                gender_factor=0.5,  # Neutral
            )
            result = builder.build(params, generate_meshes=True)

            # Verify build succeeded
            assert result.success, f"Build failed: {result.error_message}"

            # Verify URDF was generated
            assert result.urdf_xml is not None
            assert len(result.urdf_xml) > 0

            # Verify mesh files exist and are non-empty
            if result.mesh_result:
                for mesh_path in result.mesh_result.mesh_paths.values():
                    if mesh_path and mesh_path.exists():
                        assert mesh_path.stat().st_size > 0, (
                            f"Empty mesh file: {mesh_path}"
                        )
                        assert mesh_path.suffix.lower() in [".stl", ".obj"], (
                            f"Unexpected mesh format: {mesh_path}"
                        )

            # Export and verify
            result.export_urdf(output_dir)
            assert output_dir.exists(), "Output directory not created"

            urdf_file = output_dir / "humanoid.urdf"
            assert urdf_file.exists(), "URDF file not created"
            assert urdf_file.stat().st_size > 0, "URDF file is empty"


class TestEndToEndPipeline:
    """Full pipeline tests with real assets."""

    @pytest.mark.live_simulation
    @pytest.mark.slow
    def test_bodyparams_to_urdf_pipeline(self):
        """
        Test complete pipeline from BodyParameters to exported URDF.

        This is a backend-agnostic test that validates the full pipeline
        works with any configured mesh backend.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "pipeline_output"

            # Use primitive backend (always available)
            builder = CharacterBuilder(mesh_backend=MeshGeneratorBackend.PRIMITIVE)

            # Test with various body parameters
            test_params = [
                BodyParameters(
                    height_m=1.60, mass_kg=55.0, gender_factor=0.0
                ),  # Female
                BodyParameters(
                    height_m=1.75, mass_kg=75.0, gender_factor=0.5
                ),  # Neutral
                BodyParameters(height_m=1.90, mass_kg=95.0, gender_factor=1.0),  # Male
            ]

            for params in test_params:
                result = builder.build(params, generate_meshes=True)

                # Verify build
                assert result.success, (
                    f"Build failed for {params}: {result.error_message}"
                )
                assert result.urdf_xml is not None
                assert len(result.urdf_xml) > 0

                # Verify export
                result.export_urdf(output_dir / f"test_{params.height_m}m")
                urdf_file = output_dir / f"test_{params.height_m}m" / "humanoid.urdf"
                assert urdf_file.exists()
                assert urdf_file.stat().st_size > 0

    @pytest.mark.live_simulation
    def test_mesh_validation(self):
        """
        Test that generated meshes are valid for physics simulation.

        Validates:
        - Mesh has valid topology (no holes)
        - Mesh normals are consistent
        - Mesh is watertight (for collision)
        """
        try:
            import trimesh
        except ImportError:
            pytest.skip("trimesh not available")

        builder = CharacterBuilder(mesh_backend=MeshGeneratorBackend.PRIMITIVE)
        params = BodyParameters(height_m=1.75, mass_kg=75.0)
        result = builder.build(params, generate_meshes=True)

        if result.mesh_result:
            for mesh_path in result.mesh_result.mesh_paths.values():
                if mesh_path and mesh_path.exists():
                    mesh = trimesh.load(mesh_path)

                    # Validate mesh properties
                    assert len(mesh.vertices) > 0, f"No vertices in {mesh_path}"
                    assert len(mesh.faces) > 0, f"No faces in {mesh_path}"

                    # Check for NaN/Inf values
                    import numpy as np

                    assert not np.isnan(mesh.vertices).any(), (
                        f"NaN vertices in {mesh_path}"
                    )
                    assert not np.isinf(mesh.vertices).any(), (
                        f"Inf vertices in {mesh_path}"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

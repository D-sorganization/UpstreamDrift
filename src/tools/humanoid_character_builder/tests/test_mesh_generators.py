"""
Unit tests for SMPL-X and MakeHuman mesh generators.

Tests use mocked external dependencies (smplx, trimesh, subprocess) so that
the full pipeline logic can be validated without installing heavy optional
packages.

See issues #979 (MakeHuman) and #980 (SMPL-X).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from humanoid_character_builder.core.body_parameters import (
    BodyParameters,
    GenderModel,
)
from humanoid_character_builder.generators.mesh_generator import (
    GeneratedMeshResult,
    MakeHumanMeshGenerator,
    MeshGenerator,
    MeshGeneratorBackend,
    MeshGeneratorInterface,
    SMPLXMeshGenerator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_params(**overrides: Any) -> BodyParameters:
    """Create default BodyParameters with optional overrides."""
    kwargs: dict[str, Any] = {
        "height_m": 1.80,
        "mass_kg": 80.0,
    }
    kwargs.update(overrides)
    return BodyParameters(**kwargs)


# ---------------------------------------------------------------------------
# SMPL-X Generator Tests  (See issue #980)
# ---------------------------------------------------------------------------


class TestSMPLXBetaConversion:
    """Test BodyParameters -> SMPL-X beta parameter conversion."""

    def test_default_params_produce_near_zero_betas(self):
        params = BodyParameters()  # 1.75 m, 75 kg, average
        betas = SMPLXMeshGenerator._convert_params_to_betas(params)
        assert betas.shape == (SMPLXMeshGenerator.NUM_BETAS,)
        # Default body close to SMPL-X mean -> betas should be small
        assert np.abs(betas).max() < 5.0

    def test_tall_heavy_person(self):
        params = _default_params(height_m=2.00, mass_kg=110.0)
        betas = SMPLXMeshGenerator._convert_params_to_betas(params)
        # beta[0] should be positive (tall)
        assert betas[0] > 0
        # beta[1] should be positive (high BMI)
        assert betas[1] > 0

    def test_short_light_person(self):
        params = _default_params(height_m=1.50, mass_kg=45.0)
        betas = SMPLXMeshGenerator._convert_params_to_betas(params)
        assert betas[0] < 0  # shorter than mean
        # BMI = 45 / 1.50^2 = 20.0, below mean of 22 -> negative beta
        assert betas[1] < 0  # lower BMI

    def test_proportion_factors_map(self):
        params = _default_params(
            shoulder_width_factor=1.2,
            hip_width_factor=0.9,
            arm_length_factor=1.1,
            leg_length_factor=1.1,
            torso_length_factor=1.05,
        )
        betas = SMPLXMeshGenerator._convert_params_to_betas(params)
        assert betas[2] > 0  # wider shoulders
        assert betas[3] < 0  # narrower hips
        assert betas[4] > 0  # longer arms
        assert betas[5] > 0  # longer legs
        assert betas[6] > 0  # longer torso

    def test_muscularity_mapping(self):
        lean = _default_params(muscularity=0.1)
        buff = _default_params(muscularity=0.9)
        b_lean = SMPLXMeshGenerator._convert_params_to_betas(lean)
        b_buff = SMPLXMeshGenerator._convert_params_to_betas(buff)
        assert b_buff[7] > b_lean[7]


class TestSMPLXGenderString:
    """Test gender string selection."""

    def test_male(self):
        params = BodyParameters(gender_model=GenderModel.MALE)
        assert SMPLXMeshGenerator._gender_string(params) == "male"

    def test_female(self):
        params = BodyParameters(gender_model=GenderModel.FEMALE)
        assert SMPLXMeshGenerator._gender_string(params) == "female"

    def test_neutral(self):
        params = BodyParameters(gender_model=GenderModel.NEUTRAL)
        assert SMPLXMeshGenerator._gender_string(params) == "neutral"


class TestSMPLXSegmentMesh:
    """Test the static _segment_mesh helper."""

    def test_segment_extracts_correct_vertices(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 2, 2]])
        faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])

        seg_v, seg_f = SMPLXMeshGenerator._segment_mesh(verts, faces, 0, 4)
        # Should include only the first two faces (all vertices in [0,4))
        assert seg_v.shape[0] == 4
        assert seg_f.shape[0] == 2

    def test_empty_segment(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])

        seg_v, seg_f = SMPLXMeshGenerator._segment_mesh(verts, faces, 0, 1)
        # Face uses vertices 0,1,2 but only vertex 0 is in [0,1)
        assert seg_f.shape[0] == 0


class TestSMPLXAvailability:
    """Test is_available and error paths."""

    def test_unavailable_when_smplx_missing(self):
        with patch(
            "humanoid_character_builder.generators.mesh_generator.SMPLX_AVAILABLE",
            False,
        ):
            gen = SMPLXMeshGenerator()
            assert gen.is_available is False

    def test_unavailable_when_model_dir_missing(self):
        with patch(
            "humanoid_character_builder.generators.mesh_generator.SMPLX_AVAILABLE",
            True,
        ):
            gen = SMPLXMeshGenerator(model_dir="/nonexistent/path")
            assert gen.is_available is False

    def test_returns_error_result_when_smplx_missing(self):
        with patch(
            "humanoid_character_builder.generators.mesh_generator.SMPLX_AVAILABLE",
            False,
        ):
            gen = SMPLXMeshGenerator()
            result = gen.generate(_default_params(), Path("/tmp/out"))
            assert result.success is False
            assert "smplx" in result.error_message.lower()

    def test_returns_error_when_trimesh_missing(self):
        with (
            patch(
                "humanoid_character_builder.generators.mesh_generator.SMPLX_AVAILABLE",
                True,
            ),
            patch(
                "humanoid_character_builder.generators.mesh_generator.TRIMESH_AVAILABLE",
                False,
            ),
        ):
            gen = SMPLXMeshGenerator(model_dir="/nonexistent")
            result = gen.generate(_default_params(), Path("/tmp/out"))
            assert result.success is False
            assert "trimesh" in result.error_message.lower()


class TestSMPLXGenerate:
    """Test the full SMPL-X generate pipeline with mocked smplx module."""

    def _mock_smplx_output(self, n_verts: int = 10475, n_faces: int = 20000):
        """Create a mock SMPL-X model output."""
        mock_output = MagicMock()
        mock_output.vertices = MagicMock()
        mock_output.vertices.detach.return_value.cpu.return_value.numpy.return_value.squeeze.return_value = np.random.randn(
            n_verts, 3
        ).astype(np.float32)

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_model.faces = np.random.randint(0, n_verts, size=(n_faces, 3)).astype(
            np.int64
        )

        return mock_model

    @patch("humanoid_character_builder.generators.mesh_generator.SMPLX_AVAILABLE", True)
    @patch(
        "humanoid_character_builder.generators.mesh_generator.TRIMESH_AVAILABLE", True
    )
    @patch("humanoid_character_builder.generators.mesh_generator._smplx_module")
    @patch("humanoid_character_builder.generators.mesh_generator._trimesh_module")
    def test_generate_produces_stl_files(
        self, mock_trimesh, mock_smplx, tmp_path: Path
    ):
        """Verify that generate produces per-segment STL files."""
        import torch  # noqa: F401

        mock_model = self._mock_smplx_output()
        mock_smplx.create.return_value = mock_model

        # Mock trimesh.Trimesh to track exports
        exported_files: list[str] = []

        class FakeTrimesh:
            def __init__(self, vertices=None, faces=None):
                self.vertices = vertices
                self.faces = faces

            def export(self, path: str):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).touch()
                exported_files.append(path)

            @property
            def convex_hull(self):
                return self

        mock_trimesh.Trimesh = FakeTrimesh

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        gen = SMPLXMeshGenerator(model_dir=model_dir)
        params = _default_params()
        output_dir = tmp_path / "output"

        result = gen.generate(params, output_dir)

        assert result.success is True
        assert result.metadata["backend"] == "smplx"
        assert len(result.mesh_paths) > 0
        assert len(result.collision_paths) > 0
        assert len(result.vertex_groups) > 0

    @patch("humanoid_character_builder.generators.mesh_generator.SMPLX_AVAILABLE", True)
    @patch(
        "humanoid_character_builder.generators.mesh_generator.TRIMESH_AVAILABLE", True
    )
    def test_generate_handles_exception_gracefully(self, tmp_path: Path):
        """Verify graceful failure when SMPL-X forward pass throws."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        gen = SMPLXMeshGenerator(model_dir=model_dir)

        with patch(
            "humanoid_character_builder.generators.mesh_generator._smplx_module"
        ) as mock_smplx:
            mock_smplx.create.side_effect = RuntimeError("Model load failed")
            result = gen.generate(_default_params(), tmp_path / "out")

        assert result.success is False
        assert "error" in result.error_message.lower()


class TestSMPLXSupportedSegments:
    """Test get_supported_segments."""

    def test_returns_expected_segments(self):
        gen = SMPLXMeshGenerator()
        segments = gen.get_supported_segments()
        assert "head" in segments
        assert "left_hand" in segments
        assert "right_foot" in segments
        assert len(segments) == len(SMPLXMeshGenerator.SMPLX_SEGMENT_VERTEX_RANGES)


# ---------------------------------------------------------------------------
# MakeHuman Generator Tests  (See issue #979)
# ---------------------------------------------------------------------------


class TestMakeHumanParamConversion:
    """Test BodyParameters -> MakeHuman modifier conversion."""

    def test_default_params_conversion(self):
        params = BodyParameters()
        modifiers = MakeHumanMeshGenerator._convert_params_to_makehuman(params)

        assert "macrodetails/Gender" in modifiers
        assert "macrodetails/Age" in modifiers
        assert "macrodetails-universal/Muscle" in modifiers
        assert "macrodetails-universal/Weight" in modifiers
        assert "__height_scale__" in modifiers

    def test_gender_mapping(self):
        male = BodyParameters(gender_model=GenderModel.MALE)
        female = BodyParameters(gender_model=GenderModel.FEMALE)

        m = MakeHumanMeshGenerator._convert_params_to_makehuman(male)
        f = MakeHumanMeshGenerator._convert_params_to_makehuman(female)

        assert m["macrodetails/Gender"] == 1.0
        assert f["macrodetails/Gender"] == 0.0

    def test_age_normalisation(self):
        young = BodyParameters()
        young.appearance.age_years = 20.0
        old = BodyParameters()
        old.appearance.age_years = 60.0

        m_young = MakeHumanMeshGenerator._convert_params_to_makehuman(young)
        m_old = MakeHumanMeshGenerator._convert_params_to_makehuman(old)

        assert m_young["macrodetails/Age"] < m_old["macrodetails/Age"]
        assert 0.0 <= m_young["macrodetails/Age"] <= 1.0

    def test_height_scale(self):
        tall = _default_params(height_m=1.90)
        modifiers = MakeHumanMeshGenerator._convert_params_to_makehuman(tall)
        assert modifiers["__height_scale__"] > 1.0

    def test_proportion_modifiers(self):
        params = _default_params(
            shoulder_width_factor=1.2,
            hip_width_factor=0.9,
            arm_length_factor=1.1,
            leg_length_factor=1.1,
        )
        modifiers = MakeHumanMeshGenerator._convert_params_to_makehuman(params)
        assert modifiers["macrodetails-proportions/ShoulderWidth"] > 0
        assert modifiers["macrodetails-proportions/HipWidth"] < 0
        assert modifiers["macrodetails-proportions/ArmLength"] > 0
        assert modifiers["macrodetails-proportions/LegLength"] > 0


class TestMakeHumanAvailability:
    """Test is_available for MakeHuman."""

    def test_unavailable_when_not_installed(self):
        gen = MakeHumanMeshGenerator(makehuman_path="/nonexistent/makehuman")
        assert gen.is_available is False

    def test_available_when_path_exists(self, tmp_path: Path):
        mh_dir = tmp_path / "makehuman"
        mh_dir.mkdir()
        gen = MakeHumanMeshGenerator(makehuman_path=mh_dir)
        assert gen.is_available is True

    def test_returns_error_when_unavailable(self):
        gen = MakeHumanMeshGenerator(makehuman_path="/nonexistent")
        result = gen.generate(_default_params(), Path("/tmp/out"))
        assert result.success is False
        assert "not found" in result.error_message.lower()


class TestMakeHumanOBJParsing:
    """Test the OBJ file parser."""

    def test_parse_simple_obj(self, tmp_path: Path):
        obj_content = textwrap.dedent("""\
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            v 0.0 1.0 0.0
            v 1.0 1.0 0.0
            f 1 2 3
            f 2 3 4
        """)
        obj_file = tmp_path / "test.obj"
        obj_file.write_text(obj_content, encoding="utf-8")

        vertices, faces = MakeHumanMeshGenerator._parse_obj_file(obj_file)
        assert vertices.shape == (4, 3)
        assert faces.shape == (2, 3)
        # OBJ is 1-indexed, so first face should be [0, 1, 2]
        assert faces[0].tolist() == [0, 1, 2]

    def test_parse_obj_with_normals_and_texcoords(self, tmp_path: Path):
        obj_content = textwrap.dedent("""\
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            v 0.0 1.0 0.0
            vn 0.0 0.0 1.0
            vt 0.0 0.0
            f 1/1/1 2/1/1 3/1/1
        """)
        obj_file = tmp_path / "test.obj"
        obj_file.write_text(obj_content, encoding="utf-8")

        vertices, faces = MakeHumanMeshGenerator._parse_obj_file(obj_file)
        assert vertices.shape == (3, 3)
        assert faces.shape == (1, 3)

    def test_parse_obj_quad_triangulation(self, tmp_path: Path):
        obj_content = textwrap.dedent("""\
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            v 1.0 1.0 0.0
            v 0.0 1.0 0.0
            f 1 2 3 4
        """)
        obj_file = tmp_path / "test.obj"
        obj_file.write_text(obj_content, encoding="utf-8")

        vertices, faces = MakeHumanMeshGenerator._parse_obj_file(obj_file)
        assert vertices.shape == (4, 3)
        # A quad should be split into 2 triangles
        assert faces.shape == (2, 3)


class TestMakeHumanScriptGeneration:
    """Test MakeHuman scripted-mode script generation."""

    def test_script_contains_modifiers(self):
        modifiers = {
            "macrodetails/Gender": 1.0,
            "macrodetails/Age": 0.5,
        }
        script = MakeHumanMeshGenerator._build_mh_script(
            modifiers,
            Path("/tmp/body.obj"),
            Path("/tmp/groups.json"),
        )
        assert "macrodetails/Gender" in script
        assert "macrodetails/Age" in script
        assert "exportOBJ" in script
        assert "json.dump" in script


class TestMakeHumanGenerate:
    """Test the full MakeHuman generate pipeline with mocking."""

    @patch(
        "humanoid_character_builder.generators.mesh_generator.TRIMESH_AVAILABLE", True
    )
    @patch("humanoid_character_builder.generators.mesh_generator._trimesh_module")
    def test_generate_with_mocked_subprocess(self, mock_trimesh, tmp_path: Path):
        """Test end-to-end generation with mocked MakeHuman subprocess."""
        mh_dir = tmp_path / "makehuman"
        mh_dir.mkdir()
        (mh_dir / "makehuman.py").touch()

        gen = MakeHumanMeshGenerator(makehuman_path=mh_dir)

        # Set up mock trimesh
        exported_files: list[str] = []

        class FakeTrimesh:
            def __init__(self, vertices=None, faces=None):
                self.vertices = vertices
                self.faces = faces

            def export(self, path: str):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).touch()
                exported_files.append(path)

            @property
            def convex_hull(self):
                return self

        mock_trimesh.Trimesh = FakeTrimesh

        # Mock _run_makehuman_script to simulate success
        def mock_run(script_path: Path, timeout: int = 120) -> bool:
            # Create fake OBJ output
            script_dir = script_path.parent
            obj_path = script_dir / "body.obj"
            groups_path = script_dir / "groups.json"

            # Write a simple OBJ
            obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
            obj_path.write_text(obj_content, encoding="utf-8")

            # Write vertex groups
            groups = {"head": [0, 1, 2]}
            groups_path.write_text(json.dumps(groups), encoding="utf-8")
            return True

        with patch.object(gen, "_run_makehuman_script", side_effect=mock_run):
            result = gen.generate(_default_params(), tmp_path / "output")

        assert result.success is True
        assert result.metadata["backend"] == "makehuman"

    def test_generate_fails_when_script_fails(self, tmp_path: Path):
        """Test that generation fails gracefully when MakeHuman script fails."""
        mh_dir = tmp_path / "makehuman"
        mh_dir.mkdir()
        (mh_dir / "makehuman.py").touch()

        gen = MakeHumanMeshGenerator(makehuman_path=mh_dir)

        with patch.object(gen, "_run_makehuman_script", return_value=False):
            result = gen.generate(_default_params(), tmp_path / "output")

        assert result.success is False
        assert "failed" in result.error_message.lower()


class TestMakeHumanSupportedSegments:
    """Test get_supported_segments."""

    def test_returns_all_mapped_segments(self):
        gen = MakeHumanMeshGenerator()
        segments = gen.get_supported_segments()
        assert "head" in segments
        assert "left_hand" in segments
        assert "right_foot" in segments
        assert len(segments) == len(MakeHumanMeshGenerator.MH_VERTEX_GROUP_MAP)


class TestMakeHumanVertexGroupMap:
    """Test that the vertex group map is complete."""

    def test_all_values_unique(self):
        values = list(MakeHumanMeshGenerator.MH_VERTEX_GROUP_MAP.values())
        assert len(values) == len(set(values)), "Duplicate segment names in map"

    def test_all_keys_unique(self):
        keys = list(MakeHumanMeshGenerator.MH_VERTEX_GROUP_MAP.keys())
        assert len(keys) == len(set(keys)), "Duplicate MH group names in map"


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


class TestMeshGeneratorFactory:
    """Test the MeshGenerator factory class."""

    def test_create_smplx(self):
        gen = MeshGenerator.create(MeshGeneratorBackend.SMPLX)
        assert isinstance(gen, SMPLXMeshGenerator)
        assert gen.backend_name == "smplx"

    def test_create_makehuman(self):
        gen = MeshGenerator.create(MeshGeneratorBackend.MAKEHUMAN)
        assert isinstance(gen, MakeHumanMeshGenerator)
        assert gen.backend_name == "makehuman"

    def test_create_from_string(self):
        gen = MeshGenerator.create("smplx")
        assert isinstance(gen, SMPLXMeshGenerator)

    def test_create_from_string_case_insensitive(self):
        gen = MeshGenerator.create("SMPLX")
        assert isinstance(gen, SMPLXMeshGenerator)

    def test_create_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            MeshGenerator.create(MeshGeneratorBackend.CUSTOM)

    def test_interface_compliance(self):
        """Verify both generators implement the full interface."""
        for cls in [SMPLXMeshGenerator, MakeHumanMeshGenerator]:
            gen = cls()
            assert isinstance(gen, MeshGeneratorInterface)
            assert isinstance(gen.backend_name, str)
            assert isinstance(gen.is_available, bool)
            assert isinstance(gen.get_supported_segments(), list)


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------


class TestGeneratedMeshResult:
    """Test the result dataclass."""

    def test_successful_result(self):
        result = GeneratedMeshResult(
            success=True,
            mesh_paths={"head": Path("/tmp/head.stl")},
        )
        assert result.success is True
        assert result.error_message is None

    def test_failed_result(self):
        result = GeneratedMeshResult(
            success=False,
            error_message="Something went wrong",
        )
        assert result.success is False
        assert result.error_message == "Something went wrong"

    def test_defaults(self):
        result = GeneratedMeshResult(success=True)
        assert result.mesh_paths == {}
        assert result.collision_paths == {}
        assert result.texture_paths == {}
        assert result.vertex_groups == {}
        assert result.metadata == {}

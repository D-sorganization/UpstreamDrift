"""
Unit tests for public API module.
"""

import tempfile
from pathlib import Path

import pytest
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.interfaces.api import (
    CharacterBuilder,
    ExportOptions,
    quick_build,
    quick_urdf,
)
from humanoid_character_builder.presets.loader import (
    PRESET_NAMES,
    get_preset_info,
    list_available_presets,
    load_body_preset,
)


class TestCharacterBuilder:
    """Tests for CharacterBuilder class."""

    def test_init_default(self):
        builder = CharacterBuilder()
        assert builder.urdf_config is not None

    def test_build_default_params(self):
        builder = CharacterBuilder()
        params = BodyParameters()

        result = builder.build(params, generate_meshes=False)

        assert result.success
        assert result.urdf_xml is not None
        assert len(result.segments) > 0

    def test_build_custom_params(self):
        builder = CharacterBuilder()
        params = BodyParameters(
            height_m=1.85,
            mass_kg=85.0,
            muscularity=0.7,
        )

        result = builder.build(params, generate_meshes=False)

        assert result.success
        assert result.params.height_m == 1.85

    def test_generate_urdf(self):
        builder = CharacterBuilder()
        params = BodyParameters()

        _ = builder.generate_urdf(params)

        assert urdf is not None
        assert "<robot" in urdf

    def test_generate_urdf_to_file(self):
        builder = CharacterBuilder()
        params = BodyParameters()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.urdf"
            _ = builder.generate_urdf(params, output_path=output_path)

            assert output_path.exists()

    def test_compute_segment_inertia(self):
        builder = CharacterBuilder()

        inertia = builder.compute_segment_inertia("left_thigh", mass=10.0)

        assert inertia.mass == 10.0
        assert inertia.ixx > 0
        assert inertia.iyy > 0
        assert inertia.izz > 0

    def test_compute_segment_inertia_with_dimensions(self):
        builder = CharacterBuilder()

        inertia = builder.compute_segment_inertia(
            "left_thigh",
            mass=10.0,
            dimensions={"length": 0.4, "width": 0.1, "depth": 0.08},
        )

        assert inertia.mass == 10.0
        assert inertia.is_valid()

    def test_compute_all_inertias(self):
        builder = CharacterBuilder()
        params = BodyParameters()

        inertias = builder.compute_all_inertias(params)

        assert len(inertias) > 0
        assert "pelvis" in inertias
        assert "head" in inertias
        assert "left_thigh" in inertias

        # All should be valid
        for name, inertia in inertias.items():
            assert inertia.ixx > 0, f"{name} has invalid ixx"

    def test_create_from_preset(self):
        params = CharacterBuilder.create_from_preset("athletic")

        assert params.muscularity > 0.5
        assert params.body_fat_factor < 0.2

    def test_create_from_preset_with_overrides(self):
        params = CharacterBuilder.create_from_preset(
            "athletic", height_m=1.90, mass_kg=90.0
        )

        assert params.height_m == 1.90
        assert params.mass_kg == 90.0

    def test_list_presets(self):
        presets = CharacterBuilder.list_presets()

        assert len(presets) > 0
        assert "athletic" in presets
        assert "average" in presets

    def test_list_segments(self):
        segments = CharacterBuilder.list_segments()

        assert len(segments) > 0
        assert "pelvis" in segments
        assert "head" in segments
        assert "left_hand" in segments

    def test_get_segment_definition(self):
        definition = CharacterBuilder.get_segment_definition("pelvis")

        assert definition is not None
        assert definition["name"] == "pelvis"
        assert "mass_ratio" in definition


class TestCharacterBuildResult:
    """Tests for CharacterBuildResult class."""

    def test_export_urdf(self):
        builder = CharacterBuilder()
        params = BodyParameters()
        result = builder.build(params, generate_meshes=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = result.export_urdf(tmpdir)

            assert urdf_path.exists()
            assert (Path(tmpdir) / "config").exists()

    def test_export_urdf_custom_options(self):
        builder = CharacterBuilder()
        params = BodyParameters()
        result = builder.build(params, generate_meshes=False)

        options = ExportOptions(
            urdf_filename="custom.urdf",
            save_config=True,
            config_format="json",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = result.export_urdf(tmpdir, options=options)

            assert urdf_path.name == "custom.urdf"
            assert (Path(tmpdir) / "config" / "body_params.json").exists()

    def test_get_segment(self):
        builder = CharacterBuilder()
        params = BodyParameters()
        result = builder.build(params, generate_meshes=False)

        segment = result.get_segment("pelvis")

        assert segment is not None
        assert segment.segment_name == "pelvis"
        assert segment.mass_kg > 0

    def test_get_total_mass(self):
        builder = CharacterBuilder()
        params = BodyParameters(mass_kg=75.0)
        result = builder.build(params, generate_meshes=False)

        total_mass = result.get_total_mass()

        # Should be approximately the specified mass
        assert abs(total_mass - 75.0) < 5.0  # Allow some variance

    def test_to_dict(self):
        builder = CharacterBuilder()
        params = BodyParameters()
        result = builder.build(params, generate_meshes=False)

        data = result.to_dict()

        assert "success" in data
        assert "params" in data
        assert "segment_count" in data
        assert data["success"] is True


class TestQuickFunctions:
    """Tests for quick convenience functions."""

    def test_quick_build_default(self):
        result = quick_build()

        assert result.success
        assert result.params.height_m == 1.75
        assert result.params.mass_kg == 75.0

    def test_quick_build_custom(self):
        result = quick_build(height_m=1.85, mass_kg=85.0)

        assert result.success
        assert result.params.height_m == 1.85

    def test_quick_build_with_preset(self):
        result = quick_build(preset="athletic")

        assert result.success
        assert result.params.muscularity > 0.5

    def test_quick_build_with_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = quick_build(output_dir=tmpdir)

            assert result.success
            assert (Path(tmpdir) / "humanoid.urdf").exists()

    def test_quick_urdf_default(self):
        urdf = quick_urdf()

        assert urdf is not None
        assert "<robot" in urdf

    def test_quick_urdf_custom(self):
        urdf = quick_urdf(height_m=1.90)

        assert urdf is not None
        assert "<robot" in urdf

    def test_quick_urdf_with_preset(self):
        urdf = quick_urdf(preset="heavy")

        assert urdf is not None
        assert "<robot" in urdf


class TestPresets:
    """Tests for preset loading."""

    def test_list_available_presets(self):
        presets = list_available_presets()

        assert len(presets) > 0
        assert "athletic" in presets
        assert "average" in presets
        assert "heavy" in presets

    def test_load_body_preset_athletic(self):
        params = load_body_preset("athletic")

        assert params.muscularity > 0.5
        assert params.body_fat_factor < 0.2

    def test_load_body_preset_average(self):
        params = load_body_preset("average")

        assert params.muscularity == 0.5

    def test_load_body_preset_with_overrides(self):
        params = load_body_preset("athletic", height_m=2.0, mass_kg=100.0)

        assert params.height_m == 2.0
        assert params.mass_kg == 100.0
        # Should still have athletic properties
        assert params.muscularity > 0.5

    def test_load_body_preset_invalid(self):
        with pytest.raises(ValueError):
            load_body_preset("nonexistent_preset")

    def test_get_preset_info(self):
        info = get_preset_info("athletic")

        assert info["name"] == "athletic"
        assert "height_m" in info
        assert "mass_kg" in info
        assert "description" in info

    def test_preset_names_constant(self):
        assert len(PRESET_NAMES) > 0
        assert "athletic" in PRESET_NAMES


class TestExportOptions:
    """Tests for ExportOptions."""

    def test_default_values(self):
        options = ExportOptions()

        assert options.urdf_filename == "humanoid.urdf"
        assert options.include_collision is True
        assert options.generate_meshes is True
        assert options.mesh_format == "stl"

    def test_custom_values(self):
        options = ExportOptions(
            urdf_filename="robot.urdf",
            mesh_format="obj",
            save_config=False,
        )

        assert options.urdf_filename == "robot.urdf"
        assert options.mesh_format == "obj"
        assert options.save_config is False

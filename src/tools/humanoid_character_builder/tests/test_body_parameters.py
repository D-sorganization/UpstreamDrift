"""
Unit tests for body parameters module.
"""

from humanoid_character_builder.core.body_parameters import (
    RGBA,
    BodyParameters,
    BuildType,
    GenderModel,
    SegmentParameters,
    Vector3,
    create_athletic_body,
    create_average_body,
    create_heavy_body,
)


class TestVector3:
    """Tests for Vector3 class."""

    def test_default_values(self):
        v = Vector3()
        assert v.x == 1.0
        assert v.y == 1.0
        assert v.z == 1.0

    def test_as_tuple(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.as_tuple() == (1.0, 2.0, 3.0)

    def test_uniform(self):
        v = Vector3.uniform(0.5)
        assert v.x == 0.5
        assert v.y == 0.5
        assert v.z == 0.5

    def test_from_tuple(self):
        v = Vector3.from_tuple((1.0, 2.0, 3.0))
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0


class TestRGBA:
    """Tests for RGBA class."""

    def test_default_values(self):
        c = RGBA()
        assert c.r == 0.8
        assert c.a == 1.0

    def test_as_tuple(self):
        c = RGBA(1.0, 0.5, 0.25, 0.75)
        assert c.as_tuple() == (1.0, 0.5, 0.25, 0.75)

    def test_as_hex(self):
        c = RGBA(1.0, 0.5, 0.0, 1.0)
        assert c.as_hex() == "#ff7f00"


class TestBodyParameters:
    """Tests for BodyParameters class."""

    def test_default_values(self):
        params = BodyParameters()
        assert params.height_m == 1.75
        assert params.mass_kg == 75.0
        assert params.build_type == BuildType.AVERAGE

    def test_custom_values(self):
        params = BodyParameters(
            height_m=1.80,
            mass_kg=80.0,
            build_type=BuildType.MESOMORPH,
            muscularity=0.7,
        )
        assert params.height_m == 1.80
        assert params.mass_kg == 80.0
        assert params.muscularity == 0.7

    def test_gender_factor(self):
        male = BodyParameters(gender_model=GenderModel.MALE)
        female = BodyParameters(gender_model=GenderModel.FEMALE)
        neutral = BodyParameters(gender_model=GenderModel.NEUTRAL)

        assert male.get_effective_gender_factor() == 1.0
        assert female.get_effective_gender_factor() == 0.0
        assert neutral.get_effective_gender_factor() == 0.5

    def test_validate_valid_params(self):
        params = BodyParameters(height_m=1.75, mass_kg=75.0)
        errors = params.validate()
        assert len(errors) == 0

    def test_validate_invalid_height(self):
        params = BodyParameters(height_m=-1.0)
        errors = params.validate()
        assert any("height_m" in e for e in errors)

    def test_validate_invalid_mass(self):
        params = BodyParameters(mass_kg=-10.0)
        errors = params.validate()
        assert any("mass_kg" in e for e in errors)

    def test_segment_override(self):
        params = BodyParameters()
        seg_params = SegmentParameters(mass_kg=5.0)
        params.set_segment_override("left_thigh", seg_params)

        retrieved = params.get_segment_params("left_thigh")
        assert retrieved.mass_kg == 5.0
        assert retrieved.has_mass_override()

    def test_to_dict(self):
        params = BodyParameters(height_m=1.80, name="test_model")
        data = params.to_dict()

        assert data["height_m"] == 1.80
        assert data["name"] == "test_model"
        assert "build_type" in data

    def test_from_dict(self):
        data = {
            "height_m": 1.85,
            "mass_kg": 85.0,
            "build_type": "mesomorph",
            "muscularity": 0.8,
        }
        params = BodyParameters.from_dict(data)

        assert params.height_m == 1.85
        assert params.mass_kg == 85.0
        assert params.build_type == BuildType.MESOMORPH
        assert params.muscularity == 0.8


class TestFactoryFunctions:
    """Tests for convenience factory functions."""

    def test_create_athletic_body(self):
        params = create_athletic_body()
        assert params.build_type == BuildType.MESOMORPH
        assert params.muscularity > 0.5
        assert params.body_fat_factor < 0.2

    def test_create_athletic_body_with_overrides(self):
        params = create_athletic_body(height_m=1.90, mass_kg=90.0)
        assert params.height_m == 1.90
        assert params.mass_kg == 90.0

    def test_create_average_body(self):
        params = create_average_body()
        assert params.build_type == BuildType.AVERAGE
        assert params.muscularity == 0.5

    def test_create_heavy_body(self):
        params = create_heavy_body()
        assert params.build_type == BuildType.ENDOMORPH
        assert params.body_fat_factor > 0.3


class TestSegmentParameters:
    """Tests for SegmentParameters class."""

    def test_default_values(self):
        seg = SegmentParameters()
        assert seg.mass_kg is None
        assert seg.inertia_override is None

    def test_has_overrides(self):
        seg = SegmentParameters()
        assert not seg.has_mass_override()
        assert not seg.has_inertia_override()

        seg.mass_kg = 5.0
        assert seg.has_mass_override()

        seg.inertia_override = {"ixx": 0.1, "iyy": 0.1, "izz": 0.1}
        assert seg.has_inertia_override()

    def test_scale(self):
        seg = SegmentParameters(scale=Vector3(1.2, 1.0, 0.8))
        assert seg.scale.x == 1.2
        assert seg.scale.z == 0.8

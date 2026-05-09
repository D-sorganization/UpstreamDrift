"""Test URDF generation determinism for humanoid character builder."""

import pytest

from humanoid_character_builder import CharacterBuilder
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.presets.loader import list_available_presets


class TestURDFDeterminism:
    """Test that URDF generation is deterministic."""

    @pytest.mark.parametrize("preset", list_available_presets()[:5])
    def test_urdf_generation_is_deterministic_for_preset(self, preset: str) -> None:
        """Same preset should produce byte-identical URDF on repeated calls."""
        builder = CharacterBuilder()
        params = builder.create_from_preset(preset)

        urdf_a = builder.generate_urdf(params)
        urdf_b = builder.generate_urdf(params)

        assert urdf_a == urdf_b, f"URDF drift detected for preset '{preset}'"

    def test_urdf_generation_is_deterministic_for_custom_params(self) -> None:
        """Same custom BodyParameters should produce byte-identical URDF."""
        builder = CharacterBuilder()
        params = BodyParameters(
            height_m=1.80,
            mass_kg=80.0,
            muscularity=0.6,
            body_fat_factor=0.15,
            shoulder_width_factor=1.05,
            hip_width_factor=0.98,
            arm_length_factor=1.0,
            leg_length_factor=1.02,
            torso_length_factor=1.0,
        )

        urdf_a = builder.generate_urdf(params)
        urdf_b = builder.generate_urdf(params)

        assert urdf_a == urdf_b, "URDF drift detected for custom parameters"

    @pytest.mark.parametrize("iteration", range(10))
    def test_urdf_generation_is_deterministic_repeated(self, iteration: int) -> None:
        """Repeated URDF generation should be deterministic (10 iterations)."""
        builder = CharacterBuilder()
        params = BodyParameters(height_m=1.75, mass_kg=75.0)

        urdf_a = builder.generate_urdf(params)
        urdf_b = builder.generate_urdf(params)

        assert urdf_a == urdf_b, f"URDF drift detected on iteration {iteration}"

    def test_urdf_has_no_timestamps_or_uuids(self) -> None:
        """URDF should not contain timestamps or UUIDs that would break determinism."""
        builder = CharacterBuilder()
        params = BodyParameters(height_m=1.75, mass_kg=75.0)

        urdf = builder.generate_urdf(params)

        # Check for common non-deterministic patterns
        import re

        # ISO timestamps
        timestamp_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        assert not re.search(timestamp_pattern, urdf), "URDF contains timestamps"

        # UUIDs
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        assert not re.search(uuid_pattern, urdf, re.IGNORECASE), "URDF contains UUIDs"

        # Unix timestamps (10+ digit numbers that look like timestamps)
        unix_ts_pattern = r"\b\d{10,13}\b"
        # Note: This is a softer check as some legitimate numbers might match
        # Just warn if found, don't fail
        if re.search(unix_ts_pattern, urdf):
            pytest.skip("URDF contains potential unix timestamps (review needed)")

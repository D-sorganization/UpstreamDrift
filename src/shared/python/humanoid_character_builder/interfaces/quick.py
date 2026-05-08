"""Convenience functions for quick character building."""

from __future__ import annotations

from pathlib import Path

from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.interfaces.api import CharacterBuilder
from humanoid_character_builder.interfaces.results import CharacterBuildResult


def quick_build(
    height_m: float = 1.75,
    mass_kg: float = 75.0,
    preset: str | None = None,
    output_dir: Path | str | None = None,
) -> CharacterBuildResult:
    """
    Quick function to build a character with minimal configuration.

    Args:
        height_m: Character height in meters
        mass_kg: Character mass in kg
        preset: Optional preset name (overrides height/mass defaults)
        output_dir: Optional output directory for export

    Returns:
        CharacterBuildResult
    """
    if height_m is None:
        raise ValueError("height_m must be provided")
    builder = CharacterBuilder()

    if preset:
        params = builder.create_from_preset(preset, height_m=height_m, mass_kg=mass_kg)
    else:
        params = BodyParameters(height_m=height_m, mass_kg=mass_kg)

    result = builder.build(params)

    if output_dir and result.success:
        result.export_urdf(output_dir)

    return result


def quick_urdf(
    height_m: float = 1.75,
    mass_kg: float = 75.0,
    preset: str | None = None,
) -> str:
    """
    Quick function to generate URDF XML.

    Args:
        height_m: Character height in meters
        mass_kg: Character mass in kg
        preset: Optional preset name

    Returns:
        URDF XML string
    """
    if height_m is None:
        raise ValueError("height_m must be provided")
    builder = CharacterBuilder()

    if preset:
        params = builder.create_from_preset(preset, height_m=height_m, mass_kg=mass_kg)
    else:
        params = BodyParameters(height_m=height_m, mass_kg=mass_kg)

    return builder.generate_urdf(params)

"""
Preset loader for humanoid character builder.

Provides pre-configured body types and the ability to load
custom presets from YAML files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from humanoid_character_builder.core.body_parameters import (
    BodyParameters,
    BuildType,
    GenderModel,
)

logger = logging.getLogger(__name__)


# Built-in preset definitions
_BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "athletic": {
        "height_m": 1.80,
        "mass_kg": 80.0,
        "build_type": BuildType.MESOMORPH,
        "gender_model": GenderModel.NEUTRAL,
        "muscularity": 0.7,
        "body_fat_factor": 0.12,
        "shoulder_width_factor": 1.1,
        "hip_width_factor": 0.95,
        "arm_length_factor": 1.0,
        "leg_length_factor": 1.02,
        "description": "Athletic, muscular build with broad shoulders",
    },
    "average": {
        "height_m": 1.75,
        "mass_kg": 75.0,
        "build_type": BuildType.AVERAGE,
        "gender_model": GenderModel.NEUTRAL,
        "muscularity": 0.5,
        "body_fat_factor": 0.2,
        "shoulder_width_factor": 1.0,
        "hip_width_factor": 1.0,
        "arm_length_factor": 1.0,
        "leg_length_factor": 1.0,
        "description": "Average proportions",
    },
    "heavy": {
        "height_m": 1.78,
        "mass_kg": 100.0,
        "build_type": BuildType.ENDOMORPH,
        "gender_model": GenderModel.NEUTRAL,
        "muscularity": 0.4,
        "body_fat_factor": 0.35,
        "shoulder_width_factor": 1.05,
        "hip_width_factor": 1.15,
        "arm_length_factor": 0.98,
        "leg_length_factor": 0.98,
        "description": "Heavier build with higher body fat",
    },
    "lean": {
        "height_m": 1.82,
        "mass_kg": 70.0,
        "build_type": BuildType.ECTOMORPH,
        "gender_model": GenderModel.NEUTRAL,
        "muscularity": 0.4,
        "body_fat_factor": 0.1,
        "shoulder_width_factor": 0.95,
        "hip_width_factor": 0.9,
        "arm_length_factor": 1.05,
        "leg_length_factor": 1.05,
        "description": "Lean, tall build with long limbs",
    },
    "compact": {
        "height_m": 1.65,
        "mass_kg": 65.0,
        "build_type": BuildType.AVERAGE,
        "gender_model": GenderModel.NEUTRAL,
        "muscularity": 0.5,
        "body_fat_factor": 0.18,
        "shoulder_width_factor": 1.0,
        "hip_width_factor": 1.0,
        "arm_length_factor": 0.98,
        "leg_length_factor": 0.98,
        "description": "Shorter, compact build",
    },
    "tall": {
        "height_m": 1.95,
        "mass_kg": 90.0,
        "build_type": BuildType.AVERAGE,
        "gender_model": GenderModel.NEUTRAL,
        "muscularity": 0.5,
        "body_fat_factor": 0.18,
        "shoulder_width_factor": 1.0,
        "hip_width_factor": 0.95,
        "arm_length_factor": 1.02,
        "leg_length_factor": 1.03,
        "description": "Tall build",
    },
    "male_average": {
        "height_m": 1.78,
        "mass_kg": 80.0,
        "build_type": BuildType.AVERAGE,
        "gender_model": GenderModel.MALE,
        "muscularity": 0.5,
        "body_fat_factor": 0.2,
        "shoulder_width_factor": 1.05,
        "hip_width_factor": 0.95,
        "description": "Average male proportions",
    },
    "female_average": {
        "height_m": 1.65,
        "mass_kg": 62.0,
        "build_type": BuildType.AVERAGE,
        "gender_model": GenderModel.FEMALE,
        "muscularity": 0.4,
        "body_fat_factor": 0.25,
        "shoulder_width_factor": 0.95,
        "hip_width_factor": 1.05,
        "description": "Average female proportions",
    },
    "golfer_pro": {
        "height_m": 1.83,
        "mass_kg": 82.0,
        "build_type": BuildType.MESOMORPH,
        "gender_model": GenderModel.MALE,
        "muscularity": 0.6,
        "body_fat_factor": 0.15,
        "shoulder_width_factor": 1.08,
        "hip_width_factor": 0.98,
        "arm_length_factor": 1.0,
        "leg_length_factor": 1.0,
        "torso_length_factor": 1.0,
        "description": "Professional golfer body type",
    },
    "minimal": {
        "height_m": 1.70,
        "mass_kg": 60.0,
        "build_type": BuildType.ECTOMORPH,
        "gender_model": GenderModel.NEUTRAL,
        "muscularity": 0.3,
        "body_fat_factor": 0.1,
        "description": "Minimal/lightweight build for testing",
    },
}

# List of preset names
PRESET_NAMES = list(_BUILTIN_PRESETS.keys())


def load_body_preset(
    preset_name: str,
    height_m: float | None = None,
    mass_kg: float | None = None,
    **overrides: Any,
) -> BodyParameters:
    """
    Load body parameters from a preset.

    Args:
        preset_name: Name of the preset to load
        height_m: Override the preset's height
        mass_kg: Override the preset's mass
        **overrides: Additional parameter overrides

    Returns:
        BodyParameters configured for the preset

    Raises:
        ValueError: If preset name is not found
    """
    preset_name_lower = preset_name.lower()

    if preset_name_lower not in _BUILTIN_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {', '.join(PRESET_NAMES)}"
        )

    preset_data = _BUILTIN_PRESETS[preset_name_lower].copy()

    # Apply height/mass overrides
    if height_m is not None:
        preset_data["height_m"] = height_m
    if mass_kg is not None:
        preset_data["mass_kg"] = mass_kg

    # Apply additional overrides
    for key, value in overrides.items():
        if hasattr(BodyParameters, key):
            preset_data[key] = value

    # Remove description (not a BodyParameters field)
    preset_data.pop("description", None)

    # Set name
    preset_data["name"] = f"{preset_name_lower}_humanoid"

    return BodyParameters(**preset_data)


def load_segment_template(
    template_name: str,
) -> dict[str, Any]:
    """
    Load a segment configuration template.

    Args:
        template_name: Name of the template

    Returns:
        Dict with segment configuration
    """
    # For now, return default segment configuration
    # Can be extended to load from YAML files
    from humanoid_character_builder.core.segment_definitions import (
        HUMANOID_SEGMENTS,
    )

    segment = HUMANOID_SEGMENTS.get(template_name)
    if segment is None:
        raise ValueError(f"Unknown segment template: {template_name}")

    return {
        "name": segment.name,
        "parent": segment.parent,
        "mass_ratio": segment.mass_ratio,
        "length_ratio": segment.length_ratio,
        "geometry_type": segment.visual_geometry.geometry_type.value,
    }


def list_available_presets() -> list[str]:
    """Return list of available preset names."""
    return PRESET_NAMES.copy()


def get_preset_info(preset_name: str) -> dict[str, Any]:
    """
    Get information about a preset.

    Args:
        preset_name: Name of the preset

    Returns:
        Dict with preset information
    """
    preset_name_lower = preset_name.lower()

    if preset_name_lower not in _BUILTIN_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    preset_data = _BUILTIN_PRESETS[preset_name_lower]

    return {
        "name": preset_name_lower,
        "height_m": preset_data["height_m"],
        "mass_kg": preset_data["mass_kg"],
        "build_type": preset_data["build_type"].value,
        "description": preset_data.get("description", ""),
    }


def load_preset_from_file(file_path: Path | str) -> BodyParameters:
    """
    Load body parameters from a YAML or JSON file.

    Args:
        file_path: Path to configuration file

    Returns:
        BodyParameters loaded from file
    """
    import json

    import yaml  # type: ignore[import-untyped]

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Preset file not found: {file_path}")

    content = file_path.read_text()

    if file_path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(content)
    elif file_path.suffix.lower() == ".json":
        data = json.loads(content)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    return BodyParameters.from_dict(data)


def save_preset_to_file(
    params: BodyParameters,
    file_path: Path | str,
    format: str = "yaml",
) -> None:
    """
    Save body parameters to a file.

    Args:
        params: Body parameters to save
        file_path: Output file path
        format: Output format (yaml or json)
    """
    import json

    import yaml  # type: ignore[import-untyped]

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data = params.to_dict()

    if format.lower() == "yaml":
        content = yaml.dump(data, default_flow_style=False)
    else:
        content = json.dumps(data, indent=2)

    file_path.write_text(content)
    logger.info(f"Preset saved to {file_path}")

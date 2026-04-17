"""
Format detection and conversion utilities.

This module provides convenience functions for format conversion
and automatic format detection.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any


class ModelFormat(Enum):
    """Supported model formats."""

    URDF = "urdf"
    MJCF = "mjcf"
    SDF = "sdf"
    SIMSCAPE = "simscape"
    UNKNOWN = "unknown"


def detect_format(source: str | Path) -> ModelFormat:
    """
    Detect the format of a model file.

    Args:
        source: File path or XML string

    Returns:
        Detected ModelFormat
    """
    # Check if it's a file path
    if isinstance(source, Path) or (
        isinstance(source, str) and not source.strip().startswith("<")
    ):
        path = Path(source)
        suffix = path.suffix.lower()

        if suffix == ".urdf":
            return ModelFormat.URDF
        if suffix == ".xml":
            # Could be MJCF or URDF, need to check content
            content = path.read_text() if path.exists() else ""
            return _detect_format_from_content(content)
        if suffix == ".sdf":
            return ModelFormat.SDF
        if suffix in (".mdl", ".slx"):
            return ModelFormat.SIMSCAPE
        return ModelFormat.UNKNOWN

    # It's an XML string
    return _detect_format_from_content(source)


def _detect_format_from_content(content: str) -> ModelFormat:
    """Detect format from XML content."""
    content_lower = content.lower()

    if "<robot" in content_lower:
        return ModelFormat.URDF
    if "<mujoco" in content_lower:
        return ModelFormat.MJCF
    if "<sdf" in content_lower or "<world" in content_lower:
        return ModelFormat.SDF
    return ModelFormat.UNKNOWN


def convert_urdf_to_mjcf(
    source: str | Path,
    output_path: Path | None = None,
    **config_options: Any,
) -> str:
    """
    Convert URDF to MJCF format.

    Args:
        source: URDF file path or XML string
        output_path: Optional path to save output
        **config_options: MJCFConfig options

    Returns:
        MJCF XML string

    Example:
        mjcf = convert_urdf_to_mjcf("robot.urdf", output_path="robot.xml")
    """
    if source is None:
        raise ValueError("source must be provided")
    from model_generation.converters.mjcf_converter import MJCFConfig, MJCFConverter

    config = MJCFConfig(**config_options) if config_options else None
    converter = MJCFConverter(config)
    return converter.urdf_to_mjcf(source, output_path)


def convert_mjcf_to_urdf(
    source: str | Path,
    output_path: Path | None = None,
) -> str:
    """
    Convert MJCF to URDF format.

    Args:
        source: MJCF file path or XML string
        output_path: Optional path to save output

    Returns:
        URDF XML string

    Example:
        urdf = convert_mjcf_to_urdf("robot.xml", output_path="robot.urdf")
    """
    if source is None:
        raise ValueError("source must be provided")
    from model_generation.converters.mjcf_converter import MJCFConverter

    converter = MJCFConverter()
    return str(converter.mjcf_to_urdf(source, output_path))


def convert(
    source: str | Path,
    target_format: ModelFormat | str,
    output_path: Path | None = None,
) -> str:
    """Convert a robot model between supported formats.

    Source format is auto-detected from file extension or XML content.

    Supported conversions:
        - URDF -> MJCF (via ``MJCFConverter.urdf_to_mjcf``)
        - MJCF -> URDF (via ``MJCFConverter.mjcf_to_urdf``)

    Unsupported / not planned:
        - SDF -> URDF, URDF -> SDF, SDF -> MJCF, MJCF -> SDF
        - SIMSCAPE -> any (use ``convert_simscape_to_urdf`` for MDL import)

    Args:
        source: Source file path or XML string.
        target_format: Desired output format.  Accepts a ``ModelFormat``
            enum member or a lowercase string (``"urdf"``, ``"mjcf"``).
        output_path: If provided, the converted XML is also written to
            this path.

    Returns:
        The converted XML as a string.

    Raises:
        ValueError: If *source_format* equals *target_format* and the
            source cannot be read, or if the requested conversion pair
            is not supported.

    Example::

        result = convert("robot.urdf", ModelFormat.MJCF)
    """
    if isinstance(target_format, str):
        try:
            target_format = ModelFormat(target_format.lower())
        except ValueError:
            valid = [f.value for f in ModelFormat if f != ModelFormat.UNKNOWN]
            msg = (
                f"Unknown target format {target_format!r}. "
                f"Valid formats: {', '.join(valid)}"
            )
            raise ValueError(msg) from None

    # -- Preconditions (Design by Contract) --
    if target_format == ModelFormat.UNKNOWN:
        msg = "target_format must not be ModelFormat.UNKNOWN"
        raise ValueError(msg)

    source_format = detect_format(source)

    if source_format == ModelFormat.UNKNOWN:
        msg = (
            f"Could not detect the format of {source!r}. "
            "Provide a file with a recognised extension (.urdf, .xml, .sdf) "
            "or valid XML content."
        )
        raise ValueError(msg)

    if source_format == target_format:
        # No conversion needed
        if isinstance(source, Path) or not source.strip().startswith("<"):
            return Path(source).read_text()
        return source

    # URDF -> MJCF
    if source_format == ModelFormat.URDF and target_format == ModelFormat.MJCF:
        return convert_urdf_to_mjcf(source, output_path)

    # MJCF -> URDF
    if source_format == ModelFormat.MJCF and target_format == ModelFormat.URDF:
        return convert_mjcf_to_urdf(source, output_path)

    # Only URDF <-> MJCF conversion is supported.  Other pairs (SDF,
    # SIMSCAPE, etc.) are outside the current scope.  This is a user
    # error (unsupported input), not a missing abstract-method
    # implementation, so ValueError is the correct exception.
    supported_pairs = "URDF <-> MJCF"
    msg = (
        f"Conversion from {source_format.value!r} to "
        f"{target_format.value!r} is not supported. "
        f"Currently only {supported_pairs} conversions are available."
    )
    raise ValueError(msg)


def validate_urdf(source: str | Path) -> list[str]:
    """
    Validate a URDF file.

    Args:
        source: URDF file path or XML string

    Returns:
        List of error messages (empty if valid)
    """
    from model_generation.converters.urdf_parser import URDFParser
    from model_generation.core.validation import Validator

    try:
        parser = URDFParser()
        model = parser.parse(source)

        result = Validator.validate_model(model.links, model.joints)

        errors = result.get_error_messages()
        errors.extend(model.warnings)

        return list(errors)
    except (ValueError, KeyError, OSError) as e:
        return [str(e)]


def validate_mjcf(source: str | Path) -> list[str]:
    """
    Validate an MJCF file.

    Args:
        source: MJCF file path or XML string

    Returns:
        List of error messages (empty if valid)
    """
    try:
        import mujoco

        if isinstance(source, Path) or not source.strip().startswith("<"):
            mujoco.MjModel.from_xml_path(str(source))
        else:
            mujoco.MjModel.from_xml_string(source)
        return []
    except ImportError:
        # MuJoCo not available, do basic XML validation
        import xml.etree.ElementTree as StdET

        import defusedxml.ElementTree as DefusedET

        try:
            if isinstance(source, Path) or not source.strip().startswith("<"):
                content = Path(source).read_text()
            else:
                content = source
            DefusedET.fromstring(content)
            return []
        except StdET.ParseError as e:
            return [f"XML parse error: {e}"]
    except (PermissionError, OSError) as e:
        return [str(e)]

"""Tests for converters.format_utils: format detection and convert dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest
from model_generation.converters.format_utils import (
    ModelFormat,
    _detect_format_from_content,
    convert,
    detect_format,
    validate_mjcf,
    validate_urdf,
)


def test_detect_format_from_content_urdf() -> None:
    assert _detect_format_from_content("<robot name='r'></robot>") == ModelFormat.URDF


def test_detect_format_from_content_mjcf() -> None:
    assert _detect_format_from_content("<mujoco/>") == ModelFormat.MJCF


def test_detect_format_from_content_sdf() -> None:
    assert _detect_format_from_content("<sdf>...</sdf>") == ModelFormat.SDF


def test_detect_format_from_content_world_is_sdf() -> None:
    assert _detect_format_from_content("<world/>") == ModelFormat.SDF


def test_detect_format_from_content_unknown() -> None:
    assert _detect_format_from_content("<other/>") == ModelFormat.UNKNOWN


def test_detect_format_from_xml_string() -> None:
    assert detect_format("<robot></robot>") == ModelFormat.URDF
    assert detect_format("<mujoco/>") == ModelFormat.MJCF


def test_detect_format_from_path_urdf(tmp_path: Path) -> None:
    f = tmp_path / "robot.urdf"
    f.write_text("<robot/>")
    assert detect_format(f) == ModelFormat.URDF


def test_detect_format_from_path_sdf(tmp_path: Path) -> None:
    f = tmp_path / "robot.sdf"
    f.write_text("<sdf/>")
    assert detect_format(f) == ModelFormat.SDF


def test_detect_format_from_path_xml_content(tmp_path: Path) -> None:
    f = tmp_path / "robot.xml"
    f.write_text("<mujoco/>")
    assert detect_format(f) == ModelFormat.MJCF


def test_detect_format_from_path_mdl(tmp_path: Path) -> None:
    f = tmp_path / "model.mdl"
    f.write_text("")
    assert detect_format(f) == ModelFormat.SIMSCAPE


def test_detect_format_unknown_extension(tmp_path: Path) -> None:
    f = tmp_path / "weird.foo"
    f.write_text("")
    assert detect_format(f) == ModelFormat.UNKNOWN


def test_detect_format_string_path_not_starting_with_angle(tmp_path: Path) -> None:
    f = tmp_path / "r.urdf"
    f.write_text("<robot/>")
    assert detect_format(str(f)) == ModelFormat.URDF


def test_convert_invalid_target_string() -> None:
    with pytest.raises(ValueError, match="Unknown target format"):
        convert("<robot/>", "potato")


def test_convert_target_unknown_enum() -> None:
    with pytest.raises(ValueError, match="must not be ModelFormat.UNKNOWN"):
        convert("<robot/>", ModelFormat.UNKNOWN)


def test_convert_undetectable_source() -> None:
    with pytest.raises(ValueError, match="Could not detect"):
        convert("<weird/>", ModelFormat.URDF)


def test_convert_same_format_xml_string_passthrough() -> None:
    src = "<robot/>"
    out = convert(src, ModelFormat.URDF)
    assert out == src


def test_convert_same_format_from_file(tmp_path: Path) -> None:
    f = tmp_path / "r.urdf"
    f.write_text("<robot/>")
    out = convert(f, "urdf")
    assert "<robot" in out


def test_convert_unsupported_pair_raises() -> None:
    # SDF -> URDF is unsupported
    with pytest.raises(ValueError, match="is not supported"):
        convert("<sdf/>", ModelFormat.URDF)


def test_validate_urdf_invalid_content_returns_errors() -> None:
    # Garbage XML — parser will fail. validate_urdf returns error list.
    errs = validate_urdf("<not-a-robot/>")
    assert isinstance(errs, list)
    assert errs  # at least one error


def test_validate_mjcf_no_mujoco_basic_xml_ok() -> None:
    # Either uses mujoco (succeeds) or falls back to defusedxml parse (succeeds).
    errs = validate_mjcf("<mujoco><worldbody/></mujoco>")
    assert isinstance(errs, list)


def test_validate_mjcf_malformed_xml() -> None:
    errs = validate_mjcf("<mujoco><unclosed>")
    assert isinstance(errs, list)
    assert errs

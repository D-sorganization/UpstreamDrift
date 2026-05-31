"""Contracts for the JaxSim canonical URDF-to-SDF conversion gate."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from src.shared.python.simulation_backends.jaxsim_urdf_sdf_gate import (
    SdformatTool,
    assert_inertials_round_trip,
    build_jaxsim_model_from_sdf,
    compare_inertials,
    convert_urdf_to_sdf,
    find_sdformat_tool,
    read_sdf_inertials,
    read_urdf_inertials,
)
from src.shared.python.simulation_backends import jaxsim_urdf_sdf_gate as gate


_URDF = """\
<robot name="sample">
  <link name="base">
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="1.0" ixy="0.1" ixz="0.2" iyy="1.1" iyz="0.3" izz="1.2"/>
    </inertial>
  </link>
</robot>
"""

_SDF = """\
<sdf version="1.10">
  <model name="sample">
    <link name="base">
      <inertial>
        <mass>2.5</mass>
        <inertia>
          <ixx>1.0</ixx>
          <ixy>0.1</ixy>
          <ixz>0.2</ixz>
          <iyy>1.1</iyy>
          <iyz>0.3</iyz>
          <izz>1.2</izz>
        </inertia>
      </inertial>
    </link>
  </model>
</sdf>
"""


def test_inertial_readers_compare_matching_urdf_and_sdf(tmp_path: Path) -> None:
    urdf = tmp_path / "sample.urdf"
    sdf = tmp_path / "sample.sdf"
    urdf.write_text(_URDF, encoding="utf-8")
    sdf.write_text(_SDF, encoding="utf-8")

    expected = read_urdf_inertials(urdf)
    actual = read_sdf_inertials(sdf)

    assert expected["base"].mass == pytest.approx(2.5)
    assert actual["base"].inertia["izz"] == pytest.approx(1.2)
    assert compare_inertials(expected, actual) == []
    assert_inertials_round_trip(urdf, sdf)


def test_compare_inertials_reports_field_level_mismatches(tmp_path: Path) -> None:
    urdf = tmp_path / "sample.urdf"
    sdf = tmp_path / "sample.sdf"
    urdf.write_text(_URDF, encoding="utf-8")
    sdf.write_text(_SDF.replace("<iyy>1.1</iyy>", "<iyy>9.9</iyy>"), encoding="utf-8")

    mismatches = compare_inertials(read_urdf_inertials(urdf), read_sdf_inertials(sdf))

    assert [(m.link, m.field) for m in mismatches] == [("base", "iyy")]


def test_compare_inertials_reports_unexpected_sdf_links(tmp_path: Path) -> None:
    urdf = tmp_path / "sample.urdf"
    sdf = tmp_path / "sample.sdf"
    extra_link = """\
    <link name="sdformat_added_link">
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.1</iyy>
          <iyz>0.0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>
    </link>
"""
    urdf.write_text(_URDF, encoding="utf-8")
    sdf.write_text(
        _SDF.replace("  </model>", extra_link + "  </model>"), encoding="utf-8"
    )

    mismatches = compare_inertials(read_urdf_inertials(urdf), read_sdf_inertials(sdf))

    assert [(m.link, m.field) for m in mismatches] == [
        ("sdformat_added_link", "<unexpected link>")
    ]


def test_find_sdformat_tool_prefers_gz_command_shape(tmp_path: Path) -> None:
    tool_path = tmp_path / ("gz.exe" if sys.platform == "win32" else "gz")
    tool_path.write_text("", encoding="utf-8")
    tool_path.chmod(tool_path.stat().st_mode | stat.S_IEXEC)

    tool = find_sdformat_tool(search_path=str(tmp_path))

    assert tool is not None
    assert tool.executable.name.lower() == tool_path.name.lower()
    assert tool.mode == "gz"
    assert tool.command_for(Path("model.urdf")) == [
        str(tool.executable),
        "sdf",
        "-p",
        "model.urdf",
    ]


def test_convert_urdf_to_sdf_writes_stdout_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    urdf = tmp_path / "sample.urdf"
    out = tmp_path / "sample.sdf"
    urdf.write_text(_URDF, encoding="utf-8")

    def fake_run(*args: object, **kwargs: object) -> object:
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": _SDF, "stderr": ""},
        )()

    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    converted = convert_urdf_to_sdf(
        urdf,
        out,
        tool=SdformatTool(executable=Path("gz"), mode="gz"),
        timeout_s=10,
    )

    assert converted == out
    assert out.read_text(encoding="utf-8").startswith("<sdf")


@pytest.mark.requires_jaxsim
@pytest.mark.requires_sdformat
def test_canonical_urdf_converts_to_sdf_and_loads_in_jaxsim(tmp_path: Path) -> None:
    """#6648 acceptance: canonical URDF survives sdformat conversion."""
    pytest.importorskip("jaxsim")
    if find_sdformat_tool() is None:
        pytest.skip("sdformat CLI not installed; install gz-tools/libsdformat")

    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
        EXPECTED_NUM_VELOCITIES,
        build_humanoid_urdf,
    )

    urdf = build_humanoid_urdf(out_path=tmp_path / "golfer.urdf")
    sdf = convert_urdf_to_sdf(urdf, tmp_path / "golfer.sdf")

    assert_inertials_round_trip(urdf, sdf, abs_tolerance=1e-9, rel_tolerance=1e-9)
    model = build_jaxsim_model_from_sdf(sdf)
    assert model.dofs() == EXPECTED_NUM_VELOCITIES

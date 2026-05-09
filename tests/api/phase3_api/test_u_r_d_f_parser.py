"""Tests for Phase 3 API: URDF/MJCF rendering, analysis tools, simulation controls.

Validates Pydantic contract models and route logic for:
- URDF model parsing and serving (#1201)
- Analysis metrics, statistics, and export (#1203)
- Body positioning, measurement tools (#1179)

See issue #1201, #1203, #1179
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.api.models.requests import (
    BodyPositionUpdateRequest,
    DataExportRequest,
    MeasurementRequest,
)
from src.api.models.responses import (
    AnalysisMetricsSummary,
    AnalysisStatisticsResponse,
    BodyPositionResponse,
    JointAngleDisplay,
    MeasurementResult,
    MeasurementToolsResponse,
    ModelListResponse,
    URDFJointDescriptor,
    URDFLinkGeometry,
    URDFModelResponse,
)

# ──────────────────────────────────────────────────────────────
#  Contract Tests: URDF Model Responses (#1201)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Analysis Tools (#1203)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Body Positioning & Measurements (#1179)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  URDF Parser Tests (#1201)
# ──────────────────────────────────────────────────────────────


class TestURDFParser:
    """Test the URDF XML parser in the models route."""

    def test_parse_simple_humanoid(self) -> None:
        """Parse the simple_humanoid.urdf from test fixtures."""
        from src.api.routes.models import _parse_urdf

        urdf = """<?xml version="1.0"?>
        <robot name="test_robot">
          <material name="blue">
            <color rgba="0 0 0.8 1"/>
          </material>
          <link name="base">
            <visual>
              <geometry>
                <box size="0.2 0.3 0.4"/>
              </geometry>
              <material name="blue"/>
              <origin xyz="0 0 0.2" rpy="0 0 0"/>
            </visual>
          </link>
          <link name="arm">
            <visual>
              <geometry>
                <cylinder radius="0.05" length="0.3"/>
              </geometry>
              <origin xyz="0.15 0 0" rpy="0 1.57 0"/>
            </visual>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base"/>
            <child link="arm"/>
            <origin xyz="0.1 0 0.4" rpy="0 0 0"/>
            <axis xyz="0 1 0"/>
            <limit lower="-3.14" upper="3.14" effort="30" velocity="1"/>
          </joint>
        </robot>
        """
        result = _parse_urdf(urdf)

        assert result.model_name == "test_robot"
        assert len(result.links) == 2
        assert len(result.joints) == 1
        assert result.root_link == "base"

        # Check link parsing
        base_link = next(lnk for lnk in result.links if lnk.link_name == "base")
        assert base_link.geometry_type == "box"
        assert base_link.dimensions["width"] == 0.2
        assert base_link.color == [0.0, 0.0, 0.8, 1.0]  # From material

        arm_link = next(lnk for lnk in result.links if lnk.link_name == "arm")
        assert arm_link.geometry_type == "cylinder"
        assert arm_link.dimensions["radius"] == 0.05

        # Check joint parsing
        shoulder = result.joints[0]
        assert shoulder.name == "shoulder"
        assert shoulder.joint_type == "revolute"
        assert shoulder.parent_link == "base"
        assert shoulder.child_link == "arm"
        assert shoulder.axis == [0.0, 1.0, 0.0]
        assert shoulder.lower_limit == -3.14

    def test_parse_sphere_geometry(self) -> None:
        """Parse sphere geometry."""
        from src.api.routes.models import _parse_urdf

        urdf = """<robot name="sphere_test">
          <link name="ball">
            <visual>
              <geometry>
                <sphere radius="0.1"/>
              </geometry>
            </visual>
          </link>
        </robot>"""
        result = _parse_urdf(urdf)
        assert result.links[0].geometry_type == "sphere"
        assert result.links[0].dimensions["radius"] == 0.1

    def test_parse_mesh_geometry(self) -> None:
        """Parse mesh geometry with scale."""
        from src.api.routes.models import _parse_urdf

        urdf = """<robot name="mesh_test">
          <link name="body">
            <visual>
              <geometry>
                <mesh filename="body.stl" scale="0.001 0.001 0.001"/>
              </geometry>
            </visual>
          </link>
        </robot>"""
        result = _parse_urdf(urdf)
        assert result.links[0].geometry_type == "mesh"
        assert result.links[0].mesh_path == "body.stl"
        assert result.links[0].dimensions["scale_x"] == 0.001

    def test_parse_fixed_joint(self) -> None:
        """Parse fixed joint type."""
        from src.api.routes.models import _parse_urdf

        urdf = """<robot name="fixed_test">
          <link name="world"/>
          <link name="base">
            <visual>
              <geometry><box size="1 1 1"/></geometry>
            </visual>
          </link>
          <joint name="world_joint" type="fixed">
            <parent link="world"/>
            <child link="base"/>
          </joint>
        </robot>"""
        result = _parse_urdf(urdf)
        assert result.joints[0].joint_type == "fixed"

    def test_parse_invalid_xml(self) -> None:
        """Invalid XML raises ValueError."""
        from src.api.routes.models import _parse_urdf

        with pytest.raises(ValueError, match="Invalid URDF XML"):
            _parse_urdf("not valid xml <><><>")

    def test_parse_inline_material_color(self) -> None:
        """Parse material color defined inline in the visual."""
        from src.api.routes.models import _parse_urdf

        urdf = """<robot name="color_test">
          <link name="colored">
            <visual>
              <geometry><sphere radius="0.1"/></geometry>
              <material name="red">
                <color rgba="1 0 0 1"/>
              </material>
            </visual>
          </link>
        </robot>"""
        result = _parse_urdf(urdf)
        assert result.links[0].color == [1.0, 0.0, 0.0, 1.0]

    def test_parse_multiple_joints_kinematic_chain(self) -> None:
        """Parse a chain of multiple joints."""
        from src.api.routes.models import _parse_urdf

        urdf = """<robot name="chain_test">
          <link name="a"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
          <link name="b"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
          <link name="c"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
          <joint name="j1" type="revolute">
            <parent link="a"/><child link="b"/>
            <axis xyz="0 0 1"/>
            <limit lower="-1" upper="1" effort="10" velocity="1"/>
          </joint>
          <joint name="j2" type="revolute">
            <parent link="b"/><child link="c"/>
            <axis xyz="0 1 0"/>
            <limit lower="-1" upper="1" effort="10" velocity="1"/>
          </joint>
        </robot>"""
        result = _parse_urdf(urdf)
        assert result.root_link == "a"
        assert len(result.joints) == 2
        assert result.joints[0].parent_link == "a"
        assert result.joints[1].parent_link == "b"


# ──────────────────────────────────────────────────────────────
#  Model Discovery Tests (#1201)
# ──────────────────────────────────────────────────────────────

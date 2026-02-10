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


class TestURDFLinkGeometryContract:
    """Validate URDFLinkGeometry response model."""

    def test_box_geometry(self) -> None:
        """Box geometry with all fields."""
        link = URDFLinkGeometry(
            link_name="torso",
            geometry_type="box",
            dimensions={"width": 0.2, "height": 0.4, "depth": 0.6},
            origin=[0.0, 0.0, 0.3],
            rotation=[0.0, 0.0, 0.0],
            color=[0.0, 0.0, 0.8, 1.0],
        )
        assert link.link_name == "torso"
        assert link.geometry_type == "box"
        assert link.dimensions["width"] == 0.2

    def test_cylinder_geometry(self) -> None:
        """Cylinder geometry."""
        link = URDFLinkGeometry(
            link_name="arm",
            geometry_type="cylinder",
            dimensions={"radius": 0.05, "length": 0.3},
        )
        assert link.geometry_type == "cylinder"
        assert link.dimensions["radius"] == 0.05

    def test_sphere_geometry(self) -> None:
        """Sphere geometry."""
        link = URDFLinkGeometry(
            link_name="head",
            geometry_type="sphere",
            dimensions={"radius": 0.12},
        )
        assert link.geometry_type == "sphere"

    def test_mesh_geometry(self) -> None:
        """Mesh geometry with path."""
        link = URDFLinkGeometry(
            link_name="hand",
            geometry_type="mesh",
            dimensions={"scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0},
            mesh_path="meshes/hand.stl",
        )
        assert link.geometry_type == "mesh"
        assert link.mesh_path == "meshes/hand.stl"

    def test_defaults(self) -> None:
        """Defaults are applied when not specified."""
        link = URDFLinkGeometry(
            link_name="test",
            geometry_type="box",
        )
        assert link.origin == [0.0, 0.0, 0.0]
        assert link.rotation == [0.0, 0.0, 0.0]
        assert link.color == [0.5, 0.5, 0.5, 1.0]
        assert link.mesh_path is None


class TestURDFJointDescriptorContract:
    """Validate URDFJointDescriptor response model."""

    def test_revolute_joint(self) -> None:
        """Revolute joint with limits."""
        joint = URDFJointDescriptor(
            name="shoulder",
            joint_type="revolute",
            parent_link="torso",
            child_link="upper_arm",
            origin=[0.1, 0.0, 0.5],
            rotation=[0.0, 0.0, 0.0],
            axis=[0.0, 1.0, 0.0],
            lower_limit=-3.14,
            upper_limit=3.14,
        )
        assert joint.name == "shoulder"
        assert joint.joint_type == "revolute"
        assert joint.parent_link == "torso"
        assert joint.child_link == "upper_arm"
        assert joint.lower_limit == -3.14

    def test_fixed_joint(self) -> None:
        """Fixed joint (no limits needed)."""
        joint = URDFJointDescriptor(
            name="base_fixed",
            joint_type="fixed",
            parent_link="world",
            child_link="base",
        )
        assert joint.joint_type == "fixed"
        assert joint.lower_limit is None

    def test_defaults(self) -> None:
        """Defaults are applied."""
        joint = URDFJointDescriptor(
            name="test",
            joint_type="revolute",
            parent_link="a",
            child_link="b",
        )
        assert joint.origin == [0.0, 0.0, 0.0]
        assert joint.axis == [0.0, 0.0, 1.0]


class TestURDFModelResponseContract:
    """Validate URDFModelResponse contract."""

    def test_full_model(self) -> None:
        """Complete model with links and joints."""
        model = URDFModelResponse(
            model_name="simple_humanoid",
            links=[
                URDFLinkGeometry(
                    link_name="torso",
                    geometry_type="box",
                    dimensions={"width": 0.2, "height": 0.4, "depth": 0.6},
                ),
                URDFLinkGeometry(
                    link_name="head",
                    geometry_type="sphere",
                    dimensions={"radius": 0.12},
                ),
            ],
            joints=[
                URDFJointDescriptor(
                    name="neck",
                    joint_type="revolute",
                    parent_link="torso",
                    child_link="head",
                    origin=[0.0, 0.0, 0.6],
                ),
            ],
            root_link="torso",
        )
        assert model.model_name == "simple_humanoid"
        assert len(model.links) == 2
        assert len(model.joints) == 1
        assert model.root_link == "torso"

    def test_empty_model(self) -> None:
        """Model with no links or joints."""
        model = URDFModelResponse(
            model_name="empty",
            links=[],
            joints=[],
            root_link="base",
        )
        assert model.model_name == "empty"

    def test_with_raw_urdf(self) -> None:
        """Model includes raw URDF XML."""
        model = URDFModelResponse(
            model_name="test",
            links=[],
            joints=[],
            root_link="base",
            urdf_raw="<robot name='test'></robot>",
        )
        assert model.urdf_raw is not None


class TestModelListResponseContract:
    """Validate ModelListResponse contract."""

    def test_list_with_models(self) -> None:
        """List with multiple model entries."""
        resp = ModelListResponse(
            models=[
                {"name": "humanoid", "format": "urdf", "path": "models/humanoid.urdf"},
                {"name": "golfer", "format": "urdf", "path": "models/golfer.urdf"},
            ]
        )
        assert len(resp.models) == 2

    def test_empty_list(self) -> None:
        """Empty model list."""
        resp = ModelListResponse(models=[])
        assert len(resp.models) == 0


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Analysis Tools (#1203)
# ──────────────────────────────────────────────────────────────


class TestDataExportRequestContract:
    """Validate DataExportRequest preconditions."""

    def test_csv_format(self) -> None:
        """CSV format is valid."""
        req = DataExportRequest(format="csv")
        assert req.format == "csv"

    def test_json_format(self) -> None:
        """JSON format is valid."""
        req = DataExportRequest(format="json")
        assert req.format == "json"

    def test_invalid_format_rejected(self) -> None:
        """Invalid format raises ValidationError."""
        with pytest.raises(ValidationError):
            DataExportRequest(format="hdf5")

    def test_case_insensitive(self) -> None:
        """Format is normalized to lowercase."""
        req = DataExportRequest(format="CSV")
        assert req.format == "csv"

    def test_time_range(self) -> None:
        """Optional time range is accepted."""
        req = DataExportRequest(format="csv", time_range=[0.0, 1.5])
        assert req.time_range == [0.0, 1.5]

    def test_defaults(self) -> None:
        """Default values are applied."""
        req = DataExportRequest(format="csv")
        assert req.include_metrics is True
        assert req.include_time_series is True
        assert req.time_range is None


class TestAnalysisMetricsSummaryContract:
    """Validate AnalysisMetricsSummary response model."""

    def test_full_summary(self) -> None:
        """Complete metric summary."""
        summary = AnalysisMetricsSummary(
            metric_name="club_head_speed",
            current=45.2,
            minimum=0.0,
            maximum=52.1,
            mean=30.5,
            std_dev=12.3,
        )
        assert summary.metric_name == "club_head_speed"
        assert summary.current == 45.2

    def test_default_std_dev(self) -> None:
        """std_dev defaults to 0."""
        summary = AnalysisMetricsSummary(
            metric_name="test",
            current=1.0,
            minimum=0.0,
            maximum=2.0,
            mean=1.0,
        )
        assert summary.std_dev == 0.0


class TestAnalysisStatisticsResponseContract:
    """Validate AnalysisStatisticsResponse contract."""

    def test_full_response(self) -> None:
        """Response with metrics and time series."""
        resp = AnalysisStatisticsResponse(
            sim_time=2.5,
            sample_count=100,
            metrics=[
                AnalysisMetricsSummary(
                    metric_name="ke",
                    current=10.0,
                    minimum=0.0,
                    maximum=15.0,
                    mean=8.0,
                    std_dev=3.0,
                ),
            ],
            time_series={"ke": [1.0, 2.0, 5.0, 10.0]},
        )
        assert resp.sim_time == 2.5
        assert resp.sample_count == 100
        assert len(resp.metrics) == 1

    def test_no_time_series(self) -> None:
        """Response without time series."""
        resp = AnalysisStatisticsResponse(
            sim_time=0.0,
            sample_count=0,
            metrics=[],
        )
        assert resp.time_series is None


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Body Positioning & Measurements (#1179)
# ──────────────────────────────────────────────────────────────


class TestBodyPositionUpdateRequestContract:
    """Validate BodyPositionUpdateRequest preconditions."""

    def test_valid_position(self) -> None:
        """Position with 3 elements is valid."""
        req = BodyPositionUpdateRequest(
            body_name="torso",
            position=[1.0, 2.0, 3.0],
        )
        assert req.body_name == "torso"
        assert req.position == [1.0, 2.0, 3.0]

    def test_valid_rotation(self) -> None:
        """Rotation with 3 elements is valid."""
        req = BodyPositionUpdateRequest(
            body_name="head",
            rotation=[0.1, 0.2, 0.3],
        )
        assert req.rotation == [0.1, 0.2, 0.3]

    def test_invalid_position_length(self) -> None:
        """Position with wrong length raises error."""
        with pytest.raises(ValidationError):
            BodyPositionUpdateRequest(
                body_name="torso",
                position=[1.0, 2.0],  # Only 2 elements
            )

    def test_invalid_rotation_length(self) -> None:
        """Rotation with wrong length raises error."""
        with pytest.raises(ValidationError):
            BodyPositionUpdateRequest(
                body_name="torso",
                rotation=[1.0],  # Only 1 element
            )

    def test_both_none(self) -> None:
        """Both position and rotation can be None."""
        req = BodyPositionUpdateRequest(body_name="arm")
        assert req.position is None
        assert req.rotation is None


class TestMeasurementRequestContract:
    """Validate MeasurementRequest model."""

    def test_valid_request(self) -> None:
        """Valid measurement request."""
        req = MeasurementRequest(body_a="torso", body_b="head")
        assert req.body_a == "torso"
        assert req.body_b == "head"


class TestBodyPositionResponseContract:
    """Validate BodyPositionResponse model."""

    def test_full_response(self) -> None:
        """Complete position response."""
        resp = BodyPositionResponse(
            body_name="torso",
            position=[1.0, 2.0, 3.0],
            rotation=[0.0, 0.0, 0.0],
            status="Position set for torso",
        )
        assert resp.body_name == "torso"


class TestMeasurementResultContract:
    """Validate MeasurementResult model."""

    def test_full_measurement(self) -> None:
        """Complete measurement result."""
        result = MeasurementResult(
            body_a="torso",
            body_b="head",
            distance=0.6,
            position_a=[0.0, 0.0, 0.0],
            position_b=[0.0, 0.0, 0.6],
            delta=[0.0, 0.0, 0.6],
        )
        assert result.distance == 0.6
        assert result.delta == [0.0, 0.0, 0.6]


class TestJointAngleDisplayContract:
    """Validate JointAngleDisplay model."""

    def test_full_display(self) -> None:
        """Complete joint angle display."""
        display = JointAngleDisplay(
            joint_name="shoulder",
            angle_rad=1.57,
            angle_deg=90.0,
            velocity=0.5,
            torque=10.0,
        )
        assert display.joint_name == "shoulder"
        assert display.angle_deg == 90.0

    def test_defaults(self) -> None:
        """Defaults for velocity and torque."""
        display = JointAngleDisplay(
            joint_name="elbow",
            angle_rad=0.0,
            angle_deg=0.0,
        )
        assert display.velocity == 0.0
        assert display.torque == 0.0


class TestMeasurementToolsResponseContract:
    """Validate MeasurementToolsResponse model."""

    def test_with_data(self) -> None:
        """Response with joint angles and measurements."""
        resp = MeasurementToolsResponse(
            joint_angles=[
                JointAngleDisplay(
                    joint_name="hip",
                    angle_rad=0.5,
                    angle_deg=28.6,
                ),
            ],
            measurements=[
                MeasurementResult(
                    body_a="a",
                    body_b="b",
                    distance=1.0,
                    position_a=[0, 0, 0],
                    position_b=[1, 0, 0],
                    delta=[1, 0, 0],
                ),
            ],
        )
        assert len(resp.joint_angles) == 1
        assert len(resp.measurements) == 1

    def test_empty(self) -> None:
        """Empty response."""
        resp = MeasurementToolsResponse(
            joint_angles=[],
        )
        assert len(resp.measurements) == 0


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


class TestModelDiscovery:
    """Test model file discovery."""

    def test_discover_models_returns_list(self) -> None:
        """Model discovery returns a list of dicts."""
        from src.api.routes.models import _discover_models

        models = _discover_models()
        assert isinstance(models, list)

        # Should find at least some URDF files in the project
        if models:
            assert "name" in models[0]
            assert "format" in models[0]
            assert "path" in models[0]

    def test_discover_models_finds_urdf(self) -> None:
        """Model discovery finds URDF files."""
        from src.api.routes.models import _discover_models

        models = _discover_models()
        urdf_models = [m for m in models if m["format"] == "urdf"]
        # There are URDF files in the test fixtures
        assert len(urdf_models) > 0

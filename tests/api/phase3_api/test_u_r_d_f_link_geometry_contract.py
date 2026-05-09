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

    def test_phase3_api_defaults(self) -> None:
        """Defaults are applied when not specified."""
        link = URDFLinkGeometry(
            link_name="test",
            geometry_type="box",
        )
        assert link.origin == [0.0, 0.0, 0.0]
        assert link.rotation == [0.0, 0.0, 0.0]
        assert link.color == [0.5, 0.5, 0.5, 1.0]
        assert link.mesh_path is None


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Analysis Tools (#1203)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Body Positioning & Measurements (#1179)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  URDF Parser Tests (#1201)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Model Discovery Tests (#1201)
# ──────────────────────────────────────────────────────────────

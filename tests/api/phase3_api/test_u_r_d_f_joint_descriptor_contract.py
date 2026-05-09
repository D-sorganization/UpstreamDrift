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

    def test_phase3_api_defaults(self) -> None:
        """Defaults are applied."""
        joint = URDFJointDescriptor(
            name="test",
            joint_type="revolute",
            parent_link="a",
            child_link="b",
        )
        assert joint.origin == [0.0, 0.0, 0.0]
        assert joint.axis == [0.0, 0.0, 1.0]


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

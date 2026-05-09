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

    def test_phase3_api_defaults(self) -> None:
        """Defaults for velocity and torque."""
        display = JointAngleDisplay(
            joint_name="elbow",
            angle_rad=0.0,
            angle_deg=0.0,
        )
        assert display.velocity == 0.0
        assert display.torque == 0.0


# ──────────────────────────────────────────────────────────────
#  URDF Parser Tests (#1201)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Model Discovery Tests (#1201)
# ──────────────────────────────────────────────────────────────

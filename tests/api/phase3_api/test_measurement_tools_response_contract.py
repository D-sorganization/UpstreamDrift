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


# ──────────────────────────────────────────────────────────────
#  Model Discovery Tests (#1201)
# ──────────────────────────────────────────────────────────────

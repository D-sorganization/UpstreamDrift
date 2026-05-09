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


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Body Positioning & Measurements (#1179)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  URDF Parser Tests (#1201)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Model Discovery Tests (#1201)
# ──────────────────────────────────────────────────────────────

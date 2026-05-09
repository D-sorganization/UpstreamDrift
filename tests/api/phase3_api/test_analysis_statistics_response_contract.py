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


# ──────────────────────────────────────────────────────────────
#  URDF Parser Tests (#1201)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Model Discovery Tests (#1201)
# ──────────────────────────────────────────────────────────────

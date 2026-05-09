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

    def test_phase3_api_case_insensitive(self) -> None:
        """Format is normalized to lowercase."""
        req = DataExportRequest(format="CSV")
        assert req.format == "csv"

    def test_time_range(self) -> None:
        """Optional time range is accepted."""
        req = DataExportRequest(format="csv", time_range=[0.0, 1.5])
        assert req.time_range == [0.0, 1.5]

    def test_phase3_api_defaults(self) -> None:
        """Default values are applied."""
        req = DataExportRequest(format="csv")
        assert req.include_metrics is True
        assert req.include_time_series is True
        assert req.time_range is None


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Body Positioning & Measurements (#1179)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  URDF Parser Tests (#1201)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Model Discovery Tests (#1201)
# ──────────────────────────────────────────────────────────────

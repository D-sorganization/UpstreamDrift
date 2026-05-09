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


# ──────────────────────────────────────────────────────────────
#  URDF Parser Tests (#1201)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Model Discovery Tests (#1201)
# ──────────────────────────────────────────────────────────────

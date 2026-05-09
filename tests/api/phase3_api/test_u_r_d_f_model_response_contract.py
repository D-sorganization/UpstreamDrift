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

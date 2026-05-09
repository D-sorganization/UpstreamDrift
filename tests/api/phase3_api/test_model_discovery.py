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


# ──────────────────────────────────────────────────────────────
#  URDF Parser Tests (#1201)
# ──────────────────────────────────────────────────────────────


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

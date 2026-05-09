"""Tests for Phase 4 API: Force overlays, actuator controls, model explorer, AIP.

Validates Pydantic contract models and route logic for:
- Force/torque vector overlays (#1199)
- Per-actuator control sliders (#1198)
- Model explorer & URDF editor (#1200)
- AIP JSON-RPC server (#763)

See issue #1199, #1198, #1200, #763
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.api.models.requests import (
    ActuatorBatchCommandRequest,
    ActuatorCommandRequest,
    AIPJsonRpcRequest,
    ForceOverlayRequest,
    ModelCompareRequest,
    ModelExplorerRequest,
)
from src.api.models.responses import (
    ActuatorCommandResponse,
    ActuatorInfo,
    ActuatorPanelResponse,
    AIPCapability,
    AIPHandshakeResponse,
    AIPJsonRpcResponse,
    ForceOverlayResponse,
    ForceVector3D,
    ModelCompareResponse,
    ModelExplorerResponse,
    URDFTreeNode,
)

# ──────────────────────────────────────────────────────────────
#  Contract Tests: Force Overlay (#1199)
# ──────────────────────────────────────────────────────────────


class TestForceOverlayRequestContract:
    """Validate ForceOverlayRequest model."""

    def test_phase4_api_default_values(self) -> None:
        """Defaults should be sensible."""
        req = ForceOverlayRequest()
        assert req.enabled is True
        assert req.force_types == ["applied"]
        assert req.scale_factor == 0.01
        assert req.color_by_magnitude is True
        assert req.body_filter is None
        assert req.show_labels is False

    def test_valid_force_types(self) -> None:
        """All valid force types accepted."""
        req = ForceOverlayRequest(
            force_types=["applied", "gravity", "contact", "bias", "all"]
        )
        assert len(req.force_types) == 5

    def test_invalid_force_type_rejected(self) -> None:
        """Unknown force type should fail validation."""
        with pytest.raises(ValidationError, match="Unknown force type"):
            ForceOverlayRequest(force_types=["invalid_type"])

    def test_scale_factor_range(self) -> None:
        """Scale factor must be positive and <= 1.0."""
        with pytest.raises(ValidationError):
            ForceOverlayRequest(scale_factor=0.0)
        with pytest.raises(ValidationError):
            ForceOverlayRequest(scale_factor=2.0)

    def test_body_filter(self) -> None:
        """Body filter should accept list of names."""
        req = ForceOverlayRequest(body_filter=["torso", "hand"])
        assert req.body_filter == ["torso", "hand"]


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Actuator Controls (#1198)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Model Explorer (#1200)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Contract Tests: AIP JSON-RPC (#763)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Dispatcher (#763)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Methods (#763)
# ──────────────────────────────────────────────────────────────

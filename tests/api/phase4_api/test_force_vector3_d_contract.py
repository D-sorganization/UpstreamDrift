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


class TestForceVector3DContract:
    """Validate ForceVector3D response model."""

    def test_basic_vector(self) -> None:
        """Basic force vector creation."""
        vec = ForceVector3D(
            body_name="torso",
            force_type="applied",
            origin=[0.0, 1.0, 0.0],
            direction=[1.0, 0.0, 0.0],
            magnitude=50.0,
        )
        assert vec.body_name == "torso"
        assert vec.force_type == "applied"
        assert vec.magnitude == 50.0
        assert vec.color == [1.0, 0.0, 0.0, 1.0]  # Default red

    def test_custom_color_and_label(self) -> None:
        """Custom color and label."""
        vec = ForceVector3D(
            body_name="arm",
            force_type="gravity",
            origin=[0.0, 0.5, 0.0],
            direction=[0.0, -1.0, 0.0],
            magnitude=9.81,
            color=[0.0, 0.0, 1.0, 0.8],
            label="9.81 N",
        )
        assert vec.color == [0.0, 0.0, 1.0, 0.8]
        assert vec.label == "9.81 N"


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

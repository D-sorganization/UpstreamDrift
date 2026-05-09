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


class TestForceOverlayResponseContract:
    """Validate ForceOverlayResponse model."""

    def test_empty_response(self) -> None:
        """Response with no vectors."""
        resp = ForceOverlayResponse(
            sim_time=0.0,
            vectors=[],
            total_force_magnitude=0.0,
            total_torque_magnitude=0.0,
        )
        assert resp.sim_time == 0.0
        assert len(resp.vectors) == 0

    def test_response_with_vectors(self) -> None:
        """Response with multiple vectors."""
        vectors = [
            ForceVector3D(
                body_name="torso",
                force_type="applied",
                origin=[0.0, 1.0, 0.0],
                direction=[1.0, 0.0, 0.0],
                magnitude=50.0,
            ),
            ForceVector3D(
                body_name="arm",
                force_type="gravity",
                origin=[0.0, 0.5, 0.0],
                direction=[0.0, -1.0, 0.0],
                magnitude=9.81,
            ),
        ]
        resp = ForceOverlayResponse(
            sim_time=1.5,
            vectors=vectors,
            total_force_magnitude=59.81,
            total_torque_magnitude=50.0,
        )
        assert len(resp.vectors) == 2
        assert resp.total_force_magnitude == pytest.approx(59.81)


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

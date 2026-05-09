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


class TestAIPMethods:
    """Test the AIP method implementations."""

    def test_create_registry(self) -> None:
        """Registry should contain all expected methods."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        methods = registry.list_methods()

        # Verify expected methods exist
        assert "simulation.start" in methods
        assert "simulation.stop" in methods
        assert "simulation.step" in methods
        assert "simulation.status" in methods
        assert "simulation.set_control" in methods
        assert "model.load" in methods
        assert "model.query" in methods
        assert "model.list" in methods
        assert "analysis.metrics" in methods
        assert "analysis.export" in methods
        assert "analysis.time_series" in methods
        assert "system.capabilities" in methods
        assert "system.ping" in methods

    def test_system_ping(self) -> None:
        """Ping returns pong."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("system.ping")
        assert handler is not None
        result = handler()
        assert result["status"] == "pong"

    def test_system_capabilities(self) -> None:
        """Capabilities returns structured data."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("system.capabilities")
        assert handler is not None
        result = handler()
        assert "server_name" in result
        assert "capabilities" in result
        assert "supported_methods" in result
        assert len(result["supported_methods"]) > 0

    def test_simulation_start(self) -> None:
        """Start returns status."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("simulation.start")
        assert handler is not None
        result = handler(engine_type="pendulum", duration=1.0)
        assert result["status"] == "started"
        assert result["engine_type"] == "pendulum"

    def test_simulation_stop(self) -> None:
        """Stop returns stopped status."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("simulation.stop")
        assert handler is not None
        result = handler()
        assert result["status"] == "stopped"

    def test_simulation_status_no_engine(self) -> None:
        """Status with no engine returns not running."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("simulation.status")
        assert handler is not None
        result = handler()
        assert result["running"] is False

    def test_model_load_requires_path(self) -> None:
        """Model load with empty path returns error."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("model.load")
        assert handler is not None
        result = handler(path="")
        assert result["status"] == "error"

    def test_model_load_valid_path(self) -> None:
        """Model load with valid path returns loaded."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("model.load")
        assert handler is not None
        result = handler(path="test.urdf")
        assert result["status"] == "loaded"
        assert result["format"] == "urdf"

    def test_analysis_export(self) -> None:
        """Analysis export returns status."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("analysis.export")
        assert handler is not None
        result = handler(format="csv")
        assert result["status"] == "ok"
        assert result["format"] == "csv"

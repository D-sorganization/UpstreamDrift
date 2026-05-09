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


class TestAIPDispatcher:
    """Test the JSON-RPC dispatcher logic."""

    def test_method_registry(self) -> None:
        """Registry stores and retrieves methods."""
        from src.api.aip.dispatcher import MethodRegistry

        registry = MethodRegistry()
        registry.register("test.hello", lambda: {"msg": "hello"}, "Say hello")

        assert "test.hello" in registry.list_methods()
        assert registry.get_method("test.hello") is not None
        assert registry.get_method("nonexistent") is None
        assert registry.get_description("test.hello") == "Say hello"

    def test_list_by_namespace(self) -> None:
        """Methods grouped by namespace."""
        from src.api.aip.dispatcher import MethodRegistry

        registry = MethodRegistry()
        registry.register("sim.start", lambda: None)
        registry.register("sim.stop", lambda: None)
        registry.register("model.load", lambda: None)

        namespaces = registry.list_by_namespace()
        assert "sim" in namespaces
        assert len(namespaces["sim"]) == 2
        assert "model" in namespaces
        assert len(namespaces["model"]) == 1

    def test_dispatch_success(self) -> None:
        """Dispatch resolves method and returns result."""
        import asyncio

        from src.api.aip.dispatcher import MethodRegistry, dispatch

        registry = MethodRegistry()
        registry.register("test.add", lambda a, b, **kw: a + b)

        result = asyncio.run(
            dispatch(
                registry,
                {
                    "jsonrpc": "2.0",
                    "method": "test.add",
                    "params": [3, 4],
                    "id": 1,
                },
            )
        )

        assert result is not None
        assert result["result"] == 7
        assert result["id"] == 1

    def test_dispatch_method_not_found(self) -> None:
        """Unknown method returns -32601."""
        import asyncio

        from src.api.aip.dispatcher import METHOD_NOT_FOUND, MethodRegistry, dispatch

        registry = MethodRegistry()

        result = asyncio.run(
            dispatch(
                registry,
                {
                    "jsonrpc": "2.0",
                    "method": "nonexistent",
                    "id": 1,
                },
            )
        )

        assert result is not None
        assert result["error"]["code"] == METHOD_NOT_FOUND

    def test_dispatch_invalid_version(self) -> None:
        """Wrong JSON-RPC version returns error."""
        import asyncio

        from src.api.aip.dispatcher import INVALID_REQUEST, MethodRegistry, dispatch

        registry = MethodRegistry()

        result = asyncio.run(
            dispatch(
                registry,
                {
                    "jsonrpc": "1.0",
                    "method": "test",
                    "id": 1,
                },
            )
        )

        assert result is not None
        assert result["error"]["code"] == INVALID_REQUEST

    def test_dispatch_notification(self) -> None:
        """Notification (no id) returns None."""
        import asyncio

        from src.api.aip.dispatcher import MethodRegistry, dispatch

        registry = MethodRegistry()
        registry.register("test.noop", lambda **kw: None)

        result = asyncio.run(
            dispatch(
                registry,
                {
                    "jsonrpc": "2.0",
                    "method": "test.noop",
                },
            )
        )

        assert result is None

    def test_dispatch_with_kwargs(self) -> None:
        """Dispatch with named parameters."""
        import asyncio

        from src.api.aip.dispatcher import MethodRegistry, dispatch

        registry = MethodRegistry()
        registry.register(
            "test.greet",
            lambda name="world", **kw: f"hello {name}",
        )

        result = asyncio.run(
            dispatch(
                registry,
                {
                    "jsonrpc": "2.0",
                    "method": "test.greet",
                    "params": {"name": "alice"},
                    "id": 42,
                },
            )
        )

        assert result is not None
        assert result["result"] == "hello alice"
        assert result["id"] == 42


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Methods (#763)
# ──────────────────────────────────────────────────────────────

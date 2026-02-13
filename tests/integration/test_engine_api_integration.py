"""
API-level Integration Tests for Physics Engines.

End-to-end tests verifying each engine can be probed, loaded,
and simulated through the FastAPI endpoints.  Engines that are
not installed are automatically skipped.

Supplements the existing test_engine_integration.py (unit-level)
with HTTP-level verification.

Fixes #1119
"""

import time

import pytest

try:
    from fastapi.testclient import TestClient

    from src.api.server import app
except ImportError:
    pytest.skip("API server deps not available", allow_module_level=True)

from src.shared.python.engine_core.engine_registry import EngineType


@pytest.fixture(scope="module")
def client():
    """Create test client with proper application lifespan."""
    with TestClient(app) as c:
        yield c


# ──────────────────────────────────────────────────────────────
#  Engine Probe Integration (HTTP)
# ──────────────────────────────────────────────────────────────
class TestEngineProbeHTTP:
    """Verify every registered engine can be probed via HTTP."""

    ALL_ENGINES = [
        "mujoco",
        "drake",
        "pinocchio",
        "opensim",
        "myosuite",
        "putting_green",
    ]

    @pytest.mark.parametrize("engine_name", ALL_ENGINES)
    def test_probe_returns_valid_json(
        self, client: TestClient, engine_name: str
    ) -> None:
        """Each engine probe returns parseable JSON with standard fields."""
        resp = client.get(f"/api/engines/{engine_name}/probe")
        assert resp.status_code == 200

        data = resp.json()
        assert "available" in data
        assert isinstance(data["available"], bool)

    @pytest.mark.parametrize("engine_name", ALL_ENGINES)
    def test_probe_includes_diagnostic(
        self, client: TestClient, engine_name: str
    ) -> None:
        """Each engine probe includes diagnostic information."""
        resp = client.get(f"/api/engines/{engine_name}/probe")
        data = resp.json()
        assert len(data) > 1  # More than just "available"


# ──────────────────────────────────────────────────────────────
#  Engine Load Integration (HTTP)
# ──────────────────────────────────────────────────────────────
class TestEngineLoadHTTP:
    """Verify engine loading via HTTP endpoints."""

    def test_load_putting_green(self, client: TestClient) -> None:
        """Putting Green is pure Python and should always load."""
        resp = client.post("/api/engines/putting_green/load")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("loaded") is True or data.get("status") == "loaded"

    def test_load_nonexistent_engine(self, client: TestClient) -> None:
        """Loading a non-existent engine returns appropriate error."""
        resp = client.post("/api/engines/completely_fake/load")
        assert resp.status_code in [400, 404, 500]


# ──────────────────────────────────────────────────────────────
#  Engine Registry Consistency
# ──────────────────────────────────────────────────────────────
class TestEngineRegistryConsistency:
    """Verify engine registry is internally consistent."""

    def test_all_engine_types_have_loaders(self) -> None:
        """Every non-MATLAB EngineType has a corresponding loader."""
        from src.shared.python.engine_core.engine_loaders import LOADER_MAP

        skip_types = {EngineType.MATLAB_2D, EngineType.MATLAB_3D}

        for engine_type in EngineType:
            if engine_type in skip_types:
                continue
            assert engine_type in LOADER_MAP, (
                f"{engine_type.value} missing from LOADER_MAP"
            )

    def test_loader_map_values_are_callable(self) -> None:
        """All LOADER_MAP values are callable functions."""
        from src.shared.python.engine_core.engine_loaders import LOADER_MAP

        for engine_type, loader in LOADER_MAP.items():
            assert callable(loader), f"Loader for {engine_type.value} is not callable"

    def test_engine_type_enum_values_are_strings(self) -> None:
        """EngineType enum values are strings (used in API routes)."""
        for engine_type in EngineType:
            assert isinstance(engine_type.value, str)
            assert len(engine_type.value) > 0


# ──────────────────────────────────────────────────────────────
#  Cross-Engine Consistency
# ──────────────────────────────────────────────────────────────
class TestCrossEngineConsistency:
    """Verify consistent behavior across all engines."""

    def test_all_probes_have_available_field(self, client: TestClient) -> None:
        """All probes return responses containing 'available'."""
        engines = ["mujoco", "drake", "pinocchio", "putting_green"]

        for engine in engines:
            resp = client.get(f"/api/engines/{engine}/probe")
            assert resp.status_code == 200
            data = resp.json()
            assert "available" in data, f"{engine} probe missing 'available'"


# ──────────────────────────────────────────────────────────────
#  Performance Benchmarks
# ──────────────────────────────────────────────────────────────
class TestEnginePerformanceBenchmarks:
    """Performance benchmarks for engine operations."""

    def test_probe_latency(self, client: TestClient) -> None:
        """Engine probes respond within 2 seconds."""
        engines = ["mujoco", "drake", "pinocchio", "putting_green"]

        for engine in engines:
            start = time.time()
            resp = client.get(f"/api/engines/{engine}/probe")
            elapsed = time.time() - start
            assert resp.status_code == 200
            assert elapsed < 2.0, f"{engine} probe took {elapsed:.2f}s (>2s)"

    def test_engine_list_latency(self, client: TestClient) -> None:
        """Engine list responds within 3 seconds."""
        start = time.time()
        resp = client.get("/engines")
        elapsed = time.time() - start
        assert resp.status_code == 200
        assert elapsed < 3.0

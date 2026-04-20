"""TDD tests for API route prefix consistency (issue #2451).

Five route modules hardcode the "/api/" segment in their APIRouter prefix.
When the server also registers these routers under "/api/v1/", the versioned
path becomes "/api/v1/api/<resource>" — double "/api/".

The fix: remove "/api/" from each router prefix so:
  - root registration (prefix="") yields /launcher/... etc.
  - versioned registration (prefix="/api/v1") yields /api/v1/launcher/... (no double)

The server must also change the root-level registration to use prefix="/api"
so that legacy "/api/launcher/..." clients still work.
"""

from __future__ import annotations

import pytest


class TestRouterPrefixesNoHardcodedApiSegment:
    """Route module APIRouter instances must NOT include '/api' in their prefix.

    The server is responsible for injecting the '/api' or '/api/v1' segment.
    Modules that embed '/api' force double-prefixing when registered under the
    versioned path.
    """

    def test_launcher_router_prefix_no_api(self) -> None:
        """launcher.router prefix must not start with /api."""
        from src.api.routes.launcher import router

        assert not router.prefix.startswith("/api"), (
            f"launcher router prefix starts with /api: {router.prefix!r}. "
            "Remove /api — the server injects this."
        )

    def test_terrain_router_prefix_no_api(self) -> None:
        """terrain.router prefix must not start with /api."""
        from src.api.routes.terrain import router

        assert not router.prefix.startswith("/api"), (
            f"terrain router prefix starts with /api: {router.prefix!r}."
        )

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("multipart"),
        reason="python-multipart not installed; data_explorer imports UploadFile",
    )
    def test_data_explorer_router_prefix_no_api(self) -> None:
        """data_explorer.router prefix must not start with /api."""
        from src.api.routes.data_explorer import router

        assert not router.prefix.startswith("/api"), (
            f"data_explorer router prefix starts with /api: {router.prefix!r}."
        )

    def test_motion_capture_router_prefix_no_api(self) -> None:
        """motion_capture.router prefix must not start with /api."""
        from src.api.routes.motion_capture import router

        assert not router.prefix.startswith("/api"), (
            f"motion_capture router prefix starts with /api: {router.prefix!r}."
        )

    def test_putting_green_router_prefix_no_api(self) -> None:
        """putting_green.router prefix must not start with /api."""
        from src.api.routes.putting_green import router

        assert not router.prefix.startswith("/api"), (
            f"putting_green router prefix starts with /api: {router.prefix!r}."
        )


class TestVersionedRoutePathsNoDoubleApi:
    """When routes are registered under /api/v1/, the resulting path must not contain /api/v1/api/."""

    def test_launcher_prefix_under_versioned_no_double(self) -> None:
        """launcher router's prefix joined with /api/v1 must not produce /api/v1/api/."""
        from src.api.routes.launcher import router

        simulated_path = "/api/v1" + router.prefix
        assert "/api/v1/api" not in simulated_path, (
            f"Joining launcher prefix with /api/v1 gives double-api: {simulated_path!r}"
        )

    def test_terrain_prefix_under_versioned_no_double(self) -> None:
        """terrain router's prefix joined with /api/v1 must not produce /api/v1/api/."""
        from src.api.routes.terrain import router

        simulated_path = "/api/v1" + router.prefix
        assert "/api/v1/api" not in simulated_path, (
            f"Joining terrain prefix with /api/v1 gives double-api: {simulated_path!r}"
        )

    def test_data_explorer_prefix_under_versioned_no_double(self) -> None:
        """data_explorer router prefix literal must not start with /api (source-level check)."""
        import ast
        from pathlib import Path

        source = Path("src/api/routes/data_explorer.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(getattr(node.func, "attr", None), str)
                and node.func.attr == "APIRouter"  # type: ignore[union-attr]
            ):
                for kw in node.keywords:
                    if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                        prefix_val: str = kw.value.value
                        assert not prefix_val.startswith("/api"), (
                            f"data_explorer APIRouter prefix starts with /api: {prefix_val!r}"
                        )

    def test_motion_capture_prefix_under_versioned_no_double(self) -> None:
        """motion_capture router's prefix joined with /api/v1 must not produce /api/v1/api/."""
        from src.api.routes.motion_capture import router

        simulated_path = "/api/v1" + router.prefix
        assert "/api/v1/api" not in simulated_path, (
            f"Joining motion_capture prefix with /api/v1 gives double-api: {simulated_path!r}"
        )

    def test_putting_green_prefix_under_versioned_no_double(self) -> None:
        """putting_green router's prefix joined with /api/v1 must not produce /api/v1/api/."""
        from src.api.routes.putting_green import router

        simulated_path = "/api/v1" + router.prefix
        assert "/api/v1/api" not in simulated_path, (
            f"Joining putting_green prefix with /api/v1 gives double-api: {simulated_path!r}"
        )

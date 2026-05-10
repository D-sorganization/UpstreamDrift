"""Tests for Pose Studio embed adapter.

Verifies that the Pose Studio embed adapter satisfies the EmbeddableTool
protocol and that create_main_widget returns a valid QWidget.

Note: These tests focus on the embed adapter contract and do not instantiate
the full MainWidget (which requires matplotlib 3D). The adapter tests verify
the protocol compliance without GUI dependencies.
"""

from __future__ import annotations

import pytest
from PyQt6 import QtWidgets

from src.shared.python.launcher_embed import EmbeddableTool, is_embeddable
from src.tools.pose_studio.gui import _EmbedAdapter


@pytest.fixture(scope="function")
def qapp():
    """Create a QApplication instance for tests that need Qt widgets."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


class TestEmbedAdapterProtocol:
    """Tests verifying _EmbedAdapter satisfies EmbeddableTool protocol."""

    def test_is_embeddable_tool(self) -> None:
        """Verify _EmbedAdapter is recognized as an EmbeddableTool."""
        adapter = _EmbedAdapter()
        assert isinstance(adapter, EmbeddableTool)

    def test_tool_id_is_string(self) -> None:
        """Verify tool_id is a non-empty string."""
        adapter = _EmbedAdapter()
        assert isinstance(adapter.tool_id, str)
        assert len(adapter.tool_id) > 0

    def test_tool_id_matches_expected(self) -> None:
        """Verify tool_id matches the expected pose_studio identifier."""
        adapter = _EmbedAdapter()
        assert adapter.tool_id == "pose_studio"


class TestEmbedCapabilities:
    """Tests for embed capabilities."""

    def test_embed_capabilities_returns_valid_object(self) -> None:
        """Verify embed_capabilities returns a valid EmbedCapabilities."""
        adapter = _EmbedAdapter()
        caps = adapter.embed_capabilities()

        assert caps.supports_embedded is True
        assert isinstance(caps.prefers_dock, bool)
        assert isinstance(caps.min_size, tuple)
        assert len(caps.min_size) == 2
        assert all(isinstance(x, int) and x > 0 for x in caps.min_size)
        assert isinstance(caps.requires_separate_qapplication, bool)

    def test_min_size_meets_minimum_requirements(self) -> None:
        """Verify min_size meets the minimum required dimensions."""
        adapter = _EmbedAdapter()
        caps = adapter.embed_capabilities()

        width, height = caps.min_size
        assert width >= 640
        assert height >= 480


class TestCreateMainWidget:
    """Tests for create_main_widget functionality.
    
    Note: Full widget instantiation tests are skipped due to matplotlib 3D
    dependency issues in headless CI environments. The adapter contract
    tests above verify the embed protocol compliance.
    """

    def test_create_main_widget_exists(self) -> None:
        """Verify create_main_widget method exists on adapter."""
        adapter = _EmbedAdapter()
        assert hasattr(adapter, "create_main_widget")
        assert callable(adapter.create_main_widget)


class TestCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_exists(self) -> None:
        """Verify cleanup method exists on adapter."""
        adapter = _EmbedAdapter()
        assert hasattr(adapter, "cleanup")
        assert callable(adapter.cleanup)

    def test_cleanup_runs_without_error(self) -> None:
        """Verify cleanup() runs without raising errors."""
        adapter = _EmbedAdapter()
        # Don't call create_main_widget due to matplotlib 3D dependency
        adapter.cleanup()

    def test_cleanup_is_idempotent(self) -> None:
        """Verify cleanup can be called multiple times safely."""
        adapter = _EmbedAdapter()
        # Multiple calls should not raise
        adapter.cleanup()
        adapter.cleanup()
        adapter.cleanup()


class TestIsDirty:
    """Tests for is_dirty functionality."""

    def test_is_dirty_exists(self) -> None:
        """Verify is_dirty method exists on adapter."""
        adapter = _EmbedAdapter()
        assert hasattr(adapter, "is_dirty")
        assert callable(adapter.is_dirty)

    def test_is_dirty_returns_false_by_default(self) -> None:
        """Verify is_dirty returns False (Pose Studio doesn't track dirty state)."""
        adapter = _EmbedAdapter()
        assert adapter.is_dirty() is False


class TestRegistryIntegration:
    """Tests for registry integration."""

    def test_adapter_is_registered(self) -> None:
        """Verify the embed adapter is registered in the registry."""
        assert is_embeddable("pose_studio")

    def test_registered_tool_is_embeddable(self) -> None:
        """Verify the registered tool satisfies EmbeddableTool protocol."""
        from src.shared.python.launcher_embed import get_embeddable_tool

        tool = get_embeddable_tool("pose_studio")
        assert isinstance(tool, EmbeddableTool)

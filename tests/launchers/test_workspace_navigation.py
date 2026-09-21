"""Tests for task-oriented desktop navigation over existing embedded tools (ORG-05, #10515).

Tests:
1. Five primary task workspaces + Home + secondary utilities.
2. Services and model variants do not clutter Home.
3. Authoritative workspace membership and actions from metadata.
4. Saved tile layouts and favorites migrate through the alias map without resetting customizations.
5. Single-instance tool reuse policy: opening a tool from multiple links focuses the existing tool.
6. Dirty tab cancellation preserves state.
7. Keyboard navigation, visible focus, and accessible names.
8. Sidebar scrolling and narrow-window responsiveness.
9. Actionable status for missing/unavailable providers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QTabWidget, QToolButton, QWidget

from src.launchers.workspace_navigation import (
    ALIAS_MAP,
    PRIMARY_WORKSPACE_IDS,
    PRIMARY_WORKSPACES,
    SECONDARY_UTILITY_IDS,
    WorkspaceDestination,
    WorkspaceInfo,
    get_workspace_for_tool,
    get_workspace_tools,
    is_primary_workspace,
    migrate_favorites,
    migrate_model_order,
    migrate_saved_layout,
)


@pytest.mark.unit
class TestWorkspaceDefinitions:
    """Validate authoritative five primary task workspaces and utilities."""

    def test_exactly_five_primary_task_workspaces(self) -> None:
        """Requirement 1: Exactly five task workspace destinations exist."""
        assert len(PRIMARY_WORKSPACES) == 5
        assert set(PRIMARY_WORKSPACE_IDS) == {
            "capture_analyze",
            "model_match",
            "shot_course_lab",
            "optimize_train",
            "results_compare",
        }

    def test_primary_workspace_titles_match_specification(self) -> None:
        """Requirement 1: Titles match exact specification strings."""
        expected_titles = {
            "capture_analyze": "Capture & Analyze",
            "model_match": "Model & Match",
            "shot_course_lab": "Shot & Course Lab",
            "optimize_train": "Optimize & Train",
            "results_compare": "Results & Compare",
        }
        for ws_id, expected_title in expected_titles.items():
            assert ws_id in PRIMARY_WORKSPACES
            assert PRIMARY_WORKSPACES[ws_id].title == expected_title

    def test_global_utilities_and_secondary_navigation_defined(self) -> None:
        """Requirement 3: All Tools, Favorites, History, Dev/Research exist."""
        assert "all_tools" in SECONDARY_UTILITY_IDS
        assert "favorites" in SECONDARY_UTILITY_IDS
        assert "history" in SECONDARY_UTILITY_IDS
        assert "dev_research" in SECONDARY_UTILITY_IDS

    def test_home_does_not_contain_raw_services_or_model_variants(self) -> None:
        """RED Acceptance: Services/model variants do not fill Home."""
        home_excluded_services = {
            "aip",
            "realtime_ws",
            "motion_pipeline",
            "perturbation_analysis",
            "force_overlays",
            "actuator_controls",
            "robotics_module",
            "unreal_integration",
        }
        for ws in PRIMARY_WORKSPACES.values():
            overlap = set(ws.member_tool_ids) & home_excluded_services
            assert not overlap, (
                f"Raw services leaked into primary workspace {ws.id}: {overlap}"
            )


@pytest.mark.unit
class TestWorkspaceMembership:
    """Validate tool membership across the task workspaces."""

    def test_canonical_tools_mapped_to_correct_workspace(self) -> None:
        """Requirement 2: Tools belong to their logical task workspace."""
        # Capture & Analyze
        assert get_workspace_for_tool("motion_capture") == "capture_analyze"
        assert get_workspace_for_tool("video_analyzer") == "capture_analyze"
        assert get_workspace_for_tool("data_explorer") == "capture_analyze"

        # Model & Match
        assert get_workspace_for_tool("mujoco_unified") == "model_match"
        assert get_workspace_for_tool("drake_golf") == "model_match"
        assert get_workspace_for_tool("pinocchio_golf") == "model_match"
        assert get_workspace_for_tool("opensim_golf") == "model_match"
        assert get_workspace_for_tool("model_explorer") == "model_match"

        # Shot & Course Lab
        assert get_workspace_for_tool("putting_green") == "shot_course_lab"
        assert get_workspace_for_tool("golf_simulator") == "shot_course_lab"

        # Optimize & Train
        assert get_workspace_for_tool("tools_movement_optimizer") == "optimize_train"
        assert get_workspace_for_tool("swing_objective_lab") == "optimize_train"

        # Results & Compare
        assert get_workspace_for_tool("cross_engine_dashboard") == "results_compare"
        assert get_workspace_for_tool("canonical_core_comparison") == "results_compare"

    def test_developer_and_research_tools_in_secondary_workspace(self) -> None:
        """Requirement 3: Developer/research tools are secondary, not primary."""
        secondary_tools = [
            "character_builder",
            "aip",
            "realtime_ws",
            "actuator_controls",
        ]
        for tool_id in secondary_tools:
            ws = get_workspace_for_tool(tool_id)
            assert ws in (
                None,
                "dev_research",
            ), f"Secondary tool {tool_id} should not be in primary workspace {ws}"


@pytest.mark.unit
class TestAliasMigrationAndCustomization:
    """Validate migration through alias map without resetting customizations."""

    def test_alias_map_covers_all_known_legacy_aliases(self) -> None:
        """Requirement 4: Known retired and legacy aliases are mapped."""
        assert ALIAS_MAP["putting_green_gui"] == "putting_green"
        assert ALIAS_MAP["starting_pose_matcher"] == "motion_target_preview"
        assert ALIAS_MAP["cross_engine"] == "cross_engine_dashboard"
        assert ALIAS_MAP["matlab_unified"] == "matlab_suite"
        assert ALIAS_MAP["movement_optimizer"] == "tools_movement_optimizer"

    def test_migrate_model_order_replaces_aliases_and_deduplicates(self) -> None:
        """Saved model order with legacy aliases is updated without duplicates."""
        saved_order = [
            "putting_green_gui",
            "mujoco_unified",
            "cross_engine",
            "putting_green",
        ]
        migrated = migrate_model_order(saved_order)
        assert migrated == ["putting_green", "mujoco_unified", "cross_engine_dashboard"]

    def test_migrate_favorites_replaces_aliases_and_preserves_order(self) -> None:
        """Favorites with legacy aliases migrate to canonical IDs."""
        saved_favorites = ["cross_engine", "matlab_unified", "drake_golf"]
        migrated = migrate_favorites(saved_favorites)
        assert migrated == ["cross_engine_dashboard", "matlab_suite", "drake_golf"]

    def test_migrate_saved_layout_preserves_customizations(self) -> None:
        """User customizations (tile_scale, view_mode, dock_state) are preserved."""
        saved_layout = {
            "model_order": ["putting_green_gui", "drake_golf"],
            "favorites": ["cross_engine"],
            "tile_scale": 1.25,
            "view_mode": "LARGE",
            "dock_state": "state_blob_123",
            "launch_stats": {
                "putting_green_gui": {
                    "count": 5,
                    "last_launched": "2026-09-18T10:00:00",
                }
            },
        }
        migrated = migrate_saved_layout(saved_layout)
        assert migrated["model_order"] == ["putting_green", "drake_golf"]
        assert migrated["favorites"] == ["cross_engine_dashboard"]
        assert migrated["tile_scale"] == 1.25
        assert migrated["view_mode"] == "LARGE"
        assert migrated["dock_state"] == "state_blob_123"
        # Launch stats should merge into canonical key
        assert "putting_green" in migrated["launch_stats"]
        assert migrated["launch_stats"]["putting_green"]["count"] == 5


@pytest.mark.unit
class TestSingleInstanceAndDirtyTabPreservation:
    """Validate single instance tool reuse and dirty state protection."""

    def test_duplicate_links_focus_existing_tool_tab(self, qapp) -> None:
        """Requirement 4: Opening a second link focuses existing tab rather than duplicating."""
        from src.launchers.workspace_navigation import focus_or_open_tool_tab

        tab_widget = QTabWidget()
        page1 = QWidget()
        page1.setProperty("tool_id", "video_analyzer")
        tab_widget.addTab(page1, "Video Analyzer")

        opened_new = False

        def launcher_factory(tool_id: str) -> QWidget:
            nonlocal opened_new
            opened_new = True
            w = QWidget()
            w.setProperty("tool_id", tool_id)
            return w

        # Try to open video_analyzer again
        idx = focus_or_open_tool_tab(tab_widget, "video_analyzer", launcher_factory)
        assert idx == 0
        assert tab_widget.currentIndex() == 0
        assert tab_widget.count() == 1
        assert not opened_new, "Should reuse existing tab, not call factory"

    def test_dirty_tab_cancellation_preserves_widget(self, qapp, monkeypatch) -> None:
        """RED Acceptance: Cancelling close on a dirty tab preserves the widget."""
        from src.launchers.workspace_navigation import close_tool_tab_guarded

        tab_widget = QTabWidget()
        dirty_widget = QWidget()
        dirty_widget.setProperty("is_dirty", True)
        tab_widget.addTab(dirty_widget, "Dirty Tool")

        # Simulate user clicking 'Cancel' in confirmation dialog
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
        )

        closed = close_tool_tab_guarded(tab_widget, 0)
        assert not closed, "Close operation must be cancelled"
        assert tab_widget.count() == 1, "Tab must remain open when cancelled"


@pytest.mark.unit
class TestSidebarAccessibilityAndKeyboardNavigation:
    """Validate accessibility, keyboard navigation, and responsive scrolling."""

    def test_sidebar_buttons_have_accessible_metadata(self, qapp) -> None:
        """Requirement 5: Visible focus, accessible names and descriptions."""
        from src.launchers._launcher_navigation_ui import LauncherNavigationUIMixin

        class DummyLauncher(LauncherNavigationUIMixin):
            def __init__(self) -> None:
                self.launcher = QWidget()

        dummy = DummyLauncher()
        btn = dummy._build_sidebar_button("Capture & Analyze", "camera", checkable=True)
        assert btn.text() == "Capture & Analyze"
        assert btn.accessibleName() == "Capture & Analyze"
        assert "Capture & Analyze" in btn.accessibleDescription()
        assert btn.focusPolicy() == Qt.FocusPolicy.StrongFocus

    def test_sidebar_scroll_wrapper_enables_narrow_window_resizing(self, qapp) -> None:
        """Requirement 5: Narrow window scrolling via QScrollArea container."""
        from src.launchers._launcher_navigation_ui import LauncherNavigationUIMixin

        class DummyLauncher(LauncherNavigationUIMixin):
            def __init__(self) -> None:
                self.launcher = QWidget()

        dummy = DummyLauncher()
        sidebar, _layout, colors = dummy._create_sidebar_shell()
        scroll_area = dummy._wrap_sidebar(sidebar, colors)
        assert scroll_area.widgetResizable() is True
        assert scroll_area.widget() is sidebar


@pytest.mark.unit
class TestWorkspaceBreadcrumbs:
    """Validate return-to-workspace breadcrumb bar (ORG-05, #10515)."""

    def test_breadcrumb_renders_workspace_title_and_tool_name(self, qapp) -> None:
        """Breadcrumb renders '← Capture & Analyze / Video Analyzer'."""
        from src.launchers.workspace_navigation import WorkspaceBreadcrumbBar

        returned = []
        bar = WorkspaceBreadcrumbBar(
            workspace_id="capture_analyze",
            tool_name="Video Analyzer",
            on_return=lambda ws_id: returned.append(ws_id),
        )
        assert "Capture & Analyze" in bar.btn_workspace.text()
        assert bar.lbl_tool.text() == "Video Analyzer"
        assert (
            bar.btn_workspace.accessibleName()
            == "Return to Capture & Analyze workspace"
        )
        assert bar.btn_workspace.focusPolicy() == Qt.FocusPolicy.StrongFocus

    def test_breadcrumb_button_click_triggers_on_return_callback(self, qapp) -> None:
        """Clicking breadcrumb button invokes on_return with workspace ID."""
        from src.launchers.workspace_navigation import WorkspaceBreadcrumbBar

        returned = []
        bar = WorkspaceBreadcrumbBar(
            workspace_id="model_match",
            tool_name="MuJoCo Unified",
            on_return=lambda ws_id: returned.append(ws_id),
        )
        bar.btn_workspace.click()
        assert returned == ["model_match"]


@pytest.mark.unit
class TestDiscoverabilityAndExplanation:
    """Validate discoverability and actionable explanations for missing capabilities."""

    def test_ready_tool_status_explanation(self) -> None:
        """Ready tools report available=True with friendly launch message."""
        from src.launchers.workspace_navigation import explain_tool_status

        mock_model = MagicMock()
        mock_model.name = "MuJoCo Unified"
        mock_model.launcher = MagicMock(status="ready")
        available = {"mujoco_unified": mock_model}

        info = explain_tool_status("mujoco_unified", available)
        assert info["available"] is True
        assert info["status"] == "ready"
        assert "ready to launch" in info["explanation"]

    def test_external_dependency_explanation(self) -> None:
        """Tools with external dependencies provide helpful remediation steps."""
        from src.launchers.workspace_navigation import explain_tool_status

        info = explain_tool_status("matlab_suite")
        assert info["available"] is False
        assert info["status"] == "external_dependency"
        assert "MATLAB" in info["explanation"]

    def test_unregistered_capability_explanation(self) -> None:
        """Unregistered tools explain that they are not configured in catalog."""
        from src.launchers.workspace_navigation import explain_tool_status

        info = explain_tool_status("nonexistent_tool_xyz")
        assert info["available"] is False
        assert info["status"] == "unregistered"
        assert "not currently configured" in info["explanation"]


@pytest.mark.unit
class TestLayoutManagerWorkspaceFiltering:
    """Validate LayoutManager filtering by task workspaces."""

    def test_filter_by_workspace_id_limits_to_workspace_tools(self, tmp_path) -> None:
        """LayoutManager filters to members of the requested workspace."""
        from src.launchers.launcher_layout_manager import LayoutManager

        models = {
            "motion_capture": MagicMock(
                id="motion_capture", name="Mocap", description=""
            ),
            "video_analyzer": MagicMock(
                id="video_analyzer", name="Video", description=""
            ),
            "drake_golf": MagicMock(id="drake_golf", name="Drake", description=""),
            "putting_green": MagicMock(
                id="putting_green", name="Green", description=""
            ),
        }
        for m in models.values():
            m.hidden = False

        lm = LayoutManager(
            config_file=tmp_path / "layout.json",
            available_models=models,
            get_model_func=lambda mid: models.get(mid),
            create_card_func=lambda m: MagicMock(),
        )
        lm.model_order = list(models.keys())
        lm.current_category_filter = "workspace:capture_analyze"

        filtered = lm.get_filtered_order()
        assert set(filtered) == {"motion_capture", "video_analyzer"}

    def test_filter_by_workspace_title_limits_to_workspace_tools(
        self, tmp_path
    ) -> None:
        """Filtering by workspace title 'Model & Match' filters correctly."""
        from src.launchers.launcher_layout_manager import LayoutManager

        models = {
            "drake_golf": MagicMock(id="drake_golf", name="Drake", description=""),
            "video_analyzer": MagicMock(
                id="video_analyzer", name="Video", description=""
            ),
        }
        for m in models.values():
            m.hidden = False

        lm = LayoutManager(
            config_file=tmp_path / "layout.json",
            available_models=models,
            get_model_func=lambda mid: models.get(mid),
            create_card_func=lambda m: MagicMock(),
        )
        lm.model_order = list(models.keys())
        lm.current_category_filter = "Model & Match"

        filtered = lm.get_filtered_order()
        assert filtered == ["drake_golf"]

    def test_load_layout_migrates_via_alias_map(self, tmp_path) -> None:
        """load_layout migrates legacy aliases in saved configuration."""
        import json
        from src.launchers.launcher_layout_manager import LayoutManager

        models = {
            "putting_green": MagicMock(
                id="putting_green", name="Green", description=""
            ),
            "cross_engine_dashboard": MagicMock(
                id="cross_engine_dashboard", name="Cross", description=""
            ),
        }
        for m in models.values():
            m.hidden = False

        cfg_file = tmp_path / "layout.json"
        saved = {
            "model_order": ["putting_green_gui", "cross_engine"],
            "favorites": ["cross_engine"],
            "launch_stats": {"putting_green_gui": {"count": 3}},
            "tile_scale": 1.5,
        }
        cfg_file.write_text(json.dumps(saved), encoding="utf-8")

        lm = LayoutManager(
            config_file=cfg_file,
            available_models=models,
            get_model_func=lambda mid: models.get(mid),
            create_card_func=lambda m: MagicMock(),
        )
        loaded = lm.load_layout()
        assert loaded is not None
        assert "putting_green" in lm.model_order
        assert "cross_engine_dashboard" in lm.favorites
        assert lm.tile_scale == 1.5
        assert "putting_green" in lm.launch_stats

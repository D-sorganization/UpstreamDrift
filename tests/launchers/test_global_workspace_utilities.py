"""Regression and acceptance tests for Global Workspace Utilities (ORG-19, #10528).

Covers:
- Switching workspaces updates assistant context without duplicate sessions or stale run references.
- Old assistant/library/setup IDs remain reachable through canonical alias resolution.
- Dismissed onboarding stays dismissed across session migration and reload.
- Keyboard open/close restores focus to the previously focused widget.
- Browser platform environment refuses native-only controls fail-closed.
- Assistant conversation history persists across workspace navigation without deletion.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from src.shared.python.workspace.global_utilities import (
    AssistantContextSnapshot,
    GlobalWorkspaceUtilitiesCoordinator,
    NativeActionUnavailableError,
    OnboardingPreferences,
    PlatformExecutionEnvironment,
    UnknownUtilityError,
    WorkspaceContextualHelp,
)


def test_switching_workspaces_updates_assistant_context_without_duplicate_sessions() -> (
    None
):
    coordinator = GlobalWorkspaceUtilitiesCoordinator()

    initial_session_id = coordinator.session_id
    assert initial_session_id is not None

    # Initial context in capture workspace with a run
    ctx1 = coordinator.update_workspace_context(
        workspace_id="capture",
        project_id="proj-101",
        active_run_id="run-capture-001",
        source="workspace_navigation",
    )
    assert ctx1.workspace_id == "capture"
    assert ctx1.project_id == "proj-101"
    assert ctx1.active_run_id == "run-capture-001"
    assert ctx1.source == "workspace_navigation"
    assert coordinator.session_id == initial_session_id

    # Switch workspace to inspection: updates context, does NOT duplicate session,
    # and purges stale run reference when switching without a new run
    ctx2 = coordinator.update_workspace_context(
        workspace_id="inspection",
        project_id="proj-101",
        active_run_id=None,
        source="workspace_navigation",
    )
    assert ctx2.workspace_id == "inspection"
    assert ctx2.project_id == "proj-101"
    assert ctx2.active_run_id is None, (
        "Stale run reference must be cleared when navigating without run"
    )
    assert coordinator.session_id == initial_session_id, (
        "Session ID must remain stable across navigation"
    )

    # Context snapshot retains explicit source
    active_snapshot = coordinator.get_active_context()
    assert active_snapshot == ctx2


def test_old_assistant_library_setup_ids_remain_reachable() -> None:
    coordinator = GlobalWorkspaceUtilitiesCoordinator()

    # Old assistant IDs
    assert coordinator.resolve_utility_id("chat_assistant") == "sidekick"
    assert coordinator.resolve_utility_id("assistant") == "sidekick"
    assert coordinator.resolve_utility_id("ai_assistant") == "sidekick"
    assert coordinator.resolve_utility_id("sidekick") == "sidekick"

    # Old setup IDs
    assert coordinator.resolve_utility_id("config_setup_wizard") == "setup_wizard"
    assert coordinator.resolve_utility_id("setup") == "setup_wizard"
    assert coordinator.resolve_utility_id("setup_wizard") == "setup_wizard"

    # Old library IDs
    assert coordinator.resolve_utility_id("project_library") == "library"
    assert coordinator.resolve_utility_id("project_map") == "library"
    assert coordinator.resolve_utility_id("library") == "library"

    # Old help IDs
    assert coordinator.resolve_utility_id("contextual_help") == "help"
    assert coordinator.resolve_utility_id("workspace_help") == "help"
    assert coordinator.resolve_utility_id("help") == "help"

    # Unknown ID raises actionable error
    with pytest.raises(UnknownUtilityError, match="unrecognized_tool"):
        coordinator.resolve_utility_id("unrecognized_tool")


def test_dismissed_onboarding_stays_dismissed() -> None:
    coordinator = GlobalWorkspaceUtilitiesCoordinator()

    # Initially on first run, setup should be shown
    assert coordinator.should_show_first_run_setup() is True

    # Dismiss onboarding
    coordinator.dismiss_onboarding()
    assert coordinator.should_show_first_run_setup() is False
    assert coordinator.onboarding_preferences.onboarding_dismissed is True

    # Migration from legacy preferences preserves dismissal
    legacy_prefs = {
        "setup_complete": False,
        "skip_onboarding": True,
        "first_run_done": True,
    }
    migrated = coordinator.migrate_onboarding_preferences(legacy_prefs)
    assert migrated.onboarding_dismissed is True
    assert coordinator.should_show_first_run_setup() is False


def test_keyboard_open_close_restores_focus() -> None:
    coordinator = GlobalWorkspaceUtilitiesCoordinator()

    # Open utility from a specific focused widget
    open_state = coordinator.open_utility(
        utility_id="sidekick",
        previous_focus_widget_id="editor-input-area",
    )
    assert open_state.is_open is True
    assert open_state.active_utility_id == "sidekick"
    assert open_state.previous_focus_widget_id == "editor-input-area"

    # Close utility restores focus target
    restore_target = coordinator.close_utility("sidekick")
    assert restore_target == "editor-input-area"

    # Toggle open/close with keyboard shortcut
    state1 = coordinator.toggle_utility(
        "help", current_focus_widget_id="viewport-canvas"
    )
    assert state1.is_open is True
    assert state1.active_utility_id == "help"

    state2 = coordinator.toggle_utility("help")
    assert state2.is_open is False
    assert coordinator.get_focus_restore_target() == "viewport-canvas"


def test_browser_cannot_execute_native_only_controls() -> None:
    coordinator = GlobalWorkspaceUtilitiesCoordinator()

    # Desktop can execute desktop actions
    desktop_result = coordinator.execute_action(
        action_id="os_terminal",
        platform=PlatformExecutionEnvironment.DESKTOP,
    )
    assert desktop_result["status"] == "ok"
    assert desktop_result["executed"] is True

    # Browser cannot execute native-only controls (fails closed)
    for native_action in ("os_terminal", "jupyter", "mcp_server", "matlab"):
        with pytest.raises(
            NativeActionUnavailableError, match="cannot be executed in browser"
        ):
            coordinator.execute_action(
                action_id=native_action,
                platform=PlatformExecutionEnvironment.BROWSER,
            )

    # Browser can execute cross-platform safe actions
    web_result = coordinator.execute_action(
        action_id="view_help",
        platform=PlatformExecutionEnvironment.BROWSER,
    )
    assert web_result["status"] == "ok"
    assert web_result["executed"] is True


def test_assistant_history_persists_across_workspace_navigation() -> None:
    coordinator = GlobalWorkspaceUtilitiesCoordinator()

    # Add conversation messages
    coordinator.add_chat_message(
        role="user", content="How do I inspect optical markers?"
    )
    coordinator.add_chat_message(
        role="assistant", content="Navigate to the Pose Inspection workspace."
    )

    history_before = coordinator.get_chat_history()
    assert len(history_before) == 2

    # Switch workspaces multiple times
    coordinator.update_workspace_context("inspection", project_id="p1")
    coordinator.update_workspace_context("fitting", project_id="p1")
    coordinator.update_workspace_context("results", project_id="p1")

    # History must NOT be deleted or reset
    history_after = coordinator.get_chat_history()
    assert len(history_after) == 2
    assert history_after[0]["content"] == "How do I inspect optical markers?"
    assert history_after[1]["content"] == "Navigate to the Pose Inspection workspace."


def test_workspace_contextual_help_completeness() -> None:
    coordinator = GlobalWorkspaceUtilitiesCoordinator()

    workspaces = (
        "capture",
        "inspection",
        "model_calibration",
        "fitting",
        "dynamics",
        "course",
        "results",
        "optimization",
    )
    for ws in workspaces:
        help_doc = coordinator.get_workspace_help(ws)
        assert isinstance(help_doc, WorkspaceContextualHelp)
        assert help_doc.workspace_id == ws
        assert len(help_doc.title) > 0
        assert len(help_doc.purpose) > 0
        assert len(help_doc.required_inputs) > 0
        assert len(help_doc.produced_artifacts) > 0
        assert len(help_doc.next_compatible_actions) > 0
        assert len(help_doc.limitations) > 0

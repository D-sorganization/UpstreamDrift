"""Launcher coordination tests for resilient Sidekick startup.

These tests keep the local multitab UI independent from API availability and
guard against duplicate sidebar creation during readiness retries.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.launchers import launcher_sidekick_sidebar as sidebar_module
from src.launchers.launcher_sidekick_sidebar import SidekickSidebarManager
from src.launchers.upstream_drift_launcher import (
    SIDEKICK_API_HEALTHCHECK_MS,
    SIDEKICK_API_MAX_RESTARTS,
    SIDEKICK_API_READY_RETRY_MS,
    SIDEKICK_API_RESTART_DELAY_MS,
    UpstreamDriftLauncher,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_deferred_install_does_not_gate_local_sidebar_on_api() -> None:
    """Files/Workspace/etc. must load even while Chat is degraded."""
    calls: list[str] = []
    launcher = SimpleNamespace(
        _install_sidekick_sidebar=lambda: calls.append("install"),
        _seed_sidekick_workspace=lambda: calls.append("seed"),
        _monitor_sidekick_api_readiness=lambda: calls.append("monitor"),
        _sidekick_api_ready_for_sidebar=lambda: False,
    )

    UpstreamDriftLauncher._install_sidekick_sidebar_deferred(launcher)

    assert calls == ["install", "seed", "monitor"]


def test_sidebar_manager_install_is_idempotent() -> None:
    """Repeated readiness callbacks must never add duplicate splitter panes."""
    sidebar = MagicMock()
    launcher = SimpleNamespace(sidekick_sidebar=sidebar)
    manager = SidekickSidebarManager(launcher)

    with patch.object(
        SidekickSidebarManager,
        "_get_sidekick_module",
        side_effect=AssertionError("existing sidebar must be reused"),
    ):
        manager._install_sidekick_sidebar()

    sidebar.setVisible.assert_called_once_with(True)


def test_launcher_shutdown_delegates_to_sidekick_runtime_owner() -> None:
    """Host close must stop PTY-backed tabs before the Qt window disappears."""
    sidebar = SimpleNamespace(shutdown=MagicMock())
    launcher = SimpleNamespace(sidekick_sidebar=sidebar)

    UpstreamDriftLauncher._shutdown_sidekick_sidebar(launcher)

    sidebar.shutdown.assert_called_once_with()


def test_sidekick_import_paths_are_installed_before_first_import() -> None:
    """The local child copy must not win before canonical paths are available."""
    manager = SidekickSidebarManager(SimpleNamespace())
    order: list[str] = []

    with (
        patch.object(
            SidekickSidebarManager,
            "_install_sidekick_import_paths",
            side_effect=lambda _self: order.append("paths"),
        ),
        patch(
            "src.shared.python.gui_launcher.tools_sidebar_integration."
            "_import_sidebar_module",
            side_effect=lambda: order.append("import") or object(),
        ),
    ):
        manager._get_sidekick_module()

    assert order == ["paths", "import"]


def test_vendored_tools_precedes_mutable_sibling_checkout(tmp_path) -> None:
    """A pinned vendor source wins over an arbitrary dirty sibling worktree."""
    repo_root = tmp_path / "UpstreamDrift"
    vendor_src = repo_root / "vendor" / "ud-tools" / "src"
    sibling_src = tmp_path / "Tools" / "src"
    (vendor_src / "shared" / "python").mkdir(parents=True)
    (sibling_src / "shared" / "python").mkdir(parents=True)
    manager = SidekickSidebarManager(SimpleNamespace())
    installed: list[str] = []

    with (
        patch.object(sidebar_module, "REPOS_ROOT", repo_root),
        patch.object(
            SidekickSidebarManager,
            "_prepend_sys_path",
            side_effect=lambda path: installed.append(str(path)),
        ),
    ):
        manager._install_sidekick_import_paths()

    assert installed == [
        str(vendor_src),
        str(vendor_src / "shared" / "python"),
    ]


def test_explicit_tools_checkout_precedes_initialized_fallbacks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit parent checkout remains authoritative after installation."""
    repo_root = tmp_path / "UpstreamDrift"
    explicit_src = tmp_path / "CanonicalTools" / "src"
    vendor_src = repo_root / "vendor" / "ud-tools" / "src"
    sibling_src = tmp_path / "Tools" / "src"
    for source_root in (explicit_src, vendor_src, sibling_src):
        (source_root / "shared" / "python").mkdir(parents=True)

    synthetic_path = [
        str(explicit_src / "shared" / "python"),
        str(explicit_src),
        str(vendor_src / "shared" / "python"),
        str(vendor_src),
        str(sibling_src / "shared" / "python"),
        str(sibling_src),
        "tail",
    ]
    monkeypatch.setattr(sys, "path", synthetic_path)
    monkeypatch.setenv("TOOLS_REPO_PATH", str(explicit_src.parent))
    manager = SidekickSidebarManager(SimpleNamespace())

    with patch.object(sidebar_module, "REPOS_ROOT", repo_root):
        manager._install_sidekick_import_paths()

    assert synthetic_path[:2] == [
        str(explicit_src / "shared" / "python"),
        str(explicit_src),
    ]


def test_invalid_explicit_tools_checkout_does_not_fall_through(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed explicit contract must not silently select the vendor."""
    repo_root = tmp_path / "UpstreamDrift"
    (repo_root / "vendor" / "ud-tools" / "src").mkdir(parents=True)
    invalid_tools_root = tmp_path / "InvalidTools"
    invalid_tools_root.mkdir()
    monkeypatch.setenv("TOOLS_REPO_PATH", str(invalid_tools_root))
    manager = SidekickSidebarManager(SimpleNamespace())

    with (
        patch.object(sidebar_module, "REPOS_ROOT", repo_root),
        pytest.raises(RuntimeError, match="TOOLS_REPO_PATH"),
    ):
        manager._install_sidekick_import_paths()


def test_vendored_direct_packages_precede_legacy_alias_shims(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached bootstrap paths must be repositioned, not merely left in place."""
    repo_root = tmp_path / "UpstreamDrift"
    vendor_src = repo_root / "vendor" / "ud-tools" / "src"
    vendor_python = vendor_src / "shared" / "python"
    vendor_python.mkdir(parents=True)
    synthetic_path = [str(vendor_src), str(vendor_python), "tail"]
    monkeypatch.setattr(sys, "path", synthetic_path)
    manager = SidekickSidebarManager(SimpleNamespace())

    with patch.object(sidebar_module, "REPOS_ROOT", repo_root):
        manager._install_sidekick_import_paths()

    assert synthetic_path[:2] == [str(vendor_python), str(vendor_src)]


def test_readiness_monitor_passes_current_instance_identity() -> None:
    """A healthy API schedules continued liveness monitoring."""
    runtime = SimpleNamespace(instance_id="current-instance")
    launcher = SimpleNamespace(
        _sidekick_runtime_config=runtime,
        _sidekick_api_wait_started_at=None,
        _sidekick_api_restart_count=2,
        _sidekick_api_monitoring=True,
        background_api_process=MagicMock(),
        _monitor_sidekick_api_readiness=MagicMock(),
    )
    launcher.background_api_process.poll.return_value = None

    with (
        patch(
            "src.launchers.upstream_drift_launcher.check_sidekick_api_readiness"
        ) as readiness_check,
        patch("src.launchers.upstream_drift_launcher.QTimer.singleShot") as schedule,
    ):
        readiness_check.return_value = SimpleNamespace(
            ready=True,
            url="http://127.0.0.1:8123/readyz",
            status_code=200,
            detail="ready",
        )
        UpstreamDriftLauncher._monitor_sidekick_api_readiness(launcher)

    readiness_check.assert_called_once_with(expected_instance_id="current-instance")
    assert launcher._sidekick_api_wait_started_at is None
    assert launcher._sidekick_api_restart_count == 0
    schedule.assert_called_once_with(
        SIDEKICK_API_HEALTHCHECK_MS,
        launcher._monitor_sidekick_api_readiness,
    )


def test_closed_launcher_stops_sidekick_liveness_monitor() -> None:
    """A queued callback must become inert once host shutdown begins."""
    launcher = SimpleNamespace(_sidekick_api_monitoring=False)

    with (
        patch(
            "src.launchers.upstream_drift_launcher.check_sidekick_api_readiness"
        ) as readiness_check,
        patch("src.launchers.upstream_drift_launcher.QTimer.singleShot") as schedule,
    ):
        UpstreamDriftLauncher._monitor_sidekick_api_readiness(launcher)

    readiness_check.assert_not_called()
    schedule.assert_not_called()


def test_missing_runtime_contract_cannot_accept_unrelated_api() -> None:
    """A configuration failure remains degraded even if a stale API says ready."""
    launcher = SimpleNamespace(
        _sidekick_runtime_config=None,
        _sidekick_runtime_error="port conflict",
        _sidekick_api_wait_started_at=None,
        _sidekick_api_restart_count=0,
        background_api_process=None,
        _report_sidekick_api_failure=MagicMock(),
    )

    stale_readiness = SimpleNamespace(
        ready=True,
        url="http://127.0.0.1:8000/readyz",
        status_code=200,
        detail="ready",
    )
    with patch(
        "src.launchers.upstream_drift_launcher.check_sidekick_api_readiness",
        return_value=stale_readiness,
    ):
        UpstreamDriftLauncher._monitor_sidekick_api_readiness(launcher)

    launcher._report_sidekick_api_failure.assert_called_once_with(stale_readiness)


def test_dead_api_process_receives_bounded_restart() -> None:
    """A failed child is relaunched without reinstalling the local sidebar."""
    runtime = SimpleNamespace(instance_id="current-instance")
    dead_process = MagicMock()
    dead_process.poll.return_value = 1
    replacement = MagicMock()
    launcher = SimpleNamespace(
        _sidekick_runtime_config=runtime,
        _sidekick_api_wait_started_at=None,
        _sidekick_api_restart_count=0,
        background_api_process=dead_process,
        _restart_sidekick_background_api=MagicMock(return_value=replacement),
        _monitor_sidekick_api_readiness=MagicMock(),
        _report_sidekick_api_failure=MagicMock(),
    )

    with (
        patch(
            "src.launchers.upstream_drift_launcher.check_sidekick_api_readiness",
            return_value=SimpleNamespace(
                ready=False,
                url="http://127.0.0.1:8123/readyz",
                status_code=None,
                detail="connection refused",
            ),
        ),
        patch("src.launchers.upstream_drift_launcher.QTimer.singleShot") as schedule,
    ):
        UpstreamDriftLauncher._monitor_sidekick_api_readiness(launcher)

    assert launcher._sidekick_api_restart_count == 1
    assert launcher.background_api_process is replacement
    launcher._restart_sidekick_background_api.assert_called_once_with()
    schedule.assert_called_once()
    launcher._report_sidekick_api_failure.assert_not_called()


def test_delayed_readiness_rechecks_running_child_without_relaunch() -> None:
    """A slow but live child gets the full readiness window."""
    runtime = SimpleNamespace(instance_id="current-instance")
    running_process = MagicMock()
    running_process.poll.return_value = None
    launcher = SimpleNamespace(
        _sidekick_runtime_config=runtime,
        _sidekick_api_wait_started_at=None,
        _sidekick_api_restart_count=0,
        _sidekick_api_monitoring=True,
        _sidekick_api_was_ready=False,
        background_api_process=running_process,
        _launch_sidekick_background_api=MagicMock(),
        _monitor_sidekick_api_readiness=MagicMock(),
        _report_sidekick_api_failure=MagicMock(),
    )

    with (
        patch(
            "src.launchers.upstream_drift_launcher.check_sidekick_api_readiness",
            return_value=SimpleNamespace(
                ready=False,
                url="http://127.0.0.1:8123/readyz",
                status_code=None,
                detail="connection refused",
            ),
        ),
        patch(
            "src.launchers.upstream_drift_launcher.time.monotonic",
            return_value=10.0,
        ),
        patch("src.launchers.upstream_drift_launcher.QTimer.singleShot") as schedule,
    ):
        UpstreamDriftLauncher._monitor_sidekick_api_readiness(launcher)

    assert launcher._sidekick_api_wait_started_at == 10.0
    launcher._launch_sidekick_background_api.assert_not_called()
    launcher._report_sidekick_api_failure.assert_not_called()
    schedule.assert_called_once_with(
        SIDEKICK_API_READY_RETRY_MS,
        launcher._monitor_sidekick_api_readiness,
    )


def test_child_launch_failure_exhausts_retry_budget_observably() -> None:
    """Repeated launch failure is bounded and ends in a visible report."""
    runtime = SimpleNamespace(instance_id="current-instance")
    dead_process = MagicMock()
    dead_process.poll.return_value = 1
    launcher = SimpleNamespace(
        _sidekick_runtime_config=runtime,
        _sidekick_runtime_error="",
        _sidekick_api_wait_started_at=None,
        _sidekick_api_restart_count=SIDEKICK_API_MAX_RESTARTS - 1,
        _sidekick_api_monitoring=True,
        _sidekick_api_was_ready=False,
        background_api_process=dead_process,
        _restart_sidekick_background_api=MagicMock(return_value=None),
        _monitor_sidekick_api_readiness=MagicMock(),
        _report_sidekick_api_failure=MagicMock(),
    )
    unavailable = SimpleNamespace(
        ready=False,
        url="http://127.0.0.1:8123/readyz",
        status_code=None,
        detail="child launch failed",
    )

    with (
        patch(
            "src.launchers.upstream_drift_launcher.check_sidekick_api_readiness",
            return_value=unavailable,
        ),
        patch(
            "src.launchers.upstream_drift_launcher.time.monotonic",
            return_value=10.0,
        ),
        patch("src.launchers.upstream_drift_launcher.QTimer.singleShot") as schedule,
    ):
        UpstreamDriftLauncher._monitor_sidekick_api_readiness(launcher)
        UpstreamDriftLauncher._monitor_sidekick_api_readiness(launcher)

    assert launcher._sidekick_api_restart_count == SIDEKICK_API_MAX_RESTARTS
    assert launcher.background_api_process is None
    launcher._restart_sidekick_background_api.assert_called_once_with()
    schedule.assert_called_once_with(
        SIDEKICK_API_RESTART_DELAY_MS,
        launcher._monitor_sidekick_api_readiness,
    )
    launcher._report_sidekick_api_failure.assert_called_once_with(unavailable)

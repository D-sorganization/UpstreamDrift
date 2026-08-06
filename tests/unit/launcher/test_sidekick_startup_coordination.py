"""Launcher coordination tests for resilient Sidekick startup.

These tests keep the local multitab UI independent from API availability and
guard against duplicate sidebar creation during readiness retries.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess  # nosec B404 - fixed interpreter and local test script
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.launchers import launcher_sidekick_sidebar as sidebar_module
from src.launchers.launcher_sidekick_sidebar import SidekickSidebarManager
from src.launchers.sidekick_extension_overlay import (
    IncompleteParentSidekickRuntimeError,
)
from src.launchers.upstream_drift_launcher import (
    SIDEKICK_API_HEALTHCHECK_MS,
    SIDEKICK_API_MAX_RESTARTS,
    SIDEKICK_API_READY_RETRY_MS,
    SIDEKICK_API_RESTART_DELAY_MS,
    UpstreamDriftLauncher,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_sidebar_uses_canonical_tools_source_resolver() -> None:
    """Deferred sidebar startup must reuse the direct bootstrap contract."""
    resolver_module = importlib.import_module("src.launchers.tools_repo_path")

    assert (
        sidebar_module.resolve_tools_source_root
        is resolver_module.resolve_tools_source_root
    )


def test_direct_launcher_imports_critical_sidekick_modules_from_pinned_tools() -> None:
    """Production bootstrap must never resolve critical modules from child copies."""
    repo_root = Path(__file__).resolve().parents[3]
    vendor_python = (
        repo_root / "vendor" / "ud-tools" / "src" / "shared" / "python"
    ).resolve()
    module_names = (
        "chat.chat_dock_widget",
        "chat._chat_dock_widget_qt",
        "sidekick.ui.tools_sidebar",
    )
    script = "\n".join(
        [
            "import importlib, json",
            "import launch_upstream_drift",
            f"names = {module_names!r}",
            "print(json.dumps({name: importlib.import_module(name).__file__ for name in names}))",
        ]
    )
    env = os.environ.copy()
    env.pop("TOOLS_REPO_PATH", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(  # nosec B603 - fixed interpreter and local script
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout.strip().splitlines()[-1])
    assert set(resolved) == set(module_names)
    for module_name, module_file in resolved.items():
        path = Path(module_file).resolve()
        assert path.is_relative_to(vendor_python), (
            f"{module_name} resolved from {path}, expected pinned Tools under "
            f"{vendor_python}"
        )


def test_pinned_chat_never_forwards_launcher_token_to_remote_peer() -> None:
    """The consumed Tools pin must keep host capabilities on loopback only."""
    repo_root = Path(__file__).resolve().parents[3]
    script = "\n".join(
        [
            "from urllib.parse import parse_qsl, urlsplit",
            "import launch_upstream_drift",
            "from chat.chat_dock_widget import _build_native_websocket_url",
            "url = _build_native_websocket_url(",
            "    'wss://chat.example', '/ws/session', 'ephemeral-test-token'",
            ")",
            "query = dict(parse_qsl(urlsplit(url).query))",
            "assert 'launcher_token' not in query",
            "assert 'ephemeral-test-token' not in url",
        ]
    )
    env = os.environ.copy()
    env.pop("TOOLS_REPO_PATH", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"

    result = subprocess.run(  # nosec B603 - fixed interpreter and local script
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        "Pinned Tools forwarded a launcher capability to a non-loopback "
        f"WebSocket peer:\n{result.stderr}"
    )


def test_background_api_child_uses_explicit_tools_package_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API child must inherit the same authoritative Tools checkout."""
    repo_root = tmp_path / "UpstreamDrift"
    tools_root = tmp_path / "CanonicalTools"
    tools_source = tools_root / "src"
    (repo_root / "src").mkdir(parents=True)
    (tools_source / "shared" / "python").mkdir(parents=True)
    monkeypatch.setenv("TOOLS_REPO_PATH", str(tools_root))
    process = object()
    process_manager = MagicMock()
    process_manager.launch_module.return_value = process
    launcher = SimpleNamespace(process_manager=process_manager)

    with patch(
        "src.launchers.upstream_drift_launcher.REPOS_ROOT",
        repo_root,
    ):
        result = UpstreamDriftLauncher._launch_sidekick_background_api(launcher)

    assert result is process
    process_manager.launch_module.assert_called_once_with(
        name="background_api_server",
        module_name="src.api.server",
        cwd=repo_root,
        extra_python_paths=(
            tools_source / "shared" / "python",
            tools_source,
        ),
    )


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


def test_sidekick_import_paths_precede_canonical_api_import() -> None:
    """Startup must import the canonical API without lazy package lookup."""
    manager = SidekickSidebarManager(SimpleNamespace())
    order: list[str] = []
    canonical_api = object()

    with (
        patch.object(
            SidekickSidebarManager,
            "_install_sidekick_import_paths",
            side_effect=lambda _self: order.append("paths"),
        ),
        patch(
            "src.shared.python.gui_launcher.tools_sidebar_integration."
            "_import_sidebar_module",
            side_effect=AssertionError("canonical API import must win"),
        ),
        patch(
            "src.launchers.launcher_sidekick_sidebar.importlib.import_module",
            side_effect=lambda name: (
                order.append(name) or canonical_api
                if name == "sidekick.ui.tools_sidebar.api"
                else None
            ),
        ),
    ):
        module = manager._get_sidekick_module()

    assert module is canonical_api
    assert order == ["paths", "sidekick.ui.tools_sidebar.api"]


def test_vendored_tools_precedes_mutable_sibling_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pinned vendor source wins over an arbitrary dirty sibling worktree."""
    monkeypatch.delenv("TOOLS_REPO_PATH", raising=False)
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


def test_nested_worktree_finds_workspace_sibling_tools_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uninitialized worktree vendor falls back to workspace Tools."""
    monkeypatch.delenv("TOOLS_REPO_PATH", raising=False)
    workspace = tmp_path / "Repositories"
    repo_root = workspace / "UpstreamDrift" / ".codex-worktrees" / "feature"
    sibling_src = workspace / "Tools" / "src"
    (repo_root / "vendor" / "ud-tools").mkdir(parents=True)
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
        patch.object(sidebar_module, "_activate_source_extensions"),
    ):
        manager._install_sidekick_import_paths()

    assert installed == [
        str(sibling_src),
        str(sibling_src / "shared" / "python"),
    ]


def test_implicit_incomplete_tools_runtime_disables_optional_sidebar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing implicit Tools checkout must not abort the main launcher."""
    monkeypatch.delenv("TOOLS_REPO_PATH", raising=False)
    manager = SidekickSidebarManager(SimpleNamespace())

    with patch.object(
        SidekickSidebarManager,
        "_install_sidekick_import_paths",
        side_effect=IncompleteParentSidekickRuntimeError("missing runtime"),
    ):
        assert manager._get_sidekick_module() is None


def test_explicit_incomplete_tools_runtime_remains_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly selected but incomplete Tools checkout is an error."""
    tools_root = tmp_path / "Tools"
    (tools_root / "src").mkdir(parents=True)
    monkeypatch.setenv("TOOLS_REPO_PATH", str(tools_root))
    manager = SidekickSidebarManager(SimpleNamespace())

    with (
        patch.object(
            SidekickSidebarManager,
            "_install_sidekick_import_paths",
            side_effect=IncompleteParentSidekickRuntimeError("missing runtime"),
        ),
        pytest.raises(IncompleteParentSidekickRuntimeError, match="missing runtime"),
    ):
        manager._get_sidekick_module()


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


def test_selected_parent_source_activates_manifest_gated_extensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same selected Tools authority must configure extension loading."""
    repo_root = tmp_path / "UpstreamDrift"
    tools_root = tmp_path / "CanonicalTools"
    (repo_root / "src/shared/python").mkdir(parents=True)
    (tools_root / "src/shared/python").mkdir(parents=True)
    monkeypatch.setenv("TOOLS_REPO_PATH", str(tools_root))
    manager = SidekickSidebarManager(SimpleNamespace())

    with (
        patch.object(sidebar_module, "REPOS_ROOT", repo_root),
        patch.object(
            SidekickSidebarManager,
            "_prepend_tools_source_paths",
        ),
        patch.object(sidebar_module, "_activate_source_extensions") as activate,
    ):
        manager._install_sidekick_import_paths()

    activate.assert_called_once_with(tools_root / "src")


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
        pytest.raises(RuntimeError) as error,
    ):
        manager._install_sidekick_import_paths()

    assert str(error.value) == (
        "TOOLS_REPO_PATH must point to a Tools checkout containing a src/ "
        f"directory, got: {invalid_tools_root.resolve()}"
    )


def test_vendored_direct_packages_precede_legacy_alias_shims(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached bootstrap paths must be repositioned, not merely left in place."""
    monkeypatch.delenv("TOOLS_REPO_PATH", raising=False)
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

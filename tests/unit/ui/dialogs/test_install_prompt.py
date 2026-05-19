"""Unit tests for InstallPromptDialog (Phase 3 install-prompt UX).

All tests are headless-safe: PyQt6 widgets are instantiated but never
shown on screen.  The background ``_InstallWorker`` thread is always
patched out so no real subprocess runs.

Coverage requirements (see issue #5768):
    * Dialog is shown when feature is missing (not suppressed).
    * Yes path: worker starts, progress bar appears, registry refresh called.
    * No path: dialog rejected, no install, no suppression.
    * Don't-ask-again path: suppression written to prefs file.
    * Async install callback: ``install_finished`` signal emitted with
      correct (success, reason) values.
    * Hot-refresh: ``registry.refresh()`` is called on install completion.
    * Suppressed feature: ``prompt()`` returns SUPPRESSED without exec.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

# Use offscreen Qt platform for headless test environments.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

# Skip entire module if PyQt6 is unavailable.
pytest.importorskip("PyQt6")

if TYPE_CHECKING:
    from src.shared.python.ui.dialogs.install_prompt import InstallPromptDialog


# ---------------------------------------------------------------------------
# Module-level Qt app (created once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qt_app():
    """Provide a module-scoped QApplication for all dialog tests."""
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 runtime unavailable: {exc}")

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_prefs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the prefs file to a temporary directory."""
    import src.shared.python.ui.dialogs.install_prompt as mod

    fake_dir = tmp_path / ".upstreamdrift"
    fake_file = fake_dir / "prefs.json"
    monkeypatch.setattr(mod, "PREFS_DIR", fake_dir)
    monkeypatch.setattr(mod, "PREFS_FILE", fake_file)
    return fake_file


@pytest.fixture()
def dialog(qt_app, tmp_prefs: Path) -> InstallPromptDialog:
    """Return a fresh dialog instance with prefs in a temp location."""
    from src.shared.python.ui.dialogs.install_prompt import InstallPromptDialog

    dlg = InstallPromptDialog(feature_name="mujoco", package_name="MuJoCo")
    # Prevent exec() from blocking the event loop.
    dlg.exec = lambda: None  # type: ignore[method-assign]
    return dlg


# ---------------------------------------------------------------------------
# Prefs helpers
# ---------------------------------------------------------------------------


class TestPrefsHelpers:
    def test_is_suppressed_false_when_no_prefs_file(self, tmp_prefs: Path) -> None:
        from src.shared.python.ui.dialogs.install_prompt import is_suppressed

        assert is_suppressed("mujoco") is False

    def test_suppress_feature_writes_prefs(self, tmp_prefs: Path) -> None:
        from src.shared.python.ui.dialogs.install_prompt import (
            is_suppressed,
            suppress_feature,
        )

        suppress_feature("drake")

        assert tmp_prefs.exists()
        data = json.loads(tmp_prefs.read_text())
        assert "drake" in data.get("dont_ask_again_features", [])
        assert is_suppressed("drake") is True

    def test_suppress_feature_idempotent(self, tmp_prefs: Path) -> None:
        from src.shared.python.ui.dialogs.install_prompt import suppress_feature

        suppress_feature("drake")
        suppress_feature("drake")  # second call must not duplicate

        data = json.loads(tmp_prefs.read_text())
        assert data["dont_ask_again_features"].count("drake") == 1

    def test_suppress_feature_raises_on_empty_name(self, tmp_prefs: Path) -> None:
        from src.shared.python.ui.dialogs.install_prompt import suppress_feature

        with pytest.raises(ValueError):
            suppress_feature("")

    def test_is_suppressed_raises_on_empty_name(self, tmp_prefs: Path) -> None:
        from src.shared.python.ui.dialogs.install_prompt import is_suppressed

        with pytest.raises(ValueError):
            is_suppressed("")


# ---------------------------------------------------------------------------
# InstallPromptResult enum
# ---------------------------------------------------------------------------


class TestInstallPromptResult:
    def test_all_members_present(self) -> None:
        from src.shared.python.ui.dialogs.install_prompt import InstallPromptResult

        names = {m.name for m in InstallPromptResult}
        assert names == {"YES", "NO", "DONT_ASK", "SUPPRESSED"}


# ---------------------------------------------------------------------------
# Dialog construction
# ---------------------------------------------------------------------------


class TestInstallPromptDialogConstruction:
    def test_valid_construction(self, qt_app, tmp_prefs: Path) -> None:
        from src.shared.python.ui.dialogs.install_prompt import InstallPromptDialog

        dlg = InstallPromptDialog(feature_name="drake", package_name="Drake")
        assert dlg is not None

    def test_raises_on_empty_feature_name(self, qt_app, tmp_prefs: Path) -> None:
        from src.shared.python.ui.dialogs.install_prompt import InstallPromptDialog

        with pytest.raises(ValueError, match="feature_name"):
            InstallPromptDialog(feature_name="", package_name="Drake")

    def test_raises_on_empty_package_name(self, qt_app, tmp_prefs: Path) -> None:
        from src.shared.python.ui.dialogs.install_prompt import InstallPromptDialog

        with pytest.raises(ValueError, match="package_name"):
            InstallPromptDialog(feature_name="drake", package_name="")

    def test_dialog_title(self, dialog: InstallPromptDialog) -> None:
        title = dialog.windowTitle()
        assert "Missing" in title or "Dependency" in title

    def test_label_contains_feature_and_package(
        self, dialog: InstallPromptDialog
    ) -> None:
        label_text = dialog._label.text()
        assert "mujoco" in label_text
        assert "MuJoCo" in label_text

    def test_progress_bar_initially_hidden(self, dialog: InstallPromptDialog) -> None:
        assert dialog._progress.isHidden()


# ---------------------------------------------------------------------------
# Yes path
# ---------------------------------------------------------------------------


class TestYesPath:
    def _make_fake_worker(self) -> MagicMock:
        fake_worker = MagicMock()
        fake_worker.progress = MagicMock()
        fake_worker.progress.connect = MagicMock()
        fake_worker.finished = MagicMock()
        fake_worker.finished.connect = MagicMock()
        fake_worker.start = MagicMock()
        return fake_worker

    def test_yes_sets_result(
        self, dialog: InstallPromptDialog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.shared.python.ui.dialogs import install_prompt as mod
        from src.shared.python.ui.dialogs.install_prompt import InstallPromptResult

        monkeypatch.setattr(
            mod, "_InstallWorker", lambda *a, **kw: self._make_fake_worker()
        )
        dialog._on_yes()
        assert dialog._result == InstallPromptResult.YES

    def test_yes_shows_progress_bar(
        self, dialog: InstallPromptDialog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.shared.python.ui.dialogs import install_prompt as mod

        monkeypatch.setattr(
            mod, "_InstallWorker", lambda *a, **kw: self._make_fake_worker()
        )
        dialog._on_yes()
        # isHidden() is used here because isVisible() returns False for widgets
        # whose parent (the dialog) is not shown on screen in headless tests.
        assert not dialog._progress.isHidden()

    def test_yes_disables_buttons(
        self, dialog: InstallPromptDialog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.shared.python.ui.dialogs import install_prompt as mod

        monkeypatch.setattr(
            mod, "_InstallWorker", lambda *a, **kw: self._make_fake_worker()
        )
        dialog._on_yes()

        assert not dialog._yes_btn.isEnabled()
        assert not dialog._no_btn.isEnabled()
        assert not dialog._dont_ask_btn.isEnabled()

    def test_yes_starts_worker(
        self, dialog: InstallPromptDialog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.shared.python.ui.dialogs import install_prompt as mod

        fake = self._make_fake_worker()
        monkeypatch.setattr(mod, "_InstallWorker", lambda *a, **kw: fake)
        dialog._on_yes()
        fake.start.assert_called_once()


# ---------------------------------------------------------------------------
# No path
# ---------------------------------------------------------------------------


class TestNoPath:
    def test_no_sets_result(self, dialog: InstallPromptDialog) -> None:
        from src.shared.python.ui.dialogs.install_prompt import InstallPromptResult

        with patch.object(dialog, "reject"):
            dialog._on_no()

        assert dialog._result == InstallPromptResult.NO

    def test_no_does_not_suppress(
        self, dialog: InstallPromptDialog, tmp_prefs: Path
    ) -> None:
        with patch.object(dialog, "reject"):
            dialog._on_no()

        assert not tmp_prefs.exists()


# ---------------------------------------------------------------------------
# Don't-ask-again path
# ---------------------------------------------------------------------------


class TestDontAskPath:
    def test_dont_ask_sets_result(
        self, dialog: InstallPromptDialog, tmp_prefs: Path
    ) -> None:
        from src.shared.python.ui.dialogs.install_prompt import InstallPromptResult

        with patch.object(dialog, "reject"):
            dialog._on_dont_ask()

        assert dialog._result == InstallPromptResult.DONT_ASK

    def test_dont_ask_writes_suppression(
        self, dialog: InstallPromptDialog, tmp_prefs: Path
    ) -> None:
        from src.shared.python.ui.dialogs.install_prompt import is_suppressed

        with patch.object(dialog, "reject"):
            dialog._on_dont_ask()

        assert is_suppressed("mujoco")

    def test_dont_ask_rejects_dialog(
        self, dialog: InstallPromptDialog, tmp_prefs: Path
    ) -> None:
        with patch.object(dialog, "reject") as mock_reject:
            dialog._on_dont_ask()

        mock_reject.assert_called_once()


# ---------------------------------------------------------------------------
# Suppressed feature — prompt() short-circuits
# ---------------------------------------------------------------------------


class TestSuppressedFeature:
    def test_prompt_returns_suppressed_without_exec(
        self, qt_app, tmp_prefs: Path
    ) -> None:
        from src.shared.python.ui.dialogs.install_prompt import (
            InstallPromptDialog,
            InstallPromptResult,
            suppress_feature,
        )

        suppress_feature("mujoco")
        dlg = InstallPromptDialog(feature_name="mujoco", package_name="MuJoCo")

        exec_called: list[bool] = []
        dlg.exec = lambda: exec_called.append(True)  # type: ignore[method-assign]

        result = dlg.prompt()

        assert result == InstallPromptResult.SUPPRESSED
        assert not exec_called


# ---------------------------------------------------------------------------
# Async install callback + hot-refresh
# ---------------------------------------------------------------------------


class TestAsyncInstallCallback:
    def test_worker_finished_emits_signal(self, dialog: InstallPromptDialog) -> None:
        emitted: list[tuple[bool, str]] = []
        dialog.install_finished.connect(lambda s, r: emitted.append((s, r)))

        with (
            patch.object(dialog, "accept"),
            patch(
                "src.shared.python.ui.dialogs.install_prompt.refresh"
            ) as mock_refresh,
        ):
            dialog._on_worker_finished(True, "installed mujoco")
            mock_refresh.assert_called_once()

        assert emitted == [(True, "installed mujoco")]

    def test_worker_finished_calls_refresh_on_success(
        self, dialog: InstallPromptDialog
    ) -> None:
        with (
            patch.object(dialog, "accept"),
            patch(
                "src.shared.python.ui.dialogs.install_prompt.refresh"
            ) as mock_refresh,
        ):
            dialog._on_worker_finished(True, "installed mujoco")

        mock_refresh.assert_called_once()

    def test_worker_finished_calls_refresh_on_failure(
        self, dialog: InstallPromptDialog
    ) -> None:
        """refresh() must be called even when the install fails."""
        with (
            patch.object(dialog, "accept"),
            patch(
                "src.shared.python.ui.dialogs.install_prompt.refresh"
            ) as mock_refresh,
        ):
            dialog._on_worker_finished(False, "exit 1")

        mock_refresh.assert_called_once()

    def test_worker_finished_accepts_dialog(self, dialog: InstallPromptDialog) -> None:
        with (
            patch.object(dialog, "accept") as mock_accept,
            patch("src.shared.python.ui.dialogs.install_prompt.refresh"),
        ):
            dialog._on_worker_finished(True, "ok")

        mock_accept.assert_called_once()

    def test_worker_finished_hides_progress(self, dialog: InstallPromptDialog) -> None:
        dialog._progress.show()
        with (
            patch.object(dialog, "accept"),
            patch("src.shared.python.ui.dialogs.install_prompt.refresh"),
        ):
            dialog._on_worker_finished(True, "ok")

        assert dialog._progress.isHidden()

    def test_refresh_exception_does_not_crash_dialog(
        self, dialog: InstallPromptDialog
    ) -> None:
        emitted: list[tuple[bool, str]] = []
        dialog.install_finished.connect(lambda s, r: emitted.append((s, r)))

        with (
            patch.object(dialog, "accept"),
            patch(
                "src.shared.python.ui.dialogs.install_prompt.refresh",
                side_effect=RuntimeError("registry boom"),
            ),
        ):
            dialog._on_worker_finished(True, "installed")

        assert emitted == [(True, "installed")]


# ---------------------------------------------------------------------------
# _InstallWorker
# ---------------------------------------------------------------------------


class TestInstallWorker:
    def test_raises_on_empty_feature_name(self, qt_app) -> None:
        from src.shared.python.ui.dialogs.install_prompt import _InstallWorker

        with pytest.raises(ValueError, match="feature_name"):
            _InstallWorker("")

    def test_worker_emits_finished_on_success(
        self, qt_app, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.shared.python.feature_registry import InstallResult
        from src.shared.python.ui.dialogs.install_prompt import _InstallWorker

        fake_result = InstallResult(
            feature="mujoco",
            success=True,
            command="pip install mujoco",
            stdout="",
            stderr="",
            returncode=0,
            reason="installed mujoco",
        )
        monkeypatch.setattr(
            "src.shared.python.ui.dialogs.install_prompt.install_feature",
            lambda name: fake_result,
        )

        worker = _InstallWorker("mujoco")
        received: list[tuple[bool, str]] = []
        worker.finished.connect(lambda s, r: received.append((s, r)))
        worker.run()  # call run() directly — no real thread

        assert received == [(True, "installed mujoco")]

    def test_worker_emits_finished_on_failure(
        self, qt_app, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.shared.python.feature_registry import InstallResult
        from src.shared.python.ui.dialogs.install_prompt import _InstallWorker

        fake_result = InstallResult(
            feature="mujoco",
            success=False,
            command="pip install mujoco",
            stdout="",
            stderr="error output",
            returncode=1,
            reason="install failed for mujoco (exit 1)",
        )
        monkeypatch.setattr(
            "src.shared.python.ui.dialogs.install_prompt.install_feature",
            lambda name: fake_result,
        )

        worker = _InstallWorker("mujoco")
        received: list[tuple[bool, str]] = []
        worker.finished.connect(lambda s, r: received.append((s, r)))
        worker.run()

        assert received == [(False, "install failed for mujoco (exit 1)")]

    def test_worker_handles_unexpected_exception(
        self, qt_app, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.shared.python.ui.dialogs.install_prompt import _InstallWorker

        monkeypatch.setattr(
            "src.shared.python.ui.dialogs.install_prompt.install_feature",
            MagicMock(side_effect=RuntimeError("unexpected")),
        )

        worker = _InstallWorker("mujoco")
        received: list[tuple[bool, str]] = []
        worker.finished.connect(lambda s, r: received.append((s, r)))
        worker.run()

        assert len(received) == 1
        assert received[0][0] is False
        assert "unexpected" in received[0][1]


# ---------------------------------------------------------------------------
# src.core.capability_registry shim
# ---------------------------------------------------------------------------


class TestCapabilityRegistryShim:
    def test_shim_exports_get_registry(self) -> None:
        from src.core.capability_registry import get_registry

        assert callable(get_registry)

    def test_shim_exports_refresh(self) -> None:
        from src.core.capability_registry import refresh

        assert callable(refresh)

    def test_shim_get_registry_returns_same_instance(self) -> None:
        from src.core import capability_registry as shim
        from src.shared.python.feature_registry import get_registry as canonical

        assert shim.get_registry() is canonical()

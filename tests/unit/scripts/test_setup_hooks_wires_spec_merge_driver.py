"""``scripts/setup_hooks.py`` must register the ``spec-rows`` merge driver.

Issue #9476, second comment: the vendored installer was inert in this
repository -- nothing invoked it, so the ``spec-rows`` merge driver was never
registered for any contributor, while the vendored docstring pointed at a
Repository_Management-only entry point. The documented local-automation entry
point here is ``scripts/setup_hooks.py``; these tests pin that wiring: running
``setup_hooks.main()`` registers the driver through the vendored installer,
and the installer is invoked on this repository's resolved root.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_HOOKS = REPO_ROOT / "scripts" / "setup_hooks.py"


def _load_setup_hooks() -> ModuleType:
    """Import ``scripts/setup_hooks.py`` by path, as the repo's scripts do."""
    spec = importlib.util.spec_from_file_location(
        "ud_setup_hooks_under_test", SETUP_HOOKS
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_installer(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> MagicMock:
    """Replace the vendored installer loader with a recording fake."""
    fake = MagicMock()
    fake.install.return_value = 0
    monkeypatch.setattr(module, "_load_install_spec_merge_driver", lambda: fake)
    return fake


def test_main_registers_spec_merge_driver(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running setup end to end calls the installer on the repository root."""
    module = _load_setup_hooks()
    for name in (
        "install_pre_commit",
        "install_hooks",
        "install_push_hooks",
        "install_dev_dependencies",
        "verify_installation",
        "log_summary",
    ):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    fake = _fake_installer(monkeypatch, module)

    module.main()

    fake.install.assert_called_once_with(REPO_ROOT.resolve())


def test_install_spec_merge_driver_reports_install_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A non-zero installer result is reported, not silently swallowed."""
    import logging

    module = _load_setup_hooks()
    fake = MagicMock()
    fake.install.return_value = 1
    monkeypatch.setattr(module, "_load_install_spec_merge_driver", lambda: fake)
    caplog.set_level(logging.ERROR)

    module.install_spec_merge_driver()

    assert any(
        "spec-rows" in record.message.lower() for record in caplog.records
    ), "the failed registration must be reported to the operator"

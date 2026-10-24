"""The retired chat UI builder spelling aliases the pinned Tools module."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_import_order(tmp_path: Path, first: str, second: str) -> None:
    """Import both spellings in a clean process and check provider identity."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["QT_QPA_PLATFORM"] = "offscreen"
    code = f"""
import importlib
import sys
from pathlib import Path

repo = Path({str(_REPO_ROOT)!r})
sys.path[:0] = [
    str(repo),
    str(repo / "src"),
    str(repo / "vendor/ud-tools/src"),
    str(repo / "vendor/ud-tools/src/shared/python"),
]
first = importlib.import_module({first!r})
second = importlib.import_module({second!r})
consumer = importlib.import_module("src.shared.python.chat._chat_dock_widget_qt")
provider = (repo / "vendor/ud-tools/src/shared/python/chat/_qt/ui_builder.py").resolve()
assert Path(first.__file__).resolve() == provider
assert Path(second.__file__).resolve() == provider
assert first is second
assert consumer.ChatDockView is first.ChatDockView
assert consumer.build_chat_dock_ui is first.build_chat_dock_ui
assert consumer.mirror_chat_dock_view is first.mirror_chat_dock_view
for name in ("ChatDockView", "mirror_chat_dock_view", "build_chat_dock_ui"):
    assert getattr(first, name) is getattr(second, name)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            "shared.python.chat._qt.ui_builder",
            "src.shared.python.chat._qt.ui_builder",
        ),
        (
            "src.shared.python.chat._qt.ui_builder",
            "shared.python.chat._qt.ui_builder",
        ),
    ],
)
def test_legacy_and_canonical_import_orders_share_pinned_module(
    tmp_path: Path, first: str, second: str
) -> None:
    _run_import_order(tmp_path, first, second)


def test_exact_chat_alias_does_not_capture_ud_owned_roots() -> None:
    from src.shared.python.ud_import_alias_policy import UdSharedImportAliasFinder

    finder = UdSharedImportAliasFinder()
    assert finder._parse(
        "src.shared.python.model_generation.editor.attachment_ports"
    ) == (None, "")
    assert finder._parse("src.shared.python.humanoid_character_builder") == (
        None,
        "",
    )


def test_missing_provider_fails_without_using_the_local_ud_copy(
    tmp_path: Path,
) -> None:
    """A local same-name module cannot silently replace the opted-in provider."""
    code = f"""
import importlib
import sys
from pathlib import Path

repo = Path({str(_REPO_ROOT)!r})
sys.path[:0] = [str(repo), str(repo / "src")]
import src.shared.python.ud_import_alias_policy as policy
policy._TOOLS_SHARED_PYTHON = Path({str(tmp_path / "missing-vendor")!r})
try:
    importlib.import_module("shared.python.chat._qt.ui_builder")
except ModuleNotFoundError as error:
    assert error.name == "shared.python.chat._qt.ui_builder"
    assert "no vendored or installed Tools chat UI builder" in str(error)
else:
    raise AssertionError("missing provider fell back to UpstreamDrift's copy")
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("shared.python.chat._qt.ui_builder", "src.shared.python.chat._qt.ui_builder"),
        ("src.shared.python.chat._qt.ui_builder", "shared.python.chat._qt.ui_builder"),
    ],
)
def test_installed_tools_distribution_supplies_exact_builder_without_vendor(
    tmp_path: Path, first: str, second: str
) -> None:
    """A downstream Tools install remains a valid provider when the submodule is absent."""
    installed = tmp_path / "site-packages"
    shared_python = installed / "shared" / "python"
    vendor = _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"
    (installed / "shared").mkdir(parents=True)
    (installed / "shared" / "__init__.py").write_text("", encoding="utf-8")
    (shared_python / "__init__.py").parent.mkdir(parents=True)
    (shared_python / "__init__.py").write_text("", encoding="utf-8")
    import shutil

    shutil.copytree(vendor / "chat", shared_python / "chat")
    shutil.copy2(
        vendor / "chat" / "_thinking_indicator.py",
        shared_python / "chat" / "_thinking_indicator.py",
    )
    code = f"""
import importlib
import sys
from pathlib import Path

repo = Path({str(_REPO_ROOT)!r})
installed = Path({str(installed)!r})
sys.path[:0] = [str(installed), str(repo), str(repo / "src")]
import src.shared.python.ud_import_alias_policy as policy
policy._TOOLS_SHARED_PYTHON = installed / "missing-vendor"
first = importlib.import_module({first!r})
second = importlib.import_module({second!r})
module = importlib.import_module("shared.python.chat._qt.ui_builder")
legacy = importlib.import_module("src.shared.python.chat._qt.ui_builder")
assert first is second
expected = (installed / "shared/python/chat/_qt/ui_builder.py").resolve()
assert Path(module.__file__).resolve() == expected
assert module is legacy
for name in ("ChatDockView", "mirror_chat_dock_view", "build_chat_dock_ui"):
    assert getattr(module, name) is getattr(legacy, name)
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]

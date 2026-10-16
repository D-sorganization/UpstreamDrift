"""Headless startup regressions for ``launch_upstream_drift.py`` (UD #12882)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_CONTRACTS = REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"

pytestmark = pytest.mark.skipif(
    not (VENDORED_CONTRACTS / "contracts.py").is_file(),
    reason="vendor/ud-tools submodule not initialised",
)


def _run_launcher(*args: str, code: str | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    cmd = [sys.executable, str(REPO_ROOT / "launch_upstream_drift.py"), *args]
    if code is not None:
        cmd = [sys.executable, "-c", code]
    return subprocess.run(  # noqa: S603
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180
    )


@pytest.mark.unit
def test_classic_help_passes_parent_contract_guard() -> None:
    """The documented ``--classic`` command must not die in the contract guard."""
    result = _run_launcher("--classic", "--help")
    assert "Parent Tools contract resolution failed" not in result.stderr
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.unit
def test_load_parent_contracts_resolves_to_tools_module() -> None:
    """The guard returns the Tools-owned contracts module, not UD's shadow copy."""
    code = (
        "import sys; sys.argv=['x'];"
        "import launch_upstream_drift as L;"
        "m = L._load_parent_contracts();"
        "print(m.__file__)"
    )
    result = _run_launcher(code=code)
    assert result.returncode == 0, result.stderr[-2000:]
    resolved = Path(result.stdout.strip().splitlines()[-1]).resolve()
    assert resolved == (VENDORED_CONTRACTS / "contracts.py").resolve()


_LAUNCHER_PRELUDE = (
    "import sys; sys.argv=['x'];"
    "import launch_upstream_drift as L; L._load_parent_contracts();"
)


@pytest.mark.unit
def test_chat_service_builds_tools_compatible_registry_under_launcher() -> None:
    """UD #12893: launcher import order must not yield a second ToolRegistry class."""
    code = _LAUNCHER_PRELUDE + (
        "from unittest.mock import patch;"
        "from src.api.services.chat_service import ChatService;"
        "patch.object(ChatService,'_load_adapter',lambda self: None).start();"
        "ChatService(); print('CHAT_OK')"
    )
    result = _run_launcher(code=code)
    assert "registry must be a ToolRegistry instance" not in result.stderr
    assert "CHAT_OK" in result.stdout, result.stderr[-2000:]


@pytest.mark.unit
def test_data_explorer_embed_adapter_bootstraps_under_launcher() -> None:
    """UD #12895: ``shared.python.launcher_embed`` must resolve for Tools adapters."""
    code = _LAUNCHER_PRELUDE + (
        "import importlib;"
        "importlib.import_module('data_explorer._embed_adapter'); print('ADAPTER_OK')"
    )
    result = _run_launcher(code=code)
    assert "No module named 'shared.python.launcher_embed'" not in result.stderr
    assert "ADAPTER_OK" in result.stdout, result.stderr[-2000:]

"""Tests for the API-server bootstrap script.

The script lives at ``scripts/ci/start_api_server.py``. ``scripts/ci`` is not on
the pytest ``pythonpath``, so a bare ``import start_api_server`` never resolved
and a conftest allowlist rule silently dropped this whole file from collection
(#8006). It is loaded by path here and registered under its own name so the
``monkeypatch.setattr("start_api_server.X", ...)`` string targets below resolve.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "start_api_server.py"


def _load_start_api_server():
    """Import scripts/ci/start_api_server.py as the top-level ``start_api_server``."""
    if not _SCRIPT.is_file():
        raise AssertionError(f"expected the bootstrap script at {_SCRIPT}")
    spec = importlib.util.spec_from_file_location("start_api_server", _SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not build an import spec for {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["start_api_server"] = module
    spec.loader.exec_module(module)
    return module


start_api_server = _load_start_api_server()


def test_validate_security_no_issues(monkeypatch) -> None:
    mock_validate = MagicMock(return_value={"critical_issues": [], "warnings": []})
    monkeypatch.setattr(
        "src.shared.python.security.env_validator.validate_environment",
        mock_validate,
        raising=False,
    )

    # Needs to ensure it doesn't fail import
    assert start_api_server._validate_security() is True


def test_validate_security_with_critical(monkeypatch) -> None:
    mock_validate = MagicMock(
        return_value={"critical_issues": ["bad secret"], "warnings": []}
    )
    monkeypatch.setattr(
        "src.shared.python.security.env_validator.validate_environment",
        mock_validate,
        raising=False,
    )

    with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
        assert start_api_server._validate_security() is False

    with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
        assert start_api_server._validate_security() is True


def test_setup_api_environment(monkeypatch) -> None:
    mock_root = Path("/mock/root")
    monkeypatch.setattr("start_api_server.get_repo_root", lambda: mock_root)
    monkeypatch.setattr("start_api_server._validate_security", lambda: True)

    with patch.dict(
        os.environ, {"API_HOST": "0.0.0.0", "API_PORT": "9000"}, clear=True
    ):
        host, port = start_api_server.setup_api_environment()
        assert host == "0.0.0.0"
        assert port == 9000


@patch("start_api_server.uvicorn.run")
@patch("start_api_server.check_python_dependencies")
def test_main_success(mock_check_deps, mock_run, monkeypatch) -> None:
    mock_check_deps.return_value = True
    monkeypatch.setattr(
        "start_api_server.setup_api_environment", lambda: ("127.0.0.1", 8000)
    )
    monkeypatch.setattr("start_api_server.print_server_info", lambda h, p: None)

    assert start_api_server.main() == 0
    mock_run.assert_called_once()


@patch("start_api_server.check_python_dependencies")
def test_main_deps_fail(mock_check_deps) -> None:
    mock_check_deps.return_value = False
    assert start_api_server.main() == 1

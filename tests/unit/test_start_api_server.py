from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import start_api_server


def test_validate_security_no_issues(monkeypatch):
    mock_validate = MagicMock(return_value={"critical_issues": [], "warnings": []})
    monkeypatch.setattr(
        "src.shared.python.security.env_validator.validate_environment",
        mock_validate,
        raising=False,
    )

    # Needs to ensure it doesn't fail import
    assert start_api_server._validate_security() is True


def test_validate_security_with_critical(monkeypatch):
    mock_validate = MagicMock(return_value={"critical_issues": ["bad secret"], "warnings": []})
    monkeypatch.setattr(
        "src.shared.python.security.env_validator.validate_environment",
        mock_validate,
        raising=False,
    )

    with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
        assert start_api_server._validate_security() is False

    with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
        assert start_api_server._validate_security() is True


def test_setup_api_environment(monkeypatch):
    mock_root = Path("/mock/root")
    monkeypatch.setattr("start_api_server.get_repo_root", lambda: mock_root)
    monkeypatch.setattr("start_api_server._validate_security", lambda: True)

    with patch.dict(os.environ, {"API_HOST": "0.0.0.0", "API_PORT": "9000"}, clear=True):
        host, port = start_api_server.setup_api_environment()
        assert host == "0.0.0.0"
        assert port == 9000


@patch("start_api_server.uvicorn.run")
@patch("start_api_server.check_python_dependencies")
def test_main_success(mock_check_deps, mock_run, monkeypatch):
    mock_check_deps.return_value = True
    monkeypatch.setattr("start_api_server.setup_api_environment", lambda: ("127.0.0.1", 8000))
    monkeypatch.setattr("start_api_server.print_server_info", lambda h, p: None)

    assert start_api_server.main() == 0
    mock_run.assert_called_once()


@patch("start_api_server.check_python_dependencies")
def test_main_deps_fail(mock_check_deps):
    mock_check_deps.return_value = False
    assert start_api_server.main() == 1

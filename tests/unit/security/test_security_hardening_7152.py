"""Security-hardening tests for issue #7152.

- secure_subprocess cwd validation fails closed (no bare ValueError escapes).
- MCP npm package names from preset configs are validated before any subprocess.
- credentials module does not log env-var names at DEBUG.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# --- Defect 1: secure_subprocess cwd fails closed ---------------------------


def test_cwd_outside_roots_raises_secure_error(tmp_path: Path) -> None:
    from src.shared.python.security.secure_subprocess import (
        SecureSubprocessError,
        _validate_cwd_within_roots,
    )

    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(SecureSubprocessError):
        _validate_cwd_within_roots(outside, suite_root)


def test_cwd_pathological_value_raises_secure_error(tmp_path: Path) -> None:
    """A pathological cwd must surface SecureSubprocessError, never a bare
    ValueError from Path.is_relative_to (fail closed)."""
    from src.shared.python.security.secure_subprocess import (
        SecureSubprocessError,
        _validate_cwd_within_roots,
    )

    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    for bad in ("C:..", "../../somewhere", r"\\?\C:\nope"):
        with pytest.raises(SecureSubprocessError):
            _validate_cwd_within_roots(bad, suite_root)


def test_is_within_returns_false_on_comparison_failure() -> None:
    from src.shared.python.security.secure_subprocess import _is_within

    # Mixed/relative forms that can make is_relative_to raise must yield False.
    assert _is_within(Path("relative/child"), Path("/abs/parent")) is False


# --- Defect 2: npm package-name validation ----------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "; echo pwned",
        "https://evil/x.tgz",
        "../../local-path",
        "pkg@1.2.3",
        "git+https://github.com/x/y.git",
        "",
    ],
)
def test_invalid_npm_names_rejected(name: str) -> None:
    from src.shared.python.ai.mcp.config_loader import is_valid_npm_package_name

    assert is_valid_npm_package_name(name) is False


@pytest.mark.parametrize(
    "name",
    ["leftpad", "@scope/pkg", "some-server", "a.b_c~d", "@a-b/c.d"],
)
def test_valid_npm_names_accepted(name: str) -> None:
    from src.shared.python.ai.mcp.config_loader import is_valid_npm_package_name

    assert is_valid_npm_package_name(name) is True


# --- Defect 3: credentials does not log env-var names -----------------------


def test_credentials_debug_does_not_leak_env_var_name(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.shared.python.chat import credentials

    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    manager = credentials.CredentialManager()
    with caplog.at_level(logging.DEBUG):
        key = manager.get_api_key("openai")
    assert key == "sk-secret"
    for record in caplog.records:
        assert "OPENAI_API_KEY" not in record.getMessage()

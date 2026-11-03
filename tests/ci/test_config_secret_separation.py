"""Tests for config/secret separation and upgrade/rollback checks (#14274)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import check_config_secret_separation as mod

pytestmark = pytest.mark.unit


def _create_mock_repo(
    tmp_path: Path,
    *,
    gitignore_env: bool = True,
    dockerignore_env: bool = True,
    clean_env_example: bool = True,
    migration_has_downgrade: bool = True,
    settings_clean_secrets: bool = True,
) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    gi_lines = (
        [".env\n", ".env.*\n", "!.env.example\n"] if gitignore_env else ["*.pyc\n"]
    )
    (repo / ".gitignore").write_text("".join(gi_lines), encoding="utf-8")

    di_lines = (
        [".env\n", ".env.*\n", "!.env.example\n"]
        if dockerignore_env
        else ["node_modules\n"]
    )
    (repo / ".dockerignore").write_text("".join(di_lines), encoding="utf-8")

    if clean_env_example:
        env_ex = (
            "SECRET_KEY=change_me_in_production\n"
            "ADMIN_PASSWORD=change_me_in_production\n"
            "API_PORT=8000\n"
        )
    else:
        env_ex = (
            "SECRET_KEY=super_secret_real_prod_key_1234567890\n"
            "ADMIN_PASSWORD=RealSecretPass123!\n"
        )
    (repo / ".env.example").write_text(env_ex, encoding="utf-8")

    mig_dir = repo / "src" / "api" / "migrations" / "versions"
    mig_dir.mkdir(parents=True, exist_ok=True)
    downgrade_code = (
        "def downgrade():\n    op.drop_table('users')\n"
        if migration_has_downgrade
        else ""
    )
    mig_content = (
        '"""Initial migration."""\n'
        "def upgrade():\n"
        "    op.create_table('users')\n\n"
        f"{downgrade_code}"
    )
    (mig_dir / "20260323_0001_initial.py").write_text(mig_content, encoding="utf-8")

    sett_dir = repo / "src" / "shared" / "python" / "config"
    sett_dir.mkdir(parents=True, exist_ok=True)
    secret_field = (
        '    secret_token: str | None = Field(default=None, validation_alias="SECRET_TOKEN")\n'
        if settings_clean_secrets
        else '    secret_token: str = Field(default="hardcoded_secret_value", validation_alias="SECRET_TOKEN")\n'
    )
    sett_content = (
        "from pydantic import Field\n"
        "from pydantic_settings import BaseSettings\n\n"
        "class Settings(BaseSettings):\n"
        f"{secret_field}"
    )
    (sett_dir / "typed_settings.py").write_text(sett_content, encoding="utf-8")

    return repo


def test_check_config_secret_separation_passes_on_valid_repo(tmp_path: Path) -> None:
    repo = _create_mock_repo(tmp_path)
    report = mod.run_config_secret_checks(repo_root=repo)
    assert report["status"] == "passed"
    assert report["errors"] == []


def test_missing_gitignore_env_fails(tmp_path: Path) -> None:
    repo = _create_mock_repo(tmp_path, gitignore_env=False)
    report = mod.run_config_secret_checks(repo_root=repo)
    assert report["status"] == "failed"
    assert any(".gitignore" in err for err in report["errors"])


def test_missing_dockerignore_env_fails(tmp_path: Path) -> None:
    repo = _create_mock_repo(tmp_path, dockerignore_env=False)
    report = mod.run_config_secret_checks(repo_root=repo)
    assert report["status"] == "failed"
    assert any(".dockerignore" in err for err in report["errors"])


def test_real_secret_in_env_example_fails(tmp_path: Path) -> None:
    repo = _create_mock_repo(tmp_path, clean_env_example=False)
    report = mod.run_config_secret_checks(repo_root=repo)
    assert report["status"] == "failed"
    assert any("secret" in err.lower() for err in report["errors"])


def test_missing_migration_downgrade_fails(tmp_path: Path) -> None:
    repo = _create_mock_repo(tmp_path, migration_has_downgrade=False)
    report = mod.run_config_secret_checks(repo_root=repo)
    assert report["status"] == "failed"
    assert any("downgrade" in err.lower() for err in report["errors"])


def test_hardcoded_secret_in_settings_fails(tmp_path: Path) -> None:
    repo = _create_mock_repo(tmp_path, settings_clean_secrets=False)
    report = mod.run_config_secret_checks(repo_root=repo)
    assert report["status"] == "failed"
    assert any("hardcoded" in err.lower() for err in report["errors"])


def test_cli_invocation_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _create_mock_repo(tmp_path)
    exit_code = mod.main(["--repo-root", str(repo), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"

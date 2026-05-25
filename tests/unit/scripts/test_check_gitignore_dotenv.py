"""Tests for scripts/ci/check_gitignore_dotenv.py."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.check_gitignore_dotenv import check_gitignore_dotenv


@pytest.mark.unit
class TestCheckGitignoreDotenv:
    """Unit tests for the check_gitignore_dotenv helper."""

    def test_returns_true_when_dotenv_listed(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text(".env\n*.pyc\n", encoding="utf-8")
        assert check_gitignore_dotenv(gitignore) is True

    def test_returns_true_for_glob_pattern(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.env\n", encoding="utf-8")
        assert check_gitignore_dotenv(gitignore) is True

    def test_returns_true_for_dotenv_star(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text(".env.*\n!.env.example\n", encoding="utf-8")
        assert check_gitignore_dotenv(gitignore) is True

    def test_returns_false_when_dotenv_missing(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__/\n", encoding="utf-8")
        assert check_gitignore_dotenv(gitignore) is False

    def test_ignores_comment_lines(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("# .env is important\n*.pyc\n", encoding="utf-8")
        assert check_gitignore_dotenv(gitignore) is False

    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / ".gitignore"
        assert check_gitignore_dotenv(missing) is False

    def test_empty_gitignore_returns_false(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("", encoding="utf-8")
        assert check_gitignore_dotenv(gitignore) is False


@pytest.mark.unit
class TestCheckGitignoreDotenvScript:
    """Integration tests: run the script as a subprocess."""

    def test_script_exits_zero_on_real_repo(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/ci/check_gitignore_dotenv.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_script_exits_nonzero_when_env_missing(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, '.'); "
                "from scripts.ci.check_gitignore_dotenv import check_gitignore_dotenv; "
                f"sys.exit(0 if check_gitignore_dotenv(__import__('pathlib').Path({str(gitignore)!r})) else 1)",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

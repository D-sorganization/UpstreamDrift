"""Compose/Docker files carry no redaction placeholders or dev-only defaults (UD #15763)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDERS = ("<owner-home>", "<host>", "<redacted>")


def _container_files() -> list[Path]:
    files = sorted(REPO_ROOT.glob("docker-compose*.yml"))
    files += sorted(REPO_ROOT.glob("Dockerfile*"))
    return files


def _load(name: str) -> dict:
    return yaml.safe_load((REPO_ROOT / name).read_text(encoding="utf-8"))


def test_container_files_have_no_redaction_placeholders_in_active_lines() -> None:
    offenders = []
    for path in _container_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if any(token in line for token in PLACEHOLDERS):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert offenders == []


def test_compose_has_no_obsolete_version_key() -> None:
    assert "version" not in _load("docker-compose.yml")


def test_backend_container_path_is_container_native() -> None:
    env = _load("docker-compose.yml")["services"]["backend"]["environment"]
    path_entry = next(e for e in env if e.startswith("PATH="))
    assert path_entry == "PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"


def test_base_backend_does_not_default_to_reload() -> None:
    backend = _load("docker-compose.yml")["services"]["backend"]
    assert "--reload" not in str(backend.get("command", ""))
    assert "entrypoint" not in backend


def test_dev_reload_lives_in_the_override_file() -> None:
    backend = _load("docker-compose.override.yml")["services"]["backend"]
    assert "--reload" in backend["command"]
    assert backend["entrypoint"] == []

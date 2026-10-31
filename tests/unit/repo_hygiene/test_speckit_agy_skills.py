"""Spec Kit Antigravity (agy) skills mirror legacy Gemini command TOMLs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.unit


def _toml_to_skill_dir(stem: str) -> str:
    """Map ``speckit.git.commit`` to ``speckit-git-commit`` skill folder name."""
    if not stem.startswith("speckit."):
        msg = f"unexpected command stem: {stem}"
        raise ValueError(msg)
    suffix = stem.removeprefix("speckit.")
    return "speckit-" + suffix.replace(".", "-")


def test_git_gemini_command_tomls_have_agy_skill_counterparts() -> None:
    commands_dir = REPO_ROOT / ".gemini" / "commands"
    skills_root = REPO_ROOT / ".agents" / "skills"
    missing: list[str] = []
    for toml_path in sorted(commands_dir.glob("speckit.git.*.toml")):
        skill_dir = skills_root / _toml_to_skill_dir(toml_path.stem)
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            missing.append(toml_path.name)
    assert not missing, f"missing agy skills for: {', '.join(missing)}"


def test_agy_manifest_hashes_match_git_skill_files() -> None:
    manifest_path = REPO_ROOT / ".specify" / "integrations" / "agy.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: dict[str, str] = manifest["files"]
    mismatches: list[str] = []
    for rel_path, expected in files.items():
        path = REPO_ROOT / rel_path
        if not path.is_file():
            mismatches.append(f"missing managed file: {rel_path}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(f"hash drift: {rel_path}")
    assert not mismatches, "\n".join(mismatches)


def test_integration_json_lists_agy() -> None:
    integration_path = REPO_ROOT / ".specify" / "integration.json"
    state = json.loads(integration_path.read_text(encoding="utf-8"))
    installed = state.get("installed_integrations", [])
    assert "agy" in installed
    settings = state.get("integration_settings", {}).get("agy", {})
    assert settings.get("invoke_separator") == "-"

"""Regression guards for the provider-migration boundary modules."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BOUNDARY_FILES = (
    _ROOT / "src" / "launchers" / "launcher_model_sources.py",
    _ROOT / "src" / "launchers" / "launcher_provider_compatibility.py",
    _ROOT / "src" / "shared" / "python" / "config" / "model_registry.py",
)
_DISALLOWED_SNIPPETS = (
    'REPOS_ROOT / "src" / "engines"',
    'REPOS_ROOT / "src/config/models.yaml"',
    '"src/engines/physics_engines"',
)


def test_provider_migration_boundary_files_avoid_legacy_repo_path_shortcuts() -> None:
    """Migration boundary modules must stay on provider abstractions."""
    for file_path in _BOUNDARY_FILES:
        contents = file_path.read_text(encoding="utf-8")
        for snippet in _DISALLOWED_SNIPPETS:
            assert (
                snippet not in contents
            ), f"{file_path.name} contains legacy shortcut: {snippet}"

"""Tests for the single per-user config root and legacy migration (#8907).

Every test uses ``tmp_path`` for both the fake ``$HOME`` and the new root,
so the real user profile is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.data_io import user_config_root as ucr

pytestmark = pytest.mark.unit


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


@pytest.fixture
def new_root(tmp_path: Path) -> Path:
    return tmp_path / "config" / "upstream-drift" / "launcher"


def _seed(home: Path, legacy: str, name: str, text: str) -> Path:
    path = home / legacy / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_migration_copies_seeded_legacy_files(home: Path, new_root: Path) -> None:
    _seed(home, ".golf_modeling_suite", "preferences.json", '{"theme": "light"}')
    _seed(home, ".golf_modeling_suite", "library/library_index.db", "db")
    _seed(home, ".upstreamdrift", "onboarding_config.json", '{"dismissed": true}')

    created = ucr.migrate_legacy_user_dirs(new_root, home=home)

    assert {p.name for p in created} == {
        "preferences.json",
        "library",
        "onboarding_config.json",
    }
    assert (new_root / "preferences.json").read_text(encoding="utf-8") == (
        '{"theme": "light"}'
    )
    assert (new_root / "library" / "library_index.db").exists()
    assert (new_root / "onboarding_config.json").exists()
    assert (new_root / ucr.MIGRATION_MARKER).exists()
    # Legacy data is preserved (other subsystems still read these dirs).
    assert (home / ".golf_modeling_suite" / "preferences.json").exists()


def test_migration_is_idempotent(home: Path, new_root: Path) -> None:
    _seed(home, ".golf_modeling_suite", "recent_models.json", "[1]")
    assert ucr.migrate_legacy_user_dirs(new_root, home=home)

    # A user edit in the new root must survive a second start, even if the
    # legacy file changes too.
    (new_root / "recent_models.json").write_text("[2]", encoding="utf-8")
    _seed(home, ".golf_modeling_suite", "recent_models.json", "[3]")

    assert ucr.migrate_legacy_user_dirs(new_root, home=home) == []
    assert (new_root / "recent_models.json").read_text(encoding="utf-8") == "[2]"


def test_migration_never_clobbers_existing_new_file(home: Path, new_root: Path) -> None:
    _seed(home, ".golf_modeling_suite", "preferences.json", "legacy")
    _seed(home, ".golf_modeling_suite", "process_output.log", "old log")
    new_root.mkdir(parents=True)
    (new_root / "preferences.json").write_text("current", encoding="utf-8")

    created = ucr.migrate_legacy_user_dirs(new_root, home=home)

    assert [p.name for p in created] == ["process_output.log"]
    assert (new_root / "preferences.json").read_text(encoding="utf-8") == "current"


def test_migration_ignores_non_launcher_legacy_files(
    home: Path, new_root: Path
) -> None:
    """Tools-owned mcp_servers.json and chat data stay where their readers are."""
    _seed(home, ".upstreamdrift", "mcp_servers.json", "{}")
    _seed(home, ".golf_modeling_suite", "chat_sessions/a.json", "{}")

    assert ucr.migrate_legacy_user_dirs(new_root, home=home) == []
    assert not (new_root / "mcp_servers.json").exists()


def test_migration_without_legacy_dirs_writes_nothing(
    home: Path, new_root: Path
) -> None:
    assert ucr.migrate_legacy_user_dirs(new_root, home=home) == []
    assert not new_root.exists()


def test_failed_copy_leaves_no_marker_so_it_retries(
    home: Path, new_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(home, ".golf_modeling_suite", "preferences.json", "x")

    def _boom(*_args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ucr, "_copy_item", _boom)
    assert ucr.migrate_legacy_user_dirs(new_root, home=home) == []
    assert not (new_root / ucr.MIGRATION_MARKER).exists()

    monkeypatch.undo()
    assert ucr.migrate_legacy_user_dirs(new_root, home=home)


@pytest.mark.parametrize("bad", [Path("relative/root"), "not-a-path"])
def test_migration_rejects_non_absolute_paths(bad: object, home: Path) -> None:
    with pytest.raises(ValueError):
        ucr.migrate_legacy_user_dirs(bad, home=home)  # type: ignore[arg-type]


def test_user_config_dir_is_the_single_non_legacy_root() -> None:
    root = ucr.user_config_dir()
    assert root.is_absolute()
    assert root.name == "launcher"
    assert ".golf_modeling_suite" not in root.parts
    assert ".upstreamdrift" not in root.parts


def test_user_config_path_joins_under_root() -> None:
    assert ucr.user_config_path("library", "x.db") == (
        ucr.user_config_dir() / "library" / "x.db"
    )


@pytest.mark.parametrize("parts", [(), ("..", "escape.json"), ("/abs.json",)])
def test_user_config_path_rejects_escaping_paths(parts: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        ucr.user_config_path(*parts)

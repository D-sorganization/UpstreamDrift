"""Round-trip tests for Alembic migration downgrade/upgrade behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.script.base import Script
from sqlalchemy import create_engine, inspect

REPO_ROOT = Path(__file__).resolve().parents[2]


def _alembic_config(db_path: Path) -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _schema_snapshot(db_url: str) -> dict[str, dict[str, Any]]:
    engine = create_engine(db_url)
    inspector = inspect(engine)
    snapshot: dict[str, dict[str, Any]] = {}
    for table_name in sorted(inspector.get_table_names()):
        if table_name == "alembic_version":
            continue
        columns = inspector.get_columns(table_name)
        indexes = inspector.get_indexes(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        snapshot[table_name] = {
            "columns": [
                (
                    column["name"],
                    str(column["type"]),
                    column["nullable"],
                    column.get("default"),
                    column.get("primary_key"),
                )
                for column in columns
            ],
            "foreign_keys": [
                (
                    tuple(foreign_key["constrained_columns"]),
                    foreign_key["referred_table"],
                    tuple(foreign_key["referred_columns"]),
                )
                for foreign_key in foreign_keys
            ],
            "indexes": [
                (
                    index["name"],
                    tuple(index["column_names"]),
                    bool(index["unique"]),
                )
                for index in indexes
            ],
        }
    engine.dispose()
    return snapshot


def _downgrade_target(revision: Script) -> str:
    if revision.down_revision is None:
        return "base"
    if isinstance(revision.down_revision, str):
        return revision.down_revision
    down_revisions = cast(tuple[str, ...], revision.down_revision)
    if len(down_revisions) != 1:
        raise AssertionError(
            f"Round-trip test needs one parent for {revision.revision}, "
            f"got {down_revisions}"
        )
    return down_revisions[0]


def test_round_trip_through_every_revision(tmp_path: Path) -> None:
    """Every migration can downgrade and re-upgrade without changing head schema."""
    db_path = tmp_path / "round_trip.db"
    cfg = _alembic_config(db_path)
    db_url = cfg.get_main_option("sqlalchemy.url")
    assert db_url is not None
    script = ScriptDirectory.from_config(cfg)
    revisions = list(script.walk_revisions("base", "heads"))

    command.upgrade(cfg, "head")
    head_schema = _schema_snapshot(db_url)

    for revision in revisions:
        down_target = _downgrade_target(revision)
        command.downgrade(cfg, down_target)
        command.upgrade(cfg, revision.revision)
        assert _schema_snapshot(db_url) == head_schema

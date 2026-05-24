"""Tests for database startup behavior."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, text
from src.api import database


def test_init_db_in_production_verifies_alembic_head_without_create_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production startup must leave schema ownership to Alembic."""
    create_tables = Mock()
    verify_head = Mock()

    monkeypatch.setenv("UPSTREAM_DRIFT_ENV", "production")
    monkeypatch.setattr(database, "create_tables", create_tables)
    monkeypatch.setattr(database, "_assert_alembic_head_applied", verify_head)

    database.init_db()

    create_tables.assert_not_called()
    verify_head.assert_called_once_with()


def test_init_db_in_development_creates_tables_and_seeds_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-production startup keeps the existing create_all convenience path."""
    create_tables = Mock()
    session = Mock()
    user_query = Mock()
    role_filter = Mock()
    user_query.filter.return_value = role_filter
    role_filter.first.return_value = object()
    session.query.return_value = user_query

    monkeypatch.delenv("UPSTREAM_DRIFT_ENV", raising=False)
    monkeypatch.setattr(database, "create_tables", create_tables)
    monkeypatch.setattr(database, "SessionLocal", Mock(return_value=session))

    database.init_db()

    create_tables.assert_called_once_with()
    session.close.assert_called_once_with()


def test_assert_alembic_head_applied_passes_when_database_is_at_head(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The head verifier accepts a database stamped with the codebase head."""
    db_path = tmp_path / "at_head.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR)"))
        connection.execute(text("INSERT INTO alembic_version VALUES ('0001')"))

    monkeypatch.setattr(database, "engine", engine)

    database._assert_alembic_head_applied()


def test_assert_alembic_head_applied_raises_on_missing_version_table(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A production database without Alembic state must refuse startup."""
    db_path = tmp_path / "missing_version.db"
    engine = create_engine(f"sqlite:///{db_path}")

    monkeypatch.setattr(database, "engine", engine)

    with pytest.raises(RuntimeError, match="alembic_version"):
        database._assert_alembic_head_applied()


def test_assert_alembic_head_applied_raises_on_revision_mismatch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A production database behind the codebase head must refuse startup."""
    db_path = tmp_path / "behind_head.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR)"))
        connection.execute(text("INSERT INTO alembic_version VALUES ('old')"))

    monkeypatch.setattr(database, "engine", engine)

    with pytest.raises(RuntimeError, match="Database schema revision mismatch"):
        database._assert_alembic_head_applied()


def test_build_engine_uses_sqlite_static_pool() -> None:
    captured: dict[str, object] = {}

    def fake_create_engine(url: str, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return "sqlite-engine"

    original = database.create_engine
    database.create_engine = fake_create_engine
    try:
        engine = database._build_engine("sqlite:///./test.db")
    finally:
        database.create_engine = original

    assert engine == "sqlite-engine"
    assert captured["url"] == "sqlite:///./test.db"
    kwargs = captured["kwargs"]
    assert kwargs["connect_args"] == {"check_same_thread": False}
    assert kwargs["poolclass"] is database.StaticPool
    assert kwargs["echo"] is False
    assert "pool_pre_ping" not in kwargs
    assert "pool_recycle" not in kwargs
    assert "pool_size" not in kwargs


def test_build_engine_uses_env_driven_pool_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_create_engine(url: str, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return "postgres-engine"

    monkeypatch.setattr(database, "create_engine", fake_create_engine)
    monkeypatch.setattr(database, "get_database_pool_pre_ping", lambda: False)
    monkeypatch.setattr(database, "get_database_pool_recycle", lambda: 123)
    monkeypatch.setattr(database, "get_database_pool_size", lambda: 11)

    engine = database._build_engine("postgresql://db.example/upstream")

    assert engine == "postgres-engine"
    assert captured["url"] == "postgresql://db.example/upstream"
    assert captured["kwargs"] == {
        "pool_pre_ping": False,
        "pool_recycle": 123,
        "pool_size": 11,
        "echo": False,
    }

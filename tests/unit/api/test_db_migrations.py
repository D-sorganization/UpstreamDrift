"""Tests for database migration tooling.

Verifies that:
- Alembic is importable and configured correctly.
- The initial migration script is syntactically valid and has the expected
  metadata (revision ID, no parent revision).
- The migration can upgrade and downgrade a fresh in-memory SQLite database
  without errors.
- The db_migrate CLI helper parses commands correctly.
- The migrations env.py is importable and references the correct metadata.

Issue #2078: Add database migration tooling.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
INITIAL_MIGRATION_PATH = (
    REPO_ROOT
    / "src"
    / "api"
    / "migrations"
    / "versions"
    / "20260323_0000_0001_initial_schema.py"
)


def _import_migration_module() -> ModuleType:
    """Import the initial migration module dynamically."""
    spec = importlib.util.spec_from_file_location(
        "initial_migration", INITIAL_MIGRATION_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Alembic availability
# ---------------------------------------------------------------------------


class TestAlembicAvailability:
    """Verify Alembic is installed and importable."""

    def test_alembic_importable(self):
        """Alembic must be importable (dev dependency, issue #2078)."""
        import alembic  # noqa: F401

        assert alembic.__version__, "alembic.__version__ should be non-empty"

    def test_alembic_config_importable(self):
        """alembic.config.Config must be importable."""
        from alembic.config import Config

        assert Config is not None

    def test_alembic_command_importable(self):
        """alembic.command module must be importable."""
        import alembic.command  # noqa: F401

    def test_alembic_version_meets_minimum(self):
        """Alembic version must be >= 1.13.0 (pyproject.toml requirement)."""
        import alembic
        from packaging.version import Version

        try:
            current = Version(alembic.__version__)
            assert current >= Version("1.13.0"), (
                f"Alembic {alembic.__version__} is older than the required 1.13.0"
            )
        except ImportError:
            # packaging not installed — skip version comparison
            pytest.skip("packaging not installed, skipping version check")


# ---------------------------------------------------------------------------
# alembic.ini existence and syntax
# ---------------------------------------------------------------------------


class TestAlembicIni:
    """Verify alembic.ini is present and correct."""

    def test_alembic_ini_exists(self):
        """alembic.ini must exist at the repository root."""
        ini = REPO_ROOT / "alembic.ini"
        assert ini.exists(), f"alembic.ini not found at {ini}"

    def test_alembic_ini_parseable(self):
        """alembic.ini must be parseable as a valid config file."""
        import configparser

        ini = REPO_ROOT / "alembic.ini"
        cfg = configparser.ConfigParser()
        cfg.read(str(ini))
        assert cfg.has_section("alembic"), "alembic.ini missing [alembic] section"

    def test_alembic_ini_script_location(self):
        """script_location in alembic.ini must point to an existing directory."""
        import configparser

        ini = REPO_ROOT / "alembic.ini"
        cfg = configparser.ConfigParser()
        cfg.read(str(ini))
        script_location = cfg.get("alembic", "script_location")
        scripts_dir = REPO_ROOT / script_location
        assert scripts_dir.is_dir(), (
            f"script_location '{script_location}' does not exist as a directory"
        )

    def test_alembic_ini_versions_dir(self):
        """The versions/ directory referenced in alembic.ini must exist."""
        versions_dir = REPO_ROOT / "src" / "api" / "migrations" / "versions"
        assert versions_dir.is_dir(), f"versions/ directory not found at {versions_dir}"


# ---------------------------------------------------------------------------
# migrations/env.py
# ---------------------------------------------------------------------------


class TestMigrationsEnvPy:
    """Verify the Alembic env.py is correctly configured."""

    def test_env_py_exists(self):
        """env.py must exist in the migrations directory."""
        env = REPO_ROOT / "src" / "api" / "migrations" / "env.py"
        assert env.exists(), f"env.py not found at {env}"

    def test_env_py_imports_base_metadata(self):
        """env.py must import Base from src.api.auth.models."""
        env = REPO_ROOT / "src" / "api" / "migrations" / "env.py"
        source = env.read_text()
        assert "from src.api.auth.models import Base" in source, (
            "env.py must import Base from src.api.auth.models"
        )

    def test_env_py_sets_target_metadata(self):
        """env.py must assign target_metadata = Base.metadata."""
        env = REPO_ROOT / "src" / "api" / "migrations" / "env.py"
        source = env.read_text()
        assert "target_metadata = Base.metadata" in source, (
            "env.py must set target_metadata = Base.metadata"
        )

    def test_env_py_has_render_as_batch(self):
        """env.py must enable render_as_batch for SQLite ALTER TABLE support."""
        env = REPO_ROOT / "src" / "api" / "migrations" / "env.py"
        source = env.read_text()
        assert "render_as_batch=True" in source, (
            "env.py must set render_as_batch=True for SQLite ALTER TABLE emulation"
        )


# ---------------------------------------------------------------------------
# Initial migration script metadata
# ---------------------------------------------------------------------------


class TestInitialMigrationMetadata:
    """Verify the initial migration has correct Alembic metadata."""

    def test_initial_migration_exists(self):
        """The initial migration file must exist."""
        assert INITIAL_MIGRATION_PATH.exists(), (
            f"Initial migration not found at {INITIAL_MIGRATION_PATH}"
        )

    def test_initial_migration_revision_id(self):
        """Initial migration revision must be '0001'."""
        mod = _import_migration_module()
        assert mod.revision == "0001", f"Expected revision '0001', got '{mod.revision}'"

    def test_initial_migration_no_parent(self):
        """Initial migration must have no parent revision (down_revision is None)."""
        mod = _import_migration_module()
        assert mod.down_revision is None, (
            f"Initial migration should have no parent, got {mod.down_revision!r}"
        )

    def test_initial_migration_has_upgrade(self):
        """Initial migration must define an upgrade() function."""
        mod = _import_migration_module()
        assert callable(getattr(mod, "upgrade", None)), (
            "Initial migration must define upgrade()"
        )

    def test_initial_migration_has_downgrade(self):
        """Initial migration must define a downgrade() function."""
        mod = _import_migration_module()
        assert callable(getattr(mod, "downgrade", None)), (
            "Initial migration must define downgrade()"
        )


# ---------------------------------------------------------------------------
# Round-trip upgrade/downgrade on in-memory SQLite
# ---------------------------------------------------------------------------


class TestMigrationRoundTrip:
    """Run the initial migration up and back down on an in-memory SQLite DB."""

    @pytest.fixture()
    def alembic_cfg(self, tmp_path):
        """Return an Alembic Config pointing at a fresh temporary SQLite DB."""
        from alembic.config import Config

        db_path = tmp_path / "test_migrations.db"
        cfg = Config(str(REPO_ROOT / "alembic.ini"))
        cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
        return cfg

    def test_upgrade_to_head(self, alembic_cfg):
        """upgrade('head') must complete without errors on a fresh database."""
        from alembic import command

        command.upgrade(alembic_cfg, "head")

    def test_upgrade_then_downgrade(self, alembic_cfg):
        """upgrade then downgrade must leave the database at base (no tables)."""
        from alembic import command
        from sqlalchemy import create_engine, inspect

        db_url = alembic_cfg.get_main_option("sqlalchemy.url")

        command.upgrade(alembic_cfg, "head")

        # Verify tables were created
        engine = create_engine(db_url)
        with engine.connect():
            inspector = inspect(engine)
            tables_after_upgrade = set(inspector.get_table_names())
        assert "users" in tables_after_upgrade
        assert "api_keys" in tables_after_upgrade
        assert "sessions" in tables_after_upgrade

        command.downgrade(alembic_cfg, "base")

        # After downgrade, application tables must be absent
        engine2 = create_engine(db_url)
        with engine2.connect():
            inspector2 = inspect(engine2)
            tables_after_downgrade = set(inspector2.get_table_names())
        for tbl in ("users", "api_keys", "sessions"):
            assert tbl not in tables_after_downgrade, (
                f"Table '{tbl}' still present after downgrade to base"
            )

    def test_idempotent_upgrade(self, alembic_cfg):
        """Applying migrations twice must not raise an error."""
        from alembic import command

        command.upgrade(alembic_cfg, "head")
        # Second call should be a no-op (already at head)
        command.upgrade(alembic_cfg, "head")

    def test_current_returns_revision_after_upgrade(self, alembic_cfg, capsys):
        """current() must report the correct revision after upgrade."""
        from alembic import command

        command.upgrade(alembic_cfg, "head")
        command.current(alembic_cfg)
        captured = capsys.readouterr()
        # Alembic writes to stdout or stderr depending on version; check both
        output = captured.out + captured.err
        assert "0001" in output, f"Expected revision '0001' in output, got: {output!r}"


# ---------------------------------------------------------------------------
# db_migrate.py CLI helper
# ---------------------------------------------------------------------------


class TestDbMigrateCliParser:
    """Test the db_migrate.py CLI argument parser."""

    @pytest.fixture()
    def parser(self):
        """Return the argument parser from db_migrate."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "db_migrate", REPO_ROOT / "scripts" / "db_migrate.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.build_parser()

    def test_upgrade_defaults_to_head(self, parser):
        """'upgrade' without a revision argument defaults to 'head'."""
        args = parser.parse_args(["upgrade"])
        assert args.revision == "head"

    def test_upgrade_explicit_revision(self, parser):
        """'upgrade 0001' correctly sets revision."""
        args = parser.parse_args(["upgrade", "0001"])
        assert args.revision == "0001"

    def test_downgrade_parses_relative(self, parser):
        """'downgrade -1' correctly sets revision to '-1'."""
        args = parser.parse_args(["downgrade", "-1"])
        assert args.revision == "-1"

    def test_revision_autogenerate_flag(self, parser):
        """'revision --autogenerate -m msg' sets autogenerate=True."""
        args = parser.parse_args(["revision", "--autogenerate", "-m", "test msg"])
        assert args.autogenerate is True
        assert args.message == "test msg"

    def test_revision_message_default_none(self, parser):
        """'revision' without -m sets message to None."""
        args = parser.parse_args(["revision"])
        assert args.message is None

    def test_check_subcommand_exists(self, parser):
        """'check' sub-command must be recognised."""
        args = parser.parse_args(["check"])
        assert args.command == "check"

    def test_current_subcommand_exists(self, parser):
        """'current' sub-command must be recognised."""
        args = parser.parse_args(["current"])
        assert args.command == "current"

    def test_history_subcommand_exists(self, parser):
        """'history' sub-command must be recognised."""
        args = parser.parse_args(["history"])
        assert args.command == "history"


# ---------------------------------------------------------------------------
# db_migrate.py functional commands (with mocked Alembic)
# ---------------------------------------------------------------------------


class TestDbMigrateFunctions:
    """Test the functional command implementations in db_migrate.py."""

    @pytest.fixture()
    def db_migrate(self):
        """Import and return the db_migrate module."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "db_migrate_func", REPO_ROOT / "scripts" / "db_migrate.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def test_cmd_check_returns_0_on_success(self, db_migrate, tmp_path, capsys):
        """cmd_check returns 0 when migrations are in sync."""
        from alembic.config import Config

        db_path = tmp_path / "check_test.db"
        cfg = Config(str(REPO_ROOT / "alembic.ini"))
        cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")

        # Upgrade to head first so the check has a consistent database state
        from alembic import command

        command.upgrade(cfg, "head")

        with patch.object(db_migrate, "_get_alembic_config", return_value=cfg):
            import argparse

            args = argparse.Namespace(command="check", func=db_migrate.cmd_check)
            result = db_migrate.cmd_check(args)
        assert result == 0
        captured = capsys.readouterr()
        assert (
            "Migration check passed: models and migrations are in sync." in captured.out
        )

    def test_cmd_check_writes_failure_to_stderr(self, db_migrate, capsys):
        """cmd_check reports failures on stderr and returns 1."""
        import argparse

        with (
            patch.object(db_migrate, "_get_alembic_config", return_value=object()),
            patch("alembic.command.check", side_effect=RuntimeError("boom")),
        ):
            args = argparse.Namespace(command="check", func=db_migrate.cmd_check)
            result = db_migrate.cmd_check(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Migration check FAILED: boom" in captured.err
        assert (
            "Run: python3 scripts/db_migrate.py revision --autogenerate" in captured.err
        )

    def test_cmd_upgrade_calls_alembic_upgrade(self, db_migrate, tmp_path):
        """cmd_upgrade calls alembic.command.upgrade with correct args."""
        from alembic.config import Config

        db_path = tmp_path / "upgrade_test.db"
        cfg = Config(str(REPO_ROOT / "alembic.ini"))
        cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")

        with patch.object(db_migrate, "_get_alembic_config", return_value=cfg):
            import argparse

            args = argparse.Namespace(
                command="upgrade",
                revision="head",
                func=db_migrate.cmd_upgrade,
            )
            result = db_migrate.cmd_upgrade(args)
        assert result == 0

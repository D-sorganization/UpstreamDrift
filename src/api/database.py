"""Database configuration and session management."""

import os
from collections.abc import Generator
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql import text

from src.shared.python.config.environment import (
    get_database_pool_pre_ping,
    get_database_pool_recycle,
    get_database_pool_size,
    get_database_url,
)

# Base imported locally in create_tables to avoid circular import

# Database configuration
DATABASE_URL = get_database_url(
    default="sqlite:///./golf_modeling_suite.db",
)


def _build_engine(database_url: str):
    """Create a SQLAlchemy engine for the configured database URL."""
    if database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )

    return create_engine(
        database_url,
        pool_pre_ping=get_database_pool_pre_ping(),
        pool_recycle=get_database_pool_recycle(),
        pool_size=get_database_pool_size(),
        echo=False,
    )


engine = _build_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables() -> None:
    """Create all database tables."""
    from src.api.auth.models import Base

    Base.metadata.create_all(bind=engine)


def _repo_root() -> Path:
    """Return the repository root that contains alembic.ini."""
    return Path(__file__).resolve().parents[2]


def _get_codebase_alembic_heads() -> set[str]:
    """Return Alembic head revisions from the checked-out migration scripts."""
    cfg = Config(str(_repo_root() / "alembic.ini"))
    script = ScriptDirectory.from_config(cfg)
    return set(script.get_heads())


def _get_database_alembic_heads() -> set[str]:
    """Return the currently applied Alembic revision heads.

    Postcondition: returns the non-empty revision set recorded in
    ``alembic_version`` or raises ``RuntimeError`` with startup-safe context.
    """
    try:
        with engine.connect() as connection:
            rows = connection.execute(text("SELECT version_num FROM alembic_version"))
            revisions = {row[0] for row in rows if row[0]}
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "Database schema revision could not be verified because "
            "alembic_version is missing or unreadable. Run Alembic migrations "
            "before starting production."
        ) from exc

    if not revisions:
        raise RuntimeError(
            "Database schema revision could not be verified because "
            "alembic_version has no recorded revision."
        )
    return revisions


def _assert_alembic_head_applied() -> None:
    """Raise unless the database Alembic revision matches the codebase head.

    Production startup depends on this check so schema changes are applied by
    the deployment migration step, not by SQLAlchemy ``create_all()``.
    Postcondition: returns only when all codebase Alembic heads are applied.
    """
    code_heads = _get_codebase_alembic_heads()
    database_heads = _get_database_alembic_heads()
    if database_heads != code_heads:
        raise RuntimeError(
            "Database schema revision mismatch: "
            f"database has {sorted(database_heads)}, "
            f"codebase expects {sorted(code_heads)}. "
            "Run `python3 scripts/db_migrate.py upgrade head` before startup."
        )


def _is_production_environment() -> bool:
    """Return whether startup is running under the production DB contract."""
    return os.getenv("UPSTREAM_DRIFT_ENV", "").strip().lower() == "production"


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database with tables and default data."""
    import logging
    import secrets

    logger = logging.getLogger(__name__)

    if _is_production_environment():
        _assert_alembic_head_applied()
        logger.info("Production database schema revision verified at Alembic head")
        return

    create_tables()

    # Create default admin user if none exists
    db = SessionLocal()
    try:
        from src.api.auth.models import User, UserRole
        from src.api.auth.security import security_manager

        admin_user = db.query(User).filter(User.role == UserRole.ADMIN.value).first()
        if not admin_user:
            # SECURITY: Get password from environment variable
            admin_password = os.getenv("GOLF_ADMIN_PASSWORD")

            if not admin_password:
                # Generate a secure random password if not set
                admin_password = secrets.token_urlsafe(16)
                logger.warning(
                    "SECURITY: No GOLF_ADMIN_PASSWORD environment variable set. "
                    "Generated temporary admin password. Set GOLF_ADMIN_PASSWORD "
                    "environment variable for production."
                )
                # SECURITY FIX: Never log passwords in plaintext
                # Instead, provide instructions for recovery
                logger.info(
                    "Admin user created with randomly generated password. "
                    "To set a custom password, set the GOLF_ADMIN_PASSWORD "
                    "environment variable before starting the server, or use "
                    "the password reset API endpoint."
                )

            admin_user = User(
                email="admin@golfmodelingsuite.com",
                hashed_password=security_manager.hash_password(admin_password),
                full_name="System Administrator",
                role=UserRole.ADMIN.value,
                is_active=True,
                is_verified=True,
            )
            db.add(admin_user)
            db.commit()
            logger.info("Admin user created successfully")

    finally:
        db.close()

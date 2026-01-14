"""Database configuration and session management."""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api.auth.models import Base

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./golf_modeling_suite.db",  # Default to SQLite for development
)

# Create engine with appropriate settings
if DATABASE_URL.startswith("sqlite"):
    # SQLite-specific configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,  # Set to True for SQL debugging
    )
else:
    # PostgreSQL/other database configuration
    engine = create_engine(
        DATABASE_URL, pool_pre_ping=True, pool_recycle=300, echo=False
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables() -> None:
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


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

    create_tables()

    # Create default admin user if none exists
    db = SessionLocal()
    try:
        from api.auth.models import User, UserRole
        from api.auth.security import security_manager

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

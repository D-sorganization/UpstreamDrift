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
    create_tables()

    # Create default admin user if none exists
    db = SessionLocal()
    try:
        from api.auth.models import User, UserRole
        from api.auth.security import security_manager

        admin_user = db.query(User).filter(User.role == UserRole.ADMIN.value).first()
        if not admin_user:
            admin_user = User(
                email="admin@golfmodelingsuite.com",
                hashed_password=security_manager.hash_password("admin123"),
                full_name="System Administrator",
                role=UserRole.ADMIN.value,
                is_active=True,
                is_verified=True,
            )
            db.add(admin_user)
            db.commit()

    finally:
        db.close()

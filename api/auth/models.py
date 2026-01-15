"""Authentication and user models for Golf Modeling Suite API."""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

# Create the base class for SQLAlchemy models
Base = declarative_base()

if TYPE_CHECKING:
    # For type checking, we need to tell MyPy that Base is a class
    from sqlalchemy.ext.declarative import DeclarativeMeta

    Base = DeclarativeMeta


class UserRole(str, Enum):
    """User roles for role-based access control."""

    FREE = "free"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


class SubscriptionStatus(str, Enum):
    """Subscription status options."""

    ACTIVE = "active"
    CANCELED = "canceled"
    PAST_DUE = "past_due"
    TRIALING = "trialing"
    INCOMPLETE = "incomplete"


# SQLAlchemy Models
class User(Base):  # type: ignore[misc,valid-type]
    """User database model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    organization = Column(String(255), nullable=True)
    role = Column(String(50), default=UserRole.FREE.value)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)

    # Subscription info
    subscription_status = Column(String(50), default=SubscriptionStatus.TRIALING.value)
    subscription_id = Column(String(255), nullable=True)  # Stripe subscription ID
    customer_id = Column(String(255), nullable=True)  # Stripe customer ID

    # Usage tracking
    api_calls_this_month = Column(Integer, default=0)
    video_analyses_this_month = Column(Integer, default=0)
    simulations_this_month = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)


class APIKey(Base):  # type: ignore[misc,valid-type]
    """API key database model.

    Performance Note: key_prefix enables O(1) lookup filtering.
    Instead of loading all keys and verifying with bcrypt (O(n)),
    we first filter by prefix (SHA256 of first 8 chars), then verify
    only the matching candidates with bcrypt.
    """

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    key_hash = Column(String(255), unique=True, index=True, nullable=False)
    # Performance Issue #2 fix: Fast lookup prefix (SHA256 of first 8 chars)
    key_prefix = Column(String(64), index=True, nullable=True)
    name = Column(String(255), nullable=False)  # User-friendly name
    is_active = Column(Boolean, default=True)

    # Usage tracking
    last_used = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)


class Session(Base):  # type: ignore[misc,valid-type]
    """User session database model."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_token = Column(String(255), unique=True, index=True, nullable=False)
    refresh_token = Column(String(255), unique=True, index=True, nullable=True)

    # Session metadata
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())


# Pydantic Models for API
class UserBase(BaseModel):
    """Base user model."""

    email: EmailStr
    full_name: str | None = None
    organization: str | None = None


class UserCreate(UserBase):
    """User creation model."""

    password: str = Field(
        ..., min_length=8, description="Password must be at least 8 characters"
    )


class UserUpdate(BaseModel):
    """User update model."""

    full_name: str | None = None
    organization: str | None = None
    password: str | None = Field(None, min_length=8)


class UserResponse(UserBase):
    """User response model (excludes sensitive data)."""

    id: int
    role: UserRole
    is_active: bool
    is_verified: bool
    subscription_status: SubscriptionStatus
    api_calls_this_month: int
    video_analyses_this_month: int
    simulations_this_month: int
    created_at: datetime
    last_login: datetime | None = None

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """Login request model."""

    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    """Login response model."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserResponse


class APIKeyCreate(BaseModel):
    """API key creation model."""

    name: str = Field(
        ..., min_length=1, max_length=255, description="Friendly name for the API key"
    )


class APIKeyResponse(BaseModel):
    """API key response model."""

    id: int
    name: str
    key: str | None = None  # Only returned on creation
    is_active: bool
    last_used: datetime | None = None
    usage_count: int
    created_at: datetime
    expires_at: datetime | None = None

    class Config:
        from_attributes = True


class UsageQuotas(BaseModel):
    """Usage quotas for different subscription tiers."""

    api_calls_per_month: int
    video_analyses_per_month: int
    simulations_per_month: int
    max_simulation_duration: int  # seconds
    max_video_length: int  # seconds
    concurrent_requests: int


# Subscription tier quotas
SUBSCRIPTION_QUOTAS = {
    UserRole.FREE: UsageQuotas(
        api_calls_per_month=1000,
        video_analyses_per_month=5,
        simulations_per_month=10,
        max_simulation_duration=30,
        max_video_length=60,
        concurrent_requests=1,
    ),
    UserRole.PROFESSIONAL: UsageQuotas(
        api_calls_per_month=50000,
        video_analyses_per_month=500,
        simulations_per_month=1000,
        max_simulation_duration=300,
        max_video_length=600,
        concurrent_requests=5,
    ),
    UserRole.ENTERPRISE: UsageQuotas(
        api_calls_per_month=1000000,
        video_analyses_per_month=10000,
        simulations_per_month=50000,
        max_simulation_duration=3600,
        max_video_length=3600,
        concurrent_requests=20,
    ),
}

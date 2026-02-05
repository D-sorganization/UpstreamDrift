"""Tests for auth/models - Authentication and user models.

These tests verify the authentication models using Design by Contract principles.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError


class TestUserRoleContract:
    """Design by Contract tests for UserRole enum."""

    def test_is_string_enum(self):
        """Postcondition: UserRole inherits from str and Enum."""
        from src.api.auth.models import UserRole

        assert issubclass(UserRole, str)
        assert UserRole.FREE.value == "free"

    def test_all_values_unique(self):
        """Postcondition: All role values are unique."""
        from src.api.auth.models import UserRole

        values = [r.value for r in UserRole]
        assert len(values) == len(set(values))

    def test_has_required_roles(self):
        """Postcondition: Has all required roles."""
        from src.api.auth.models import UserRole

        roles = {r.name for r in UserRole}
        assert "FREE" in roles
        assert "PROFESSIONAL" in roles
        assert "ENTERPRISE" in roles
        assert "ADMIN" in roles


class TestSubscriptionStatusContract:
    """Design by Contract tests for SubscriptionStatus enum."""

    def test_is_string_enum(self):
        """Postcondition: SubscriptionStatus inherits from str and Enum."""
        from src.api.auth.models import SubscriptionStatus

        assert issubclass(SubscriptionStatus, str)
        assert SubscriptionStatus.ACTIVE.value == "active"

    def test_all_values_unique(self):
        """Postcondition: All status values are unique."""
        from src.api.auth.models import SubscriptionStatus

        values = [s.value for s in SubscriptionStatus]
        assert len(values) == len(set(values))


class TestUserBaseContract:
    """Design by Contract tests for UserBase Pydantic model."""

    def test_requires_email(self):
        """Precondition: Email is required."""
        from src.api.auth.models import UserBase

        with pytest.raises(ValidationError):
            UserBase()  # type: ignore[call-arg]

    def test_validates_email_format(self):
        """Precondition: Email must be valid format."""
        from src.api.auth.models import UserBase

        with pytest.raises(ValidationError):
            UserBase(email="not-an-email")

    def test_accepts_valid_email(self):
        """Postcondition: Valid email is accepted."""
        from src.api.auth.models import UserBase

        user = UserBase(email="test@example.com")
        assert user.email == "test@example.com"


class TestUserBase:
    """Functional tests for UserBase model."""

    def test_optional_full_name(self):
        """Test that full_name is optional."""
        from src.api.auth.models import UserBase

        user = UserBase(email="test@example.com")
        assert user.full_name is None

    def test_optional_organization(self):
        """Test that organization is optional."""
        from src.api.auth.models import UserBase

        user = UserBase(email="test@example.com")
        assert user.organization is None

    def test_with_all_fields(self):
        """Test with all fields provided."""
        from src.api.auth.models import UserBase

        user = UserBase(
            email="test@example.com",
            full_name="Test User",
            organization="Test Org",
        )
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.organization == "Test Org"


class TestUserCreateContract:
    """Design by Contract tests for UserCreate model."""

    def test_requires_password(self):
        """Precondition: Password is required."""
        from src.api.auth.models import UserCreate

        with pytest.raises(ValidationError):
            UserCreate(email="test@example.com")  # type: ignore[call-arg]

    def test_password_minimum_length(self):
        """Precondition: Password must be at least 8 characters."""
        from src.api.auth.models import UserCreate

        with pytest.raises(ValidationError):
            UserCreate(email="test@example.com", password="short")

    def test_accepts_valid_password(self):
        """Postcondition: Valid password is accepted."""
        from src.api.auth.models import UserCreate

        user = UserCreate(email="test@example.com", password="securepassword123")
        assert user.password == "securepassword123"


class TestUserUpdateContract:
    """Design by Contract tests for UserUpdate model."""

    def test_all_fields_optional(self):
        """Postcondition: All fields are optional."""
        from src.api.auth.models import UserUpdate

        update = UserUpdate()
        assert update.full_name is None
        assert update.organization is None
        assert update.password is None

    def test_password_minimum_length_if_provided(self):
        """Precondition: Password must be at least 8 chars if provided."""
        from src.api.auth.models import UserUpdate

        with pytest.raises(ValidationError):
            UserUpdate(password="short")


class TestUserResponseContract:
    """Design by Contract tests for UserResponse model."""

    def test_includes_required_fields(self):
        """Postcondition: UserResponse includes all required fields."""
        from src.api.auth.models import (
            SubscriptionStatus,
            UserResponse,
            UserRole,
        )

        user = UserResponse(
            id=1,
            email="test@example.com",
            role=UserRole.FREE,
            is_active=True,
            is_verified=False,
            subscription_status=SubscriptionStatus.ACTIVE,
            api_calls_this_month=0,
            video_analyses_this_month=0,
            simulations_this_month=0,
            created_at=datetime.now(),
        )
        assert user.id == 1
        assert user.email == "test@example.com"
        assert user.role == UserRole.FREE


class TestLoginRequestContract:
    """Design by Contract tests for LoginRequest model."""

    def test_requires_email_and_password(self):
        """Precondition: Both email and password required."""
        from src.api.auth.models import LoginRequest

        with pytest.raises(ValidationError):
            LoginRequest(email="test@example.com")  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            LoginRequest(password="password123")  # type: ignore[call-arg]

    def test_accepts_valid_credentials(self):
        """Postcondition: Valid credentials are accepted."""
        from src.api.auth.models import LoginRequest

        req = LoginRequest(email="test@example.com", password="password123")
        assert req.email == "test@example.com"
        assert req.password == "password123"


class TestLoginResponseContract:
    """Design by Contract tests for LoginResponse model."""

    def test_has_required_fields(self):
        """Postcondition: Has all required response fields."""
        from src.api.auth.models import (
            LoginResponse,
            SubscriptionStatus,
            UserResponse,
            UserRole,
        )

        user_response = UserResponse(
            id=1,
            email="test@example.com",
            role=UserRole.FREE,
            is_active=True,
            is_verified=True,
            subscription_status=SubscriptionStatus.ACTIVE,
            api_calls_this_month=0,
            video_analyses_this_month=0,
            simulations_this_month=0,
            created_at=datetime.now(),
        )

        response = LoginResponse(
            access_token="access_token",
            refresh_token="refresh_token",
            expires_in=3600,
            user=user_response,
        )

        assert response.access_token == "access_token"
        assert response.refresh_token == "refresh_token"
        assert response.token_type == "bearer"
        assert response.expires_in == 3600


class TestAPIKeyCreateContract:
    """Design by Contract tests for APIKeyCreate model."""

    def test_requires_name(self):
        """Precondition: Name is required."""
        from src.api.auth.models import APIKeyCreate

        with pytest.raises(ValidationError):
            APIKeyCreate()  # type: ignore[call-arg]

    def test_name_min_length(self):
        """Precondition: Name must be at least 1 character."""
        from src.api.auth.models import APIKeyCreate

        with pytest.raises(ValidationError):
            APIKeyCreate(name="")

    def test_name_max_length(self):
        """Precondition: Name must be at most 255 characters."""
        from src.api.auth.models import APIKeyCreate

        with pytest.raises(ValidationError):
            APIKeyCreate(name="a" * 256)


class TestUsageQuotasContract:
    """Design by Contract tests for UsageQuotas model."""

    def test_has_all_quota_fields(self):
        """Postcondition: Has all required quota fields."""
        from src.api.auth.models import UsageQuotas

        quotas = UsageQuotas(
            api_calls_per_month=1000,
            video_analyses_per_month=10,
            simulations_per_month=100,
            max_simulation_duration=60,
            max_video_length=120,
            concurrent_requests=5,
        )

        assert quotas.api_calls_per_month == 1000
        assert quotas.video_analyses_per_month == 10
        assert quotas.simulations_per_month == 100


class TestSubscriptionQuotas:
    """Tests for SUBSCRIPTION_QUOTAS configuration."""

    def test_all_roles_have_quotas(self):
        """Postcondition: All roles have defined quotas."""
        from src.api.auth.models import SUBSCRIPTION_QUOTAS, UserRole

        assert UserRole.FREE in SUBSCRIPTION_QUOTAS
        assert UserRole.PROFESSIONAL in SUBSCRIPTION_QUOTAS
        assert UserRole.ENTERPRISE in SUBSCRIPTION_QUOTAS

    def test_enterprise_has_higher_limits_than_professional(self):
        """Postcondition: Enterprise has higher limits."""
        from src.api.auth.models import SUBSCRIPTION_QUOTAS, UserRole

        pro = SUBSCRIPTION_QUOTAS[UserRole.PROFESSIONAL]
        ent = SUBSCRIPTION_QUOTAS[UserRole.ENTERPRISE]

        assert ent.api_calls_per_month > pro.api_calls_per_month
        assert ent.video_analyses_per_month > pro.video_analyses_per_month
        assert ent.simulations_per_month > pro.simulations_per_month

    def test_professional_has_higher_limits_than_free(self):
        """Postcondition: Professional has higher limits than free."""
        from src.api.auth.models import SUBSCRIPTION_QUOTAS, UserRole

        free = SUBSCRIPTION_QUOTAS[UserRole.FREE]
        pro = SUBSCRIPTION_QUOTAS[UserRole.PROFESSIONAL]

        assert pro.api_calls_per_month > free.api_calls_per_month
        assert pro.video_analyses_per_month > free.video_analyses_per_month
        assert pro.simulations_per_month > free.simulations_per_month

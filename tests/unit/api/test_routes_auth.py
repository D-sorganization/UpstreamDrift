"""Unit tests for the auth API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.auth import router
from src.api.database import get_db


class MockUser:
    def __init__(self, **kwargs):
        self.id = 1
        self.email = "test@example.com"
        self.full_name = "Test User"
        self.organization = "Test Org"
        self.role = "free"
        self.is_active = True
        self.is_verified = False
        self.hashed_password = "hashed_pw"
        self.last_login = None
        self.api_calls_this_month = 0
        self.video_analyses_this_month = 0
        self.simulations_this_month = 0
        self.subscription_status = "active"
        import datetime

        self.created_at = datetime.datetime.now()
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockQuery:
    def __init__(self, user=None):
        self.user = user

    def filter(self, *args):
        return self

    def first(self):
        return self.user


class MockDB:
    def __init__(self, user=None):
        self.user = user

    def query(self, model):
        return MockQuery(self.user)

    def add(self, obj):
        pass

    def commit(self):
        pass

    def refresh(self, obj):
        obj.id = 1
        obj.api_calls_this_month = 0
        obj.video_analyses_this_month = 0
        obj.simulations_this_month = 0
        obj.subscription_status = "active"
        import datetime

        obj.created_at = datetime.datetime.now()


def mock_get_db_factory(user=None):
    def get_mock_db():
        yield MockDB(user)

    return get_mock_db


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the auth router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_auth_uses_shared_limiter() -> None:
    """Issue #6636 F4: auth must use the SHARED limiter, not a private one.

    SlowAPI only enforces ``@limiter.limit(...)`` when the decorator's limiter
    is the same instance wired into ``app.state.limiter`` and the middleware
    (set in server.py to ``src.api.rate_limit.limiter``). A separate Limiter()
    in auth.py would make the advertised login/register limits silently
    non-enforcing.
    """
    from src.api import rate_limit
    from src.api.routes import auth as auth_module

    assert auth_module.limiter is rate_limit.limiter


def test_refresh_and_usage_use_typed_response_models() -> None:
    """Issue #6636 F4: /refresh and /usage use typed Pydantic response models."""
    from src.api.auth.models import RefreshTokenResponse, UsageSummaryResponse

    routes = {r.path: r for r in router.routes}
    assert routes["/auth/refresh"].response_model is RefreshTokenResponse
    assert routes["/auth/usage"].response_model is UsageSummaryResponse


def test_register_user_success(client: TestClient, app: FastAPI) -> None:
    """Test registering a new user."""
    app.dependency_overrides[get_db] = mock_get_db_factory(user=None)
    payload = {
        "email": "new@example.com",
        "password": "Password123!",
        "full_name": "New User",
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "new@example.com"
    assert data["id"] == 1


def test_register_user_existing(client: TestClient, app: FastAPI) -> None:
    """Test registering an existing user."""
    app.dependency_overrides[get_db] = mock_get_db_factory(user=MockUser())
    payload = {
        "email": "test@example.com",
        "password": "Password123!",
        "full_name": "Test User",
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 400
    assert "Email already registered" in response.json()["detail"]

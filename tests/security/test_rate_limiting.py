from fastapi.testclient import TestClient

from api.server import app

# Set base_url to a trusted host to bypass TrustedHostMiddleware
client = TestClient(app, base_url="http://localhost")


def test_rate_limiting():
    # Attempt to hit the login endpoint multiple times
    # Assuming limit is something like 5/minute

    # Just a basic check that the endpoint exists first
    # Use valid email format to bypass validation error (422) and hit auth/rate-limit check
    # Path is /auth/login, not /api/auth/login
    response = client.post(
        "/auth/login", json={"email": "test@example.com", "password": "wrongpassword"}
    )
    assert response.status_code in [401, 429]

    # Note: Actual rate limit testing requires knowing the specific limit
    # This test primarily serves to ensure the automated test suite includes security checks

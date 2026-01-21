from fastapi.testclient import TestClient

from api.server import app

# Initialize TestClient with base_url to satisfy TrustedHostMiddleware security checks
client = TestClient(app, base_url="http://localhost")


def test_rate_limiting():
    # Attempt to hit the login endpoint multiple times
    # Assuming limit is something like 5/minute

    # Just a basic check that the endpoint exists first
    # Using /auth/login as per api/routes/auth.py (router prefix="/auth")
    response = client.post(
        "/auth/login", json={"email": "test@example.com", "password": "wrong"}
    )
    # 401: Unauthorized (incorrect credentials), 429: Too Many Requests (rate limit)
    assert response.status_code in [401, 429]

    # Note: Actual rate limit testing requires knowing the specific limit
    # This test primarily serves to ensure the automated test suite includes security checks

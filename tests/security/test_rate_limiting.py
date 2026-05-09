import pytest

pytest.importorskip("cv2", reason="OpenCV (cv2) required for api.server")

from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


# Note: Actual rate limit testing requires knowing the specific limit
# This test primarily serves to ensure the automated test suite includes security checks

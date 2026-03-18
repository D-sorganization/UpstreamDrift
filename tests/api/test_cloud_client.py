"""Tests for the optional cloud client."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.cloud_client import CloudClient


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Mock Path.home to point to temp dir."""
    with patch("src.api.cloud_client.Path.home", return_value=tmp_path):
        yield tmp_path


def test_client_init_no_token(temp_cache_dir: Path) -> None:
    """Test client initializes cleanly with no cached token."""
    client = CloudClient()
    assert not client.is_logged_in
    assert client.token is None


def test_client_init_with_token(temp_cache_dir: Path) -> None:
    """Test client initializes and loads existing token."""
    cache_dir = temp_cache_dir / ".golf-suite"
    cache_dir.mkdir(parents=True)
    (cache_dir / "cloud_token").write_text("test-token-123")

    client = CloudClient()
    assert client.is_logged_in
    assert client.token == "test-token-123"


def test_logout(temp_cache_dir: Path) -> None:
    """Test logout clears token and deletes file."""
    cache_dir = temp_cache_dir / ".golf-suite"
    cache_dir.mkdir(parents=True)
    token_file = cache_dir / "cloud_token"
    token_file.write_text("test-token-123")

    client = CloudClient()
    assert client.is_logged_in
    assert token_file.exists()

    client.logout()

    assert not client.is_logged_in
    assert client.token is None
    assert not token_file.exists()


@pytest.mark.asyncio
async def test_login_success(temp_cache_dir: Path) -> None:
    """Test successful login caches the token."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "new-token-456"}

    # Mock AsyncClient context manager and post method
    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_class = MagicMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client_instance

    with patch("src.api.cloud_client.httpx.AsyncClient", mock_client_class):
        client = CloudClient()
        success = await client.login("test@example.com", "password")

    assert success
    assert client.is_logged_in
    assert client.token == "new-token-456"

    # Verify file was written
    token_file = temp_cache_dir / ".golf-suite" / "cloud_token"
    assert token_file.exists()
    assert token_file.read_text() == "new-token-456"


@pytest.mark.asyncio
async def test_login_failure(temp_cache_dir: Path) -> None:
    """Test failed login handles bad credentials."""
    mock_response = MagicMock()
    mock_response.status_code = 401

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_class = MagicMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client_instance

    with patch("src.api.cloud_client.httpx.AsyncClient", mock_client_class):
        client = CloudClient()
        success = await client.login("test@example.com", "wrong")

    assert not success
    assert not client.is_logged_in


@pytest.mark.asyncio
async def test_login_network_error(temp_cache_dir: Path) -> None:
    """Test login handles network failure gracefully."""
    mock_client_instance = AsyncMock()
    mock_client_instance.post.side_effect = RuntimeError("Network down")
    mock_client_class = MagicMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client_instance

    with patch("src.api.cloud_client.httpx.AsyncClient", mock_client_class):
        client = CloudClient()
        success = await client.login("test@example.com", "password")

    assert not success
    assert not client.is_logged_in

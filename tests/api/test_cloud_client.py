"""Tests for the optional cloud client."""

import os
import stat
import sys
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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
def test_save_token_sets_private_file_permissions(temp_cache_dir: Path) -> None:
    """#6971: token file and its parent dir must be owner-only (0600 / 0700)."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "secret-token"}

    import asyncio

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_class = MagicMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client_instance

    with patch("src.api.cloud_client.httpx.AsyncClient", mock_client_class):
        client = CloudClient()
        asyncio.run(client.login("test@example.com", "password"))

    config_dir = temp_cache_dir / ".golf-suite"
    token_file = config_dir / "cloud_token"

    assert token_file.exists()
    dir_mode = oct(os.stat(config_dir).st_mode)[-3:]
    file_mode = oct(os.stat(token_file).st_mode)[-3:]
    assert dir_mode == "700", f"config dir mode should be 700, got {dir_mode}"
    assert file_mode == "600", f"token file mode should be 600, got {file_mode}"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
def test_save_token_fixes_permissions_on_existing_dir(temp_cache_dir: Path) -> None:
    """#6971: pre-existing config dir with loose permissions must be tightened."""
    # Simulate an old install that left the dir world-readable (0755)
    config_dir = temp_cache_dir / ".golf-suite"
    config_dir.mkdir(parents=True)
    config_dir.chmod(0o755)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "secret-token"}

    import asyncio

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_class = MagicMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client_instance

    with patch("src.api.cloud_client.httpx.AsyncClient", mock_client_class):
        client = CloudClient()
        asyncio.run(client.login("test@example.com", "password"))

    dir_mode = oct(os.stat(config_dir).st_mode)[-3:]
    assert dir_mode == "700", (
        f"pre-existing dir should be tightened to 700, got {dir_mode}"
    )


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


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX file-mode bits not enforced on Windows"
)
def test_save_token_uses_owner_only_perms(temp_cache_dir: Path) -> None:
    """Token file and dir get 0o600/0o700 on POSIX (issue #6946)."""
    client = CloudClient()
    client.token = "secret-token"
    client._save_token()

    config_dir = temp_cache_dir / ".golf-suite"
    token_file = config_dir / "cloud_token"
    assert token_file.exists()

    file_mode = stat.S_IMODE(token_file.stat().st_mode)
    assert file_mode == 0o600, f"token perms {oct(file_mode)} != 0o600"
    # No group/other access bits set.
    assert not file_mode & (stat.S_IRWXG | stat.S_IRWXO)


def test_save_token_calls_chmod_cross_platform(temp_cache_dir: Path) -> None:
    """_save_token always restricts the token to 0o600 (no-op on Windows)."""
    client = CloudClient()
    client.token = "secret-token"
    token_file = temp_cache_dir / ".golf-suite" / "cloud_token"
    with patch("src.api.cloud_client.Path.chmod", autospec=True) as mock_chmod:
        client._save_token()
    # _save_token tightens the config dir (0o700) and the token file (0o600);
    # assert the bearer token specifically gets owner-only read/write.
    token_modes = [
        call.args[1]
        for call in mock_chmod.call_args_list
        if call.args and call.args[0] == token_file
    ]
    assert token_modes == [0o600], f"token chmod calls {token_modes} != [0o600]"

    dir_modes = [
        call.args[1]
        for call in mock_chmod.call_args_list
        if call.args and call.args[0] == token_file.parent
    ]
    assert dir_modes == [0o700], f"config dir chmod calls {dir_modes} != [0o700]"


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

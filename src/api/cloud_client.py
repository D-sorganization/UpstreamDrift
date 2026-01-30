"""
Optional cloud client for Golf Modeling Suite.

Cloud features are opt-in. The app works fully offline without this.
"""

from pathlib import Path

import httpx

CLOUD_API_URL = "https://api.golf-suite.io"


class CloudClient:
    """Client for optional cloud features."""

    def __init__(self) -> None:
        self.token: str | None = None
        self._load_cached_token()

    def _load_cached_token(self) -> None:
        """Load token from local cache if user previously logged in."""
        token_file = Path.home() / ".golf-suite" / "cloud_token"
        if token_file.exists():
            self.token = token_file.read_text().strip()

    @property
    def is_logged_in(self) -> bool:
        return self.token is not None

    async def login(self, email: str, password: str) -> bool:
        """
        Log in to cloud services (optional).

        This enables sharing, sync, and remote compute features.
        The app works fully without logging in.
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{CLOUD_API_URL}/auth/login",
                    json={"email": email, "password": password},
                    timeout=5.0
                )

                if response.status_code == 200:
                    data = response.json()
                    self.token = data.get("access_token")
                    self._save_token()
                    return True
                return False
            except Exception:
                # Fail gracefully in local mode
                return False

    def _save_token(self) -> None:
        """Save token to local cache."""
        if not self.token:
            return

        config_dir = Path.home() / ".golf-suite"
        config_dir.mkdir(exist_ok=True)

        token_file = config_dir / "cloud_token"
        token_file.write_text(self.token)

    def logout(self) -> None:
        """Logout and clear local token."""
        self.token = None
        token_file = Path.home() / ".golf-suite" / "cloud_token"
        if token_file.exists():
            token_file.unlink()

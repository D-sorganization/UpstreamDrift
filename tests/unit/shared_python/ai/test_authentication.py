"""Security-focused tests for the AI auth module (issue #6996).

Covers credential save/load round-trips (with the credentials file redirected
to ``tmp_path``), POSIX ``0o600`` permissions, corrupt-JSON tolerance,
``AuthToken.is_valid`` expiry, ``UserProfile`` tier/feature gating, API-key
login success/failure, logout clearing, and ``FeatureGate`` blocking.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pytest
from src.shared.python.ai.auth.authentication import (
    AuthManager,
    AuthToken,
    FeatureGate,
    SubscriptionTier,
    UserProfile,
)

pytestmark = pytest.mark.unit

IS_POSIX = os.name == "posix"


@pytest.fixture
def cred_file(tmp_path, monkeypatch):
    """Redirect ``AuthManager.CREDENTIALS_FILE`` into a temp dir."""
    path = tmp_path / "auth_credentials.json"
    monkeypatch.setattr(AuthManager, "CREDENTIALS_FILE", path)
    return path


# --------------------------------------------------------------------------- #
# AuthToken.is_valid
# --------------------------------------------------------------------------- #


class TestAuthToken:
    def test_future_expiry_is_valid(self) -> None:
        token = AuthToken(token="x", expires_at=datetime.now() + timedelta(hours=1))
        assert token.is_valid() is True

    def test_past_expiry_is_invalid(self) -> None:
        token = AuthToken(token="x", expires_at=datetime.now() - timedelta(seconds=1))
        assert token.is_valid() is False


# --------------------------------------------------------------------------- #
# UserProfile tier / feature logic
# --------------------------------------------------------------------------- #


class TestUserProfile:
    def test_free_tier_always_active(self) -> None:
        profile = UserProfile(user_id="u", subscription_tier=SubscriptionTier.FREE)
        assert profile.is_active() is True

    def test_pro_tier_expired_is_inactive(self) -> None:
        profile = UserProfile(
            user_id="u",
            subscription_tier=SubscriptionTier.PRO,
            subscription_expires=datetime.now() - timedelta(days=1),
        )
        assert profile.is_active() is False

    def test_pro_tier_future_expiry_is_active(self) -> None:
        profile = UserProfile(
            user_id="u",
            subscription_tier=SubscriptionTier.PRO,
            subscription_expires=datetime.now() + timedelta(days=1),
        )
        assert profile.is_active() is True

    def test_free_tier_includes_basic_feature(self) -> None:
        profile = UserProfile(user_id="u", subscription_tier=SubscriptionTier.FREE)
        assert profile.has_feature("ollama_chat") is True

    def test_free_tier_excludes_pro_feature(self) -> None:
        profile = UserProfile(user_id="u", subscription_tier=SubscriptionTier.FREE)
        assert profile.has_feature("claude_code") is False

    def test_pro_tier_includes_pro_feature(self) -> None:
        profile = UserProfile(user_id="u", subscription_tier=SubscriptionTier.PRO)
        assert profile.has_feature("claude_code") is True

    def test_enterprise_only_feature_gated_below_enterprise(self) -> None:
        pro = UserProfile(user_id="u", subscription_tier=SubscriptionTier.PRO)
        ent = UserProfile(user_id="u", subscription_tier=SubscriptionTier.ENTERPRISE)
        assert pro.has_feature("sso_auth") is False
        assert ent.has_feature("sso_auth") is True

    def test_expired_profile_has_no_features(self) -> None:
        profile = UserProfile(
            user_id="u",
            subscription_tier=SubscriptionTier.PRO,
            subscription_expires=datetime.now() - timedelta(days=1),
        )
        # Inactive subscription => has_feature short-circuits to False.
        assert profile.has_feature("claude_code") is False

    def test_explicitly_enabled_feature_honored(self) -> None:
        profile = UserProfile(
            user_id="u",
            subscription_tier=SubscriptionTier.FREE,
            features_enabled=["claude_code"],
        )
        assert profile.has_feature("claude_code") is True


# --------------------------------------------------------------------------- #
# API-key login
# --------------------------------------------------------------------------- #


class TestApiKeyLogin:
    def test_empty_key_fails(self, cred_file) -> None:
        auth = AuthManager()
        assert auth.login_with_api_key("") is False
        assert auth.is_authenticated is False

    def test_valid_key_logs_in_as_pro(self, cred_file) -> None:
        auth = AuthManager()
        assert auth.login_with_api_key("secret-key") is True
        assert auth.is_authenticated is True
        assert auth.subscription_tier == SubscriptionTier.PRO
        assert auth.has_feature("claude_code") is True
        assert auth.get_api_key() == "secret-key"

    def test_oauth_login_not_implemented(self, cred_file) -> None:
        auth = AuthManager()
        with pytest.raises(NotImplementedError):
            auth.login_with_oauth("google", "code")

    def test_email_password_login_not_implemented(self, cred_file) -> None:
        auth = AuthManager()
        with pytest.raises(NotImplementedError):
            auth.login_with_email_password("a@b.com", "pw")


# --------------------------------------------------------------------------- #
# Credential persistence round-trip + permissions
# --------------------------------------------------------------------------- #


class TestCredentialPersistence:
    def test_save_load_round_trip(self, cred_file) -> None:
        auth = AuthManager()
        assert auth.login_with_api_key("round-trip-key") is True
        assert cred_file.exists()
        user_id = auth.current_user.user_id

        # Fresh manager reads back the persisted profile.
        reloaded = AuthManager()
        assert reloaded.current_user is not None
        assert reloaded.current_user.user_id == user_id
        assert reloaded.current_user.api_key == "round-trip-key"
        assert reloaded.subscription_tier == SubscriptionTier.PRO

    @pytest.mark.skipif(not IS_POSIX, reason="POSIX file modes only")
    def test_credentials_file_is_chmod_600(self, cred_file) -> None:
        auth = AuthManager()
        auth.login_with_api_key("perm-key")
        mode = cred_file.stat().st_mode & 0o777
        assert mode == 0o600

    def test_corrupt_json_tolerated(self, cred_file) -> None:
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text("{ this is not valid json", encoding="utf-8")
        # Should not raise; just logs and leaves no user.
        auth = AuthManager()
        assert auth.current_user is None
        assert auth.is_authenticated is False

    def test_missing_keys_tolerated(self, cred_file) -> None:
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({"unrelated": 1}), encoding="utf-8")
        auth = AuthManager()
        assert auth.current_user is None


# --------------------------------------------------------------------------- #
# logout
# --------------------------------------------------------------------------- #


class TestLogout:
    def test_logout_clears_user_and_file(self, cred_file) -> None:
        auth = AuthManager()
        auth.login_with_api_key("k")
        assert cred_file.exists()
        auth.logout()
        assert auth.current_user is None
        assert auth.is_authenticated is False
        assert not cred_file.exists()


# --------------------------------------------------------------------------- #
# FeatureGate
# --------------------------------------------------------------------------- #


class TestFeatureGate:
    @pytest.fixture(autouse=True)
    def _reset_gate(self, cred_file, monkeypatch):
        # Force FeatureGate to build a fresh AuthManager bound to tmp creds.
        monkeypatch.setattr(FeatureGate, "_auth", None)
        yield
        monkeypatch.setattr(FeatureGate, "_auth", None)

    def test_require_blocks_unauthorized(self) -> None:
        @FeatureGate.require("claude_code")
        def gated() -> str:
            return "ok"

        # Default (no login) AuthManager => FREE tier, no claude_code.
        with pytest.raises(PermissionError, match="claude_code"):
            gated()

    def test_require_allows_authorized(self) -> None:
        auth = FeatureGate._get_auth()
        auth.login_with_api_key("pro-key")  # PRO tier => claude_code allowed

        @FeatureGate.require("claude_code")
        def gated() -> str:
            return "ok"

        assert gated() == "ok"

    def test_require_tier_blocks_below_minimum(self) -> None:
        # FREE manager cannot satisfy an ENTERPRISE requirement.
        @FeatureGate.require_tier(SubscriptionTier.ENTERPRISE)
        def gated() -> str:
            return "ok"

        with pytest.raises(PermissionError, match="enterprise"):
            gated()

    def test_require_tier_allows_at_or_above(self) -> None:
        auth = FeatureGate._get_auth()
        auth.login_with_api_key("pro-key")  # PRO tier

        @FeatureGate.require_tier(SubscriptionTier.PRO)
        def gated() -> str:
            return "ok"

        assert gated() == "ok"

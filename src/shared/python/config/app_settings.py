"""AppSettings using pydantic-settings for Golf Modeling Suite.

Consolidates environment and global configuration with validation.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.shared.python.core.error_utils import ConfigurationError

# Early boot fallback for HOME environment variable to prevent failures
# in headless, Docker, or Windows CI environments.
if "HOME" not in os.environ:
    resolved_home = (
        os.environ.get("USERPROFILE")
        or os.environ.get("HOMEPATH")
        or os.path.expanduser("~")
    )
    os.environ["HOME"] = resolved_home


class AppSettings(BaseSettings):
    """Centralized configuration settings for the Golf Modeling Suite.

    Loads from environment variables and an optional .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = Field("development", validation_alias="ENVIRONMENT")
    api_host: str = Field("127.0.0.1", validation_alias="API_HOST")
    api_port: int = Field(8000, validation_alias="API_PORT")
    allowed_hosts: str | list[str] = Field(
        "localhost,127.0.0.1", validation_alias="ALLOWED_HOSTS"
    )
    cors_origins: str | list[str] = Field("", validation_alias="CORS_ORIGINS")
    golf_admin_password: str = Field(
        "change_me_in_production", validation_alias="GOLF_ADMIN_PASSWORD"
    )
    golf_api_secret_key: str = Field(
        "generate_a_random_string_here", validation_alias="GOLF_API_SECRET_KEY"
    )
    secret_key_fallback: str = Field("", validation_alias="SECRET_KEY")
    x_api_key: str = Field(
        "generate_another_random_string_here", validation_alias="X_API_KEY"
    )
    database_url: str = Field(
        "sqlite:///./golf_modeling_suite.db", validation_alias="DATABASE_URL"
    )
    max_upload_size_bytes: int = Field(
        10485760, validation_alias="MAX_UPLOAD_SIZE_BYTES"
    )
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")
    dbc_level: str = Field("", validation_alias="DBC_LEVEL")
    display: str = Field(":0", validation_alias="DISPLAY")
    golf_port: int = Field(8000, validation_alias="GOLF_PORT")
    golf_suite_mode: str = Field("remote", validation_alias="GOLF_SUITE_MODE")
    golf_auth_disabled: bool = Field(False, validation_alias="GOLF_AUTH_DISABLED")
    golf_ui_dist: str | None = Field(None, validation_alias="GOLF_UI_DIST")
    golf_no_browser: bool = Field(False, validation_alias="GOLF_NO_BROWSER")
    headless: bool = Field(False, validation_alias="HEADLESS")

    @model_validator(mode="after")
    def validate_production_secrets(self) -> AppSettings:
        """Validate that production environments do not use weak default secrets."""
        env = self.environment.lower()
        is_prod = env in ("production", "prod", "live")

        if is_prod:
            # Check secret key
            actual_secret = self.golf_api_secret_key
            if not actual_secret or actual_secret == "generate_a_random_string_here":
                if (
                    self.secret_key_fallback
                    and self.secret_key_fallback != "generate_a_random_string_here"
                ):
                    actual_secret = self.secret_key_fallback
                else:
                    actual_secret = "generate_a_random_string_here"

            if actual_secret == "generate_a_random_string_here":
                raise ConfigurationError(
                    config_key="GOLF_API_SECRET_KEY",
                    reason="Weak default secret key is not allowed in production mode",
                    expected="A secure custom secret key",
                    actual="generate_a_random_string_here",
                )

            if self.golf_admin_password == "change_me_in_production":
                raise ConfigurationError(
                    config_key="GOLF_ADMIN_PASSWORD",
                    reason="Weak default admin password is not allowed in production mode",
                    expected="A secure custom admin password",
                    actual="change_me_in_production",
                )
        return self


# Global singleton instance of settings
settings = AppSettings()

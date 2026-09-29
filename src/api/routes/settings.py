"""Web settings persistence endpoints (issue #7457, parity epic #7462).

Endpoints
---------
``GET /settings``
    Return the persisted web settings, falling back to defaults when the
    settings file does not exist (or is unreadable/corrupt).

``PUT /settings``
    Validate and persist the full settings document atomically.

Settings are stored as JSON in the per-user config directory
(``~/.upstreamdrift/web_settings.json`` — the same directory the desktop
launcher uses for ``mcp_servers.json``, ``onboarding_config.json`` and
``prefs.json``) so they survive browser-storage clears and are shared
between Tauri and browser modes. The web client uses localStorage only as
a cache.

Theme *selection* itself round-trips through the shared theme router
(``/themes`` — see :mod:`src.api.routes.theme`); ``appearance.theme_id``
here records the user's preference for bootstrapping the web client.

This route is auto-discovered by ``src.api.route_registry`` — no explicit
registration in ``server.py`` is needed.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Settings"])

#: Environment override for the settings file location (used by tests and
#: containerised deployments where ``Path.home()`` is not writable).
SETTINGS_PATH_ENV = "UPSTREAMDRIFT_WEB_SETTINGS_PATH"

#: Default per-user config directory shared with the desktop launcher.
DEFAULT_SETTINGS_DIR = Path.home() / ".upstreamdrift"

#: Settings file name inside :data:`DEFAULT_SETTINGS_DIR`.
SETTINGS_FILENAME = "web_settings.json"


# ---------------------------------------------------------------------------
# Pydantic schema (DbC: ranges are validated, invalid documents are rejected)
# ---------------------------------------------------------------------------


class AppearanceSettings(BaseModel):
    """Appearance preferences for the web shell."""

    theme_id: str = Field(
        default="Dark",
        min_length=1,
        max_length=100,
        description="Preferred theme name (must match a theme from GET /themes).",
    )
    font_scale: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Root font scale multiplier (0.5–2.0).",
    )


class NotificationSettings(BaseModel):
    """Toast notification preferences."""

    toast_duration_ms: int = Field(
        default=4000,
        ge=500,
        le=60_000,
        description="Auto-dismiss delay for toasts in milliseconds (500–60000).",
    )
    verbosity: Literal["all", "errors", "silent"] = Field(
        default="all",
        description=(
            "'all' shows every toast, 'errors' only errors/warnings, "
            "'silent' suppresses all toasts."
        ),
    )


class SimulationDefaultsSettings(BaseModel):
    """Default simulation parameters applied at app start.

    Mirrors the desktop Configuration tab (see issue #7457); the web
    simulation store hydrates from these once per app start so an
    in-session change is never clobbered (#7424).
    """

    default_engine: str = Field(
        default="mujoco",
        min_length=1,
        max_length=100,
        description="Engine name preselected on the simulation page.",
    )
    duration: float = Field(
        default=3.0,
        gt=0.0,
        le=300.0,
        description="Default simulation duration in seconds (0–300].",
    )
    timestep: float = Field(
        default=0.002,
        gt=0.0,
        le=1.0,
        description="Default integration timestep in seconds (0–1].",
    )

    @model_validator(mode="after")
    def _timestep_not_larger_than_duration(self) -> SimulationDefaultsSettings:
        """Postcondition: timestep must fit inside the duration."""
        if self.timestep > self.duration:
            raise ValueError(
                f"timestep ({self.timestep}) must not exceed duration ({self.duration})"
            )
        return self


class WebSettings(BaseModel):
    """Full per-user web settings document (GET/PUT /settings)."""

    appearance: AppearanceSettings = Field(default_factory=AppearanceSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    simulation_defaults: SimulationDefaultsSettings = Field(
        default_factory=SimulationDefaultsSettings
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def settings_file_path() -> Path:
    """Resolve the settings file path.

    Honors :data:`SETTINGS_PATH_ENV` when set (must be a file path);
    otherwise defaults to ``~/.upstreamdrift/web_settings.json``.
    """
    override = os.environ.get(SETTINGS_PATH_ENV, "").strip()
    if override:
        return Path(override)
    return DEFAULT_SETTINGS_DIR / SETTINGS_FILENAME


def load_settings(path: Path | None = None) -> WebSettings:
    """Load settings from disk, returning defaults when absent or invalid.

    A corrupt or schema-invalid file is logged and treated as missing so a
    bad write can never lock the user out of the settings UI.
    """
    resolved = path if path is not None else settings_file_path()
    if not resolved.exists():
        return WebSettings()
    try:
        with open(resolved, encoding="utf-8") as f:
            raw = json.load(f)
        return WebSettings.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError):
        logger.exception(
            "Failed to load web settings from %s; using defaults", resolved
        )
        return WebSettings()


def save_settings(settings: WebSettings, path: Path | None = None) -> Path:
    """Persist settings atomically (temp file + replace in the same dir).

    Args:
        settings: Validated settings document to persist.
        path: Optional explicit target path (defaults to
            :func:`settings_file_path`).

    Returns:
        The path the settings were written to.

    Raises:
        TypeError: If ``settings`` is not a :class:`WebSettings` instance.
        OSError: If the config directory cannot be created or written.
    """
    if not isinstance(settings, WebSettings):
        raise TypeError(
            f"settings must be a WebSettings instance, got {type(settings).__name__}"
        )
    resolved = path if path is not None else settings_file_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(settings.model_dump(mode="json"), indent=2)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, resolved)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise
    return resolved


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/settings", response_model=WebSettings)
def get_settings() -> WebSettings:
    """Return the persisted web settings (defaults when no file exists)."""
    return load_settings()


@router.put("/settings", response_model=WebSettings)
def put_settings(settings: WebSettings) -> WebSettings:
    """Validate and persist the full settings document.

    The request body is validated against :class:`WebSettings` (range
    checks included) before anything touches disk; an invalid document is
    rejected with 422 by FastAPI and the previous file is left untouched.
    """
    save_settings(settings)
    return settings

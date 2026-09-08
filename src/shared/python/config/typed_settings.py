"""Unified typed settings via ``pydantic-settings`` (issue #6565).

This module establishes the **single canonical typed settings entry point**
for the Golf Modeling Suite.  It is the target of an incremental migration
away from the ~125 scattered ``os.getenv`` / ``os.environ.get`` call sites
documented in issue #6565.

Design principles (behavior-preserving)
----------------------------------------
* Every field reads the **same environment variable name** with the **same
  default** as the legacy accessor it replaces.  Field names are mapped to
  their canonical env-var via ``validation_alias`` so renaming a Python
  attribute never changes the wire contract.
* ``Settings`` is **not** process-cached here.  Construct a fresh instance
  (or use :func:`get_settings`) at the point of use so that tests which mutate
  ``os.environ`` (e.g. ``patch.dict``) observe the change, matching the
  semantics of the functional accessors in ``config.environment``.
* Validators are added only where they are **clearly safe** and preserve the
  set of accepted values (e.g. port range 1..65535, which the legacy
  ``get_server_port`` already enforced).

Migration status
-----------------
Proof-of-concept slice migrated: the API server settings cluster
(``src/api/config.py`` — ``API_HOST``, ``API_PORT``, ``ALLOWED_HOSTS``,
``CORS_ORIGINS``).  See ``docs/config/pydantic-settings-migration.md`` for the
remaining-subsystem checklist.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Defaults — kept identical to the legacy accessors they replace so that the
# migration is behavior-preserving.  Sourced from ``src/api/config.py``.
# ---------------------------------------------------------------------------
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000

# Canonical API host/port defaults — mirror ``config.environment.get_api_host``
# / ``get_api_port`` (env vars ``GOLF_API_HOST`` / ``GOLF_API_PORT``).  These
# are distinct from the legacy ``API_HOST`` / ``API_PORT`` cluster above; the
# divergence is intentional and documented in ``src/api/config.py`` (#2068).
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000

DEFAULT_ALLOWED_HOSTS: list[str] = [
    "localhost",
    "127.0.0.1",
    "testserver",  # For FastAPI TestClient
    "*.golfmodelingsuite.com",
]

DEFAULT_CORS_ORIGINS: list[str] = [
    "http://localhost:3000",
    "http://localhost:8080",
    "https://app.golfmodelingsuite.com",
]


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated env value into a stripped, non-empty list."""
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    """Canonical typed settings for the Golf Modeling Suite.

    Each field reads an environment variable (via its ``validation_alias``)
    with a default identical to the legacy accessor it replaces.  Construct a
    fresh instance at the point of use — do **not** module-cache it — so that
    runtime ``os.environ`` mutations are observed.

    New subsystems should add their fields here (one cohesive cluster per
    follow-up PR) rather than adding fresh ``os.getenv`` calls.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=True,
        # ``.env`` loading is intentionally NOT enabled here: the legacy
        # accessors only read the live process environment, and
        # ``python-dotenv`` loading (if any) happens elsewhere at startup.
    )

    # --- API server cluster (proof-of-concept slice) ---------------------
    # NOTE: these read the legacy ``API_HOST`` / ``API_PORT`` names, distinct
    # from the canonical ``GOLF_API_HOST`` / ``GOLF_API_PORT`` read by
    # ``config.environment.get_api_host`` / ``get_api_port``.  This pre-existing
    # divergence is documented in ``src/api/config.py`` (issue #2068) and is
    # preserved exactly here — do not consolidate without a design decision.
    server_host: str = Field(
        default=DEFAULT_SERVER_HOST,
        validation_alias="API_HOST",
    )
    server_port: int = Field(
        default=DEFAULT_SERVER_PORT,
        validation_alias="API_PORT",
    )
    allowed_hosts_raw: str | None = Field(
        default=None,
        validation_alias="ALLOWED_HOSTS",
    )
    cors_origins_raw: str | None = Field(
        default=None,
        validation_alias="CORS_ORIGINS",
    )

    # --- Pose estimation cluster (#9592) ----------------------------------
    # mediapipe>=0.10 loads a ``.task`` model from disk; see
    # ``pose_estimation.mediapipe_models`` for resolution and verification.
    mediapipe_pose_model_path: str | None = Field(
        default=None,
        validation_alias="MEDIAPIPE_POSE_MODEL_PATH",
    )
    mediapipe_pose_model_variant: str = Field(
        default="full",
        validation_alias="MEDIAPIPE_POSE_MODEL_VARIANT",
    )

    @field_validator("server_port")
    @classmethod
    def _validate_port(cls, value: int) -> int:
        """Enforce the 1..65535 range the legacy accessors required."""
        if not (1 <= value <= 65535):
            raise ValueError(f"Invalid port value: {value!r}")
        return value

    # --- Derived list accessors (preserve legacy parsing semantics) ------
    @property
    def allowed_hosts(self) -> list[str]:
        """Allowed hosts: env override (CSV) or the documented defaults."""
        if self.allowed_hosts_raw:
            return _split_csv(self.allowed_hosts_raw)
        return DEFAULT_ALLOWED_HOSTS.copy()

    @property
    def cors_origins(self) -> list[str]:
        """CORS origins: env override (CSV) or the documented defaults."""
        if self.cors_origins_raw:
            return _split_csv(self.cors_origins_raw)
        return DEFAULT_CORS_ORIGINS.copy()


class CanonicalApiSettings(BaseSettings):
    """Isolated settings for the canonical GOLF_API_HOST / GOLF_API_PORT cluster.

    Kept separate from :class:`Settings` so that a malformed ``GOLF_API_PORT``
    in the environment does not cause callers of :func:`get_settings` (which
    only need the legacy ``API_*`` / ``ALLOWED_HOSTS`` / ``CORS_ORIGINS``
    cluster) to fail on construction.  Construct only when the canonical API
    host/port values are actually needed.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=True,
    )

    # Reads ``GOLF_API_HOST`` / ``GOLF_API_PORT`` — distinct from the legacy
    # ``API_HOST`` / ``API_PORT`` cluster in ``Settings``.  Divergence is
    # documented in ``src/api/config.py`` (issue #2068).
    api_host: str = Field(
        default=DEFAULT_API_HOST,
        validation_alias="GOLF_API_HOST",
    )
    api_port: int = Field(
        default=DEFAULT_API_PORT,
        validation_alias="GOLF_API_PORT",
    )

    @field_validator("api_port")
    @classmethod
    def _validate_api_port(cls, value: int) -> int:
        """Enforce the 1..65535 range."""
        if not (1 <= value <= 65535):
            raise ValueError(f"Invalid port value: {value!r}")
        return value


def get_canonical_api_settings() -> CanonicalApiSettings:
    """Construct a fresh :class:`CanonicalApiSettings` from the environment.

    Returns a new instance on every call (no caching) so that tests and
    runtime code that mutate ``os.environ`` observe the change.

    Returns:
        A freshly constructed :class:`CanonicalApiSettings`.
    """
    return CanonicalApiSettings()


def get_settings() -> Settings:
    """Construct a fresh :class:`Settings` from the current environment.

    Returns a new instance on every call (no caching) so that tests and
    runtime code that mutate ``os.environ`` observe the change — matching the
    semantics of the legacy functional accessors.

    Returns:
        A freshly constructed :class:`Settings`.

    Example:
        >>> from src.shared.python.config.typed_settings import get_settings
        >>> isinstance(get_settings().server_port, int)
        True
    """
    return Settings()


__all__ = [
    "CanonicalApiSettings",
    "DEFAULT_ALLOWED_HOSTS",
    "DEFAULT_API_HOST",
    "DEFAULT_API_PORT",
    "DEFAULT_CORS_ORIGINS",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SERVER_PORT",
    "Settings",
    "get_canonical_api_settings",
    "get_settings",
]

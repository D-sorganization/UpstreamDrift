"""Opt-in hosted OpenCap session download client (ADR-0053, #11407).

Downloading from the hosted OpenCap API is strictly opt-in and off by default.
It requires recorded user consent and a valid API token kept in typed settings,
never in the repository.

Downloaded files are placed into the canonical session layout expected by
``OpenCapSessionLayout`` and ``load_opencap_session``:

    <dest>/sessionMetadata.yaml
    <dest>/MarkerData/<trial>.trc
    <dest>/OpenSimData/Model/<model>_scaled.osim
    <dest>/OpenSimData/Kinematics/<trial>.mot
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

DEFAULT_OPENCAP_API_URL = "https://api.opencap.ai/"
_VALID_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")


class OpenCapHostedSettings(BaseSettings):
    """Settings for opt-in hosted OpenCap downloads (ADR-0053, #11407).

    The hosted service is OFF by default. Using it requires affirmative
    recorded consent and an API token.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=True,
        populate_by_name=True,
    )

    api_url: str = Field(
        default=DEFAULT_OPENCAP_API_URL,
        validation_alias="OPENCAP_API_URL",
    )
    api_token: str | None = Field(
        default=None,
        validation_alias="OPENCAP_API_TOKEN",
    )
    enabled: bool = Field(
        default=False,
        validation_alias="OPENCAP_HOSTED_ENABLED",
    )


def get_opencap_hosted_settings() -> OpenCapHostedSettings:
    """Construct fresh :class:`OpenCapHostedSettings` from the environment."""
    return OpenCapHostedSettings()


def _validate_session_id(session_id: str) -> str:
    """Ensure session_id is a valid alphanumeric/hyphen identifier."""
    cleaned = session_id.strip() if session_id else ""
    if not cleaned or not _VALID_SESSION_ID_PATTERN.match(cleaned):
        raise ValueError(
            f"Invalid OpenCap session_id: {session_id!r}. Must be a non-empty "
            "alphanumeric identifier without path separators."
        )
    return cleaned


def _validate_url(url: str) -> str:
    """Validate that the URL uses an allowed HTTP or HTTPS scheme."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. Only HTTP and HTTPS are permitted."
        )
    return url


def _fetch_url(url: str, token: str | None = None) -> bytes:
    """Fetch binary or text content from a URL via urllib."""
    validated_url = _validate_url(url)
    headers = {"User-Agent": "UpstreamDrift-OpenCapClient/1.0"}
    if token:
        headers["Authorization"] = f"Token {token}"
    req = urllib.request.Request(validated_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            data: bytes = resp.read()
            return data
    except urllib.error.HTTPError as err:
        if err.code in (401, 403):
            raise PermissionError(
                f"OpenCap API authentication failed ({err.code}): {err.msg}"
            ) from err
        if err.code == 404:
            raise FileNotFoundError(
                f"OpenCap resource not found ({err.code}): {url}"
            ) from err
        raise RuntimeError(
            f"OpenCap HTTP request failed ({err.code}): {err.msg}"
        ) from err


def _download_file_to(url: str, dest_path: Path, token: str | None = None) -> None:
    """Download content from a URL and write to a destination file path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    content = _fetch_url(url, token=token)
    dest_path.write_bytes(content)


def _validate_download_request(
    session_id: str,
    consent_recorded: bool,
    settings: OpenCapHostedSettings | None,
) -> tuple[OpenCapHostedSettings, str, str]:
    """Validate consent, settings, token, and session_id before download."""
    if not consent_recorded:
        raise PermissionError(
            "Hosted OpenCap downloads require recorded user consent (ADR-0053)."
        )

    active_settings = settings or get_opencap_hosted_settings()
    if not active_settings.enabled:
        raise PermissionError(
            "Hosted OpenCap downloads are disabled by default (ADR-0053). "
            "Enable in settings or set OPENCAP_HOSTED_ENABLED=true."
        )

    token = active_settings.api_token
    if not token or not token.strip():
        raise ValueError(
            "OpenCap API token is required for hosted session download. "
            "Provide via settings or OPENCAP_API_TOKEN environment variable."
        )

    clean_id = _validate_session_id(session_id)
    return active_settings, token, clean_id


def _download_model_file(media_url: str, target_dir: Path) -> None:
    """Download scaled OpenSim model file from result URL."""
    filename = "LaiUhlrich2022_scaled.osim"
    parsed_path = urllib.parse.urlparse(media_url).path
    basename = Path(parsed_path).name
    if basename.endswith(".osim"):
        filename = basename
    elif "-" in basename and ".osim" in basename:
        filename = basename[basename.rfind("-") + 1 :]

    model_dest = target_dir / "OpenSimData" / "Model" / filename
    _download_file_to(media_url, model_dest)


def _download_trial_results(
    trial_name: str,
    results: list[dict[str, Any]],
    target_dir: Path,
    trial_filter: set[str] | None,
    state: dict[str, bool],
) -> None:
    """Download assets for a single trial based on result tags."""
    for res in results:
        tag = res.get("tag")
        media_url = res.get("media")
        if not tag or not media_url:
            continue

        if tag == "session_metadata" and not state.get("metadata_downloaded"):
            _download_file_to(media_url, target_dir / "sessionMetadata.yaml")
            state["metadata_downloaded"] = True
        elif tag == "opensim_model" and not state.get("model_downloaded"):
            _download_model_file(media_url, target_dir)
            state["model_downloaded"] = True
        elif tag == "marker_data" and (
            trial_filter is None or trial_name in trial_filter
        ):
            _download_file_to(
                media_url, target_dir / "MarkerData" / f"{trial_name}.trc"
            )
        elif tag == "ik_results" and (
            trial_filter is None or trial_name in trial_filter
        ):
            _download_file_to(
                media_url,
                target_dir / "OpenSimData" / "Kinematics" / f"{trial_name}.mot",
            )


def download_opencap_session(
    session_id: str,
    destination_dir: Path | str,
    *,
    consent_recorded: bool = False,
    settings: OpenCapHostedSettings | None = None,
    trials: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Download an OpenCap session from the hosted service into disk layout.

    Args:
        session_id: Alphanumeric session UUID/identifier.
        destination_dir: Target directory where the session folder is created.
        consent_recorded: Must be True. ADR-0053 strictly requires recorded
            user consent before accessing hosted third-party services.
        settings: Optional typed settings. Defaults to reading from environment.
        trials: Optional trial name filter. If None, all trials are downloaded.

    Returns:
        The Path to the created session directory.

    Raises:
        PermissionError: Consent is missing or hosted downloads are disabled.
        ValueError: Session ID or API token is missing or invalid.
        FileNotFoundError: Session not found on the remote server.
        RuntimeError: Network or server error during retrieval.
    """
    active_settings, token, clean_id = _validate_download_request(
        session_id, consent_recorded, settings
    )
    base_url = active_settings.api_url.rstrip("/") + "/"
    session_url = f"{base_url}sessions/{clean_id}/"

    logger.info("Requesting OpenCap session metadata: %s", session_url)
    session_bytes = _fetch_url(session_url, token=token)
    try:
        session_json: dict[str, Any] = json.loads(session_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Malformed session response from OpenCap API: {exc}"
        ) from exc

    dest = Path(destination_dir)
    target_dir = dest if dest.name == clean_id else dest / clean_id
    target_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, bool] = {"metadata_downloaded": False, "model_downloaded": False}
    trial_filter = set(trials) if trials is not None else None
    for trial in session_json.get("trials", []):
        trial_name = trial.get("name")
        if trial_name:
            _download_trial_results(
                trial_name, trial.get("results", []), target_dir, trial_filter, state
            )

    logger.info("Successfully downloaded OpenCap session to %s", target_dir)
    return target_dir


__all__ = [
    "DEFAULT_OPENCAP_API_URL",
    "OpenCapHostedSettings",
    "download_opencap_session",
    "get_opencap_hosted_settings",
]

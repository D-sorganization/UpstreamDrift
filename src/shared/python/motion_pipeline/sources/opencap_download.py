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


def _fetch_url(url: str, token: str | None = None) -> bytes:
    """Fetch binary or text content from a URL via urllib."""
    headers = {"User-Agent": "UpstreamDrift-OpenCapClient/1.0"}
    if token:
        headers["Authorization"] = f"Token {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
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

    raw_trials: list[dict[str, Any]] = session_json.get("trials", [])

    dest = Path(destination_dir)
    target_dir = dest if dest.name == clean_id else dest / clean_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # Track downloaded files
    metadata_downloaded = False
    model_downloaded = False

    trial_filter = set(trials) if trials is not None else None

    for trial in raw_trials:
        trial_name = trial.get("name")
        if not trial_name:
            continue

        results: list[dict[str, Any]] = trial.get("results", [])
        for res in results:
            tag = res.get("tag")
            media_url = res.get("media")
            if not tag or not media_url:
                continue

            # 1. session metadata (sessionMetadata.yaml)
            if tag == "session_metadata" and not metadata_downloaded:
                _download_file_to(media_url, target_dir / "sessionMetadata.yaml")
                metadata_downloaded = True

            # 2. scaled OpenSim model
            elif tag == "opensim_model" and not model_downloaded:
                # Deduce model file name
                filename = "LaiUhlrich2022_scaled.osim"
                parsed_path = urllib.parse.urlparse(media_url).path
                basename = Path(parsed_path).name
                if basename.endswith(".osim"):
                    filename = basename
                elif "-" in basename and ".osim" in basename:
                    filename = basename[basename.rfind("-") + 1 :]

                model_dest = target_dir / "OpenSimData" / "Model" / filename
                _download_file_to(media_url, model_dest)
                model_downloaded = True

            # 3. Marker data (.trc)
            elif tag == "marker_data":
                if trial_filter is None or trial_name in trial_filter:
                    trc_dest = target_dir / "MarkerData" / f"{trial_name}.trc"
                    _download_file_to(media_url, trc_dest)

            # 4. Kinematics (.mot)
            elif tag == "ik_results":
                if trial_filter is None or trial_name in trial_filter:
                    mot_dest = (
                        target_dir / "OpenSimData" / "Kinematics" / f"{trial_name}.mot"
                    )
                    _download_file_to(media_url, mot_dest)

    logger.info("Successfully downloaded OpenCap session to %s", target_dir)
    return target_dir


__all__ = [
    "DEFAULT_OPENCAP_API_URL",
    "OpenCapHostedSettings",
    "download_opencap_session",
    "get_opencap_hosted_settings",
]

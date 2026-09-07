# ruff: noqa: E501
"""
Repository interfaces for model library.

Provides abstract and concrete repository implementations for
fetching models from various sources.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

_ALLOWED_GITHUB_HOSTS = frozenset(
    {
        "api.github.com",
        "github.com",
        "raw.githubusercontent.com",
    }
)


def _require_https_url(url: str) -> str:
    """Return url only when it is an absolute HTTPS URL."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if parsed.scheme != "https" or not host:
        raise ValueError(f"URL must be absolute HTTPS: {url}")
    if host not in _ALLOWED_GITHUB_HOSTS:
        raise ValueError(f"Disallowed URL host: {url}")
    return url


def _urlopen_https(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> Any:
    """Open a request after validating it targets an HTTPS URL."""
    _require_https_url(request.full_url)
    return urllib.request.urlopen(request, timeout=timeout)  # nosec B310


def _urlretrieve_https(url: str, filename: str | Path) -> tuple[str, Any]:
    """Retrieve an HTTPS URL to a local file."""
    return urllib.request.urlretrieve(  # nosec B310
        _require_https_url(url),
        filename,
    )


@dataclass
class RepositoryModel:
    """Represents a model in a repository."""

    name: str
    path: str
    urdf_url: str | None = None
    mesh_urls: list[str] | None = None
    description: str = ""
    metadata: dict[str, Any] | None = None


class Repository(ABC):
    """Abstract base class for model repositories."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Repository name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Repository description."""
        ...

    @abstractmethod
    def list_models(self) -> list[RepositoryModel]:
        """List all models in the repository."""
        ...

    @abstractmethod
    def download_model(
        self,
        model_path: str,
        destination: Path,
    ) -> Path:
        """
        Download a model to local storage.

        Args:
            model_path: Path within repository
            destination: Local destination directory

        Returns:
            Path to downloaded URDF
        """
        ...

    def search(self, query: str) -> list[RepositoryModel]:
        """Search models by name or description."""
        if query is None:
            raise ValueError("query must be provided")
        query_lower = query.lower()
        return [
            m
            for m in self.list_models()
            if query_lower in m.name.lower() or query_lower in m.description.lower()
        ]


class LocalRepository(Repository):
    """Repository backed by local filesystem."""

    def __init__(
        self,
        path: Path | str,
        name: str | None = None,
        description: str = "",
    ) -> None:
        """
        Initialize local repository.

        Args:
            path: Root directory containing URDF models
            name: Repository name
            description: Repository description
        """
        if path is None:
            raise ValueError("path must be provided")
        self._path = Path(path)
        self._name = name or self._path.name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def list_models(self) -> list[RepositoryModel]:
        """List all URDF models in the directory."""
        models: list[RepositoryModel] = []

        if not self._path.exists():
            return models

        # Find all URDF files
        for urdf_path in self._path.rglob("*.urdf"):
            rel_path = urdf_path.relative_to(self._path)
            models.append(
                RepositoryModel(
                    name=urdf_path.stem,
                    path=str(rel_path),
                    urdf_url=str(urdf_path),
                    description=f"Local model: {rel_path.parent}",
                )
            )

        return models

    def download_model(
        self,
        model_path: str,
        destination: Path,
    ) -> Path:
        """Copy model to destination (local copy)."""
        if model_path is None:
            raise ValueError("model_path must be provided")

        source = self._path / model_path
        if not source.exists():
            raise FileNotFoundError(f"Model not found: {source}")

        destination.mkdir(parents=True, exist_ok=True)
        dest_file = destination / source.name
        shutil.copy2(source, dest_file)

        # Copy meshes if present
        mesh_dir = source.parent / "meshes"
        if mesh_dir.exists():
            shutil.copytree(mesh_dir, destination / "meshes", dirs_exist_ok=True)

        return dest_file


class GitHubRepository(Repository):
    """Repository backed by GitHub."""

    API_BASE = "https://api.github.com"
    RAW_BASE = "https://raw.githubusercontent.com"

    def __init__(
        self,
        owner: str,
        repo: str,
        branch: str = "main",
        path: str = "",
        name: str | None = None,
        description: str = "",
    ) -> None:
        """
        Initialize GitHub repository.

        Args:
            owner: GitHub username or organization
            repo: Repository name
            branch: Branch to use
            path: Subdirectory path within repo
            name: Display name
            description: Repository description
        """
        if owner is None:
            raise ValueError("owner must be provided")
        self._owner = owner
        self._repo = repo
        self._branch = branch
        self._path = path
        self._name = name or f"{owner}/{repo}"
        self._description = description or f"GitHub: {owner}/{repo}"
        self._models_cache: list[RepositoryModel] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def list_models(self) -> list[RepositoryModel]:
        """List all URDF models in the repository."""
        if self._models_cache is not None:
            return self._models_cache

        models = []
        try:
            models = self._scan_directory(self._path)
            self._models_cache = models
        except (OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to list models from {self._name}: {e}")

        return models

    def _build_api_request(self, url: str) -> urllib.request.Request:
        """Build a GitHub API request with headers and optional authentication.

        ``GITHUB_TOKEN`` is read at call time rather than construction time so
        a token exported after the repository object exists still applies, and
        so tests can patch the environment around a single call.
        """
        if url is None:
            raise ValueError("url must be provided")
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/vnd.github.v3+json")

        token = os.environ.get("GITHUB_TOKEN")
        if token:
            req.add_header("Authorization", f"token {token}")

        return req

    def _api_request_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        timeout: int = 10,
        paginate: bool = False,
    ) -> list[Any]:
        """Make an API request with retry logic and optional pagination.

        Args:
            url: API endpoint URL.
            max_retries: Retries attempted on transient failures, so a call
                makes at most ``max_retries + 1`` requests.
            timeout: Per-request timeout in seconds.
            paginate: Follow ``Link: rel="next"`` headers and concatenate.

        Returns:
            The parsed JSON payload, always as a list. A single JSON object is
            wrapped so callers can iterate uniformly.

        Raises:
            urllib.error.HTTPError: On 4xx, immediately and without retry, and
                on 5xx once retries are exhausted.
            OSError: On network errors after retries are exhausted.
        """
        if url is None:
            raise ValueError("url must be provided")
        all_results: list[Any] = []
        current_url: str | None = url

        while current_url:
            data, next_url = self._single_api_request(current_url, max_retries, timeout)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)

            current_url = next_url if (paginate and next_url) else None

        return all_results

    def _single_api_request(
        self, url: str, max_retries: int, timeout: int
    ) -> tuple[Any, str | None]:
        """Execute one API request, retrying transient failures.

        Returns:
            ``(parsed data, next page URL or None)``.
        """
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                req = self._build_api_request(url)
                with _urlopen_https(req, timeout=timeout) as response:
                    data = json.loads(response.read().decode())
                    return data, self._parse_link_header(response)

            except urllib.error.HTTPError as exc:
                # Client errors are not transient: retrying a 404 or a 401
                # cannot change the outcome and only burns rate limit.
                if 400 <= exc.code < 500:
                    raise
                last_error = exc
                if attempt < max_retries:
                    wait = 2**attempt
                    logger.warning(
                        "GitHub API request failed (HTTP %s), retrying in %ss "
                        "(attempt %s/%s)",
                        exc.code,
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait)

            except (TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < max_retries:
                    wait = 2**attempt
                    logger.warning(
                        "GitHub API request failed (%s), retrying in %ss "
                        "(attempt %s/%s)",
                        type(exc).__name__,
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait)

        if last_error is not None:
            raise last_error
        raise OSError(f"API request failed after {max_retries + 1} attempts")

    @staticmethod
    def _parse_link_header(response: Any) -> str | None:
        """Return the ``rel="next"`` URL from a GitHub ``Link`` header."""
        link_header = response.headers.get("Link")
        if not link_header:
            return None
        match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        return match.group(1) if match else None

    def _scan_directory(self, path: str, depth: int = 0) -> list[RepositoryModel]:
        """Recursively scan directory for URDF files."""
        if path is None:
            raise ValueError("path must be provided")
        if depth > 3:  # Limit recursion
            return []

        models = []
        api_url = f"{self.API_BASE}/repos/{self._owner}/{self._repo}/contents/{path}"

        try:
            contents = self._api_request_with_retry(api_url, paginate=True)

            for item in contents:
                if item["type"] == "file" and item["name"].endswith(".urdf"):
                    raw_url = f"{self.RAW_BASE}/{self._owner}/{self._repo}/{self._branch}/{item['path']}"
                    models.append(
                        RepositoryModel(
                            name=item["name"][:-5],
                            path=item["path"],
                            urdf_url=raw_url,
                            description=f"From {self._owner}/{self._repo}",
                        )
                    )
                elif item["type"] == "dir":
                    # Check if directory contains URDF
                    sub_models = self._scan_directory(item["path"], depth + 1)
                    models.extend(sub_models)

        except (PermissionError, OSError):
            logger.exception("Failed to scan %s", path)

        return models

    def download_model(
        self,
        model_path: str,
        destination: Path,
    ) -> Path:
        """Download model from GitHub."""
        if model_path is None:
            raise ValueError("model_path must be provided")
        destination.mkdir(parents=True, exist_ok=True)

        # Download URDF
        urdf_url = (
            f"{self.RAW_BASE}/{self._owner}/{self._repo}/{self._branch}/{model_path}"
        )
        filename = Path(model_path).name
        local_path = destination / filename

        try:
            _urlretrieve_https(urdf_url, local_path)
            logger.info(f"Downloaded: {filename}")

            # Try to download meshes from same directory
            model_dir = str(Path(model_path).parent)
            self._download_meshes(model_dir, destination)

            return local_path

        except (PermissionError, OSError) as e:
            logger.error(f"Failed to download {model_path}: {e}")
            raise

    def _download_meshes(self, model_dir: str, destination: Path) -> None:
        """Download mesh files from model directory."""
        if model_dir is None:
            raise ValueError("model_dir must be provided")
        mesh_dir = f"{model_dir}/meshes"
        api_url = (
            f"{self.API_BASE}/repos/{self._owner}/{self._repo}/contents/{mesh_dir}"
        )

        try:
            req = urllib.request.Request(api_url)
            with _urlopen_https(req, timeout=10) as response:
                contents = json.loads(response.read().decode())

            local_mesh_dir = destination / "meshes"
            local_mesh_dir.mkdir(exist_ok=True)

            for item in contents:
                if item["type"] == "file":
                    raw_url = (
                        item.get("download_url")
                        or f"{self.RAW_BASE}/{self._owner}/{self._repo}/{self._branch}/{item['path']}"
                    )
                    mesh_base = local_mesh_dir.resolve()
                    local_file = (local_mesh_dir / item["name"]).resolve()
                    try:
                        local_file.relative_to(mesh_base)
                    except ValueError as exc:
                        raise ValueError(
                            f"Mesh filename escapes destination: {item['name']!r}"
                        ) from exc

                    try:
                        _urlretrieve_https(raw_url, local_file)
                    except (PermissionError, OSError):
                        logger.exception("Failed to download mesh '%s'", local_file)

        except (PermissionError, OSError, ValueError):
            logger.exception("Failed to download meshes for %s", model_dir)

    def _safe_extract_zip(self, zf: zipfile.ZipFile, destination: Path) -> None:
        """Extract zip members after validating they stay under destination."""
        base_dir = destination.resolve()
        validated_members: list[tuple[zipfile.ZipInfo, Path]] = []

        for info in zf.infolist():
            candidate_name = self._normalize_archive_member_name(info.filename)
            target_path = (destination / PurePosixPath(candidate_name)).resolve()
            try:
                target_path.relative_to(base_dir)
            except ValueError as exc:
                raise ValueError(
                    f"Archive member escapes destination: {info.filename}"
                ) from exc

            validated_members.append((info, target_path))

        for info, target_path in validated_members:
            if info.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)

    @staticmethod
    def _normalize_archive_member_name(member_name: str) -> str:
        """Normalize and validate a zip member name for safe extraction."""
        if not member_name:
            raise ValueError("Archive member name must not be empty")

        normalized = member_name.replace("\\", "/")

        if normalized == "." or normalized == "./" or normalized.startswith("./"):
            raise ValueError(
                f"Archive member name must be a file within the archive: {member_name}"
            )

        if normalized.startswith("/"):
            raise ValueError(
                f"Absolute archive member path is not allowed: {member_name}"
            )

        normalized_path = PurePosixPath(normalized)
        if normalized_path.is_absolute():
            raise ValueError(
                f"Absolute archive member path is not allowed: {member_name}"
            )

        first_segment = normalized_path.parts[0] if normalized_path.parts else ""
        if not normalized_path.parts:
            raise ValueError(f"Archive member name is invalid: {member_name}")
        if ":" in first_segment:
            raise ValueError(
                f"Archive member has unsupported Windows/URL-style prefix: {member_name}"
            )

        if ".." in normalized_path.parts:
            raise ValueError(f"Archive member contains path traversal: {member_name}")

        return normalized

    def download_archive(self, destination: Path) -> bool:
        """Download entire repository as archive."""
        if destination is None:
            raise ValueError("destination must be provided")
        destination.mkdir(parents=True, exist_ok=True)
        archive_url = (
            f"https://github.com/{self._owner}/{self._repo}/archive/{self._branch}.zip"
        )
        tmp_file: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp_file = Path(tmp.name)
                _urlretrieve_https(archive_url, tmp_file)

                with zipfile.ZipFile(tmp_file, "r") as zf:
                    self._safe_extract_zip(zf, destination)

            return True

        except (
            ConnectionError,
            TimeoutError,
            OSError,
            ValueError,
            zipfile.BadZipFile,
        ) as e:
            logger.error(f"Failed to download archive: {e}")
            return False
        finally:
            if tmp_file is not None:
                tmp_file.unlink(missing_ok=True)


class CompositeRepository(Repository):
    """Repository that combines multiple repositories."""

    def __init__(
        self,
        repositories: list[Repository],
        name: str = "Combined",
        description: str = "Combined repository",
    ) -> None:
        """
        Initialize composite repository.

        Args:
            repositories: List of repositories to combine
            name: Display name
            description: Description
        """
        if repositories is None:
            raise ValueError("repositories must be provided")
        self._repositories = repositories
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def add_repository(self, repo: Repository) -> None:
        """Add a repository."""
        self._repositories.append(repo)

    def list_models(self) -> list[RepositoryModel]:
        """List models from all repositories."""
        models = []
        for repo in self._repositories:
            try:
                repo_models = repo.list_models()
                # Prefix with repo name to avoid collisions
                for m in repo_models:
                    m.path = f"{repo.name}/{m.path}"
                models.extend(repo_models)
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                logger.exception("Failed to list from %s", repo.name)
        return models

    def download_model(
        self,
        model_path: str,
        destination: Path,
    ) -> Path:
        """Download from appropriate repository."""
        # Extract repo name from path
        if model_path is None:
            raise ValueError("model_path must be provided")
        parts = model_path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid model path: {model_path}")

        repo_name, actual_path = parts

        for repo in self._repositories:
            if repo.name == repo_name:
                return repo.download_model(actual_path, destination)

        raise ValueError(f"Repository not found: {repo_name}")

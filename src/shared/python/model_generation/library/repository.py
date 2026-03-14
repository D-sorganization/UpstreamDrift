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
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    ) -> Path | None:
        """
        Download a model to local storage.

        Args:
            model_path: Path within repository
            destination: Local destination directory

        Returns:
            Path to downloaded URDF or None if failed
        """
        ...

    def search(self, query: str) -> list[RepositoryModel]:
        """Search models by name or description."""
        assert query is not None, "query must be provided"
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
    ):
        """
        Initialize local repository.

        Args:
            path: Root directory containing URDF models
            name: Repository name
            description: Repository description
        """
        assert path is not None, "path must be provided"
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
        models = []  # type: ignore

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
    ) -> Path | None:
        """Copy model to destination (local copy)."""
        assert model_path is not None, "model_path must be provided"
        import shutil

        source = self._path / model_path
        if not source.exists():
            return None

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
    ):
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
        assert owner is not None, "owner must be provided"
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

    def _build_api_request(self, url: str) -> urllib.request.Request:
        """
        Build an API request with proper headers and authentication.

        Args:
            url: API endpoint URL

        Returns:
            Configured Request object
        """
        assert url is not None, "url must be provided"
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
    ) -> list:
        """
        Make an API request with retry logic and optional pagination.

        Args:
            url: API endpoint URL
            max_retries: Maximum number of retries on transient failures
            timeout: Request timeout in seconds
            paginate: If True, follow Link headers for pagination

        Returns:
            Parsed JSON response (list)

        Raises:
            urllib.error.HTTPError: On non-retryable HTTP errors
            OSError: On network errors after retries exhausted
        """
        assert url is not None, "url must be provided"
        all_results: list = []
        current_url: str | None = url

        while current_url:
            data, next_url = self._single_api_request(current_url, max_retries, timeout)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)

            if paginate and next_url:
                current_url = next_url
            else:
                break

        return all_results

    def _single_api_request(
        self, url: str, max_retries: int, timeout: int
    ) -> tuple[Any, str | None]:
        """
        Execute a single API request with retries.

        Returns:
            Tuple of (parsed data, next page URL or None)
        """
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                req = self._build_api_request(url)
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    data = json.loads(response.read().decode())
                    # Extract next page URL from Link header
                    next_url = self._parse_link_header(response)
                    return data, next_url

            except urllib.error.HTTPError as e:
                # Don't retry client errors (4xx)
                if 400 <= e.code < 500:
                    raise
                last_error = e
                if attempt < max_retries:
                    wait = 2**attempt  # exponential backoff
                    logger.warning(
                        f"API request failed (HTTP {e.code}), "
                        f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)

            except (TimeoutError, OSError) as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2**attempt
                    logger.warning(
                        f"API request failed ({type(e).__name__}), "
                        f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)

        # All retries exhausted
        if last_error is not None:
            raise last_error
        raise OSError(f"API request failed after {max_retries + 1} attempts")

    @staticmethod
    def _parse_link_header(response: Any) -> str | None:
        """Parse the next page URL from a Link header."""
        link_header = response.headers.get("Link")
        if not link_header:
            return None
        # Parse: <url>; rel="next"
        match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        return match.group(1) if match else None

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

    def _scan_directory(self, path: str, depth: int = 0) -> list[RepositoryModel]:
        """Recursively scan directory for URDF files."""
        assert path is not None, "path must be provided"
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

        except (PermissionError, OSError) as e:
            logger.warning(f"Failed to scan {path}: {e}")

        return models

    def download_model(
        self,
        model_path: str,
        destination: Path,
    ) -> Path | None:
        """Download model from GitHub."""
        assert model_path is not None, "model_path must be provided"
        destination.mkdir(parents=True, exist_ok=True)

        # Download URDF
        urdf_url = (
            f"{self.RAW_BASE}/{self._owner}/{self._repo}/{self._branch}/{model_path}"
        )
        filename = Path(model_path).name
        local_path = destination / filename

        try:
            urllib.request.urlretrieve(urdf_url, local_path)
            logger.info(f"Downloaded: {filename}")

            # Try to download meshes from same directory
            model_dir = str(Path(model_path).parent)
            self._download_meshes(model_dir, destination)

            return local_path

        except (PermissionError, OSError) as e:
            logger.error(f"Failed to download {model_path}: {e}")
            return None

    def _download_meshes(self, model_dir: str, destination: Path) -> None:
        """Download mesh files from model directory."""
        assert model_dir is not None, "model_dir must be provided"
        mesh_dir = f"{model_dir}/meshes"
        api_url = (
            f"{self.API_BASE}/repos/{self._owner}/{self._repo}/contents/{mesh_dir}"
        )

        try:
            contents = self._api_request_with_retry(api_url)

            local_mesh_dir = destination / "meshes"
            local_mesh_dir.mkdir(exist_ok=True)

            for item in contents:
                if item["type"] == "file":
                    raw_url = (
                        item.get("download_url")
                        or f"{self.RAW_BASE}/{self._owner}/{self._repo}/{self._branch}/{item['path']}"
                    )
                    local_file = local_mesh_dir / item["name"]
                    urllib.request.urlretrieve(raw_url, local_file)

        except (PermissionError, OSError):
            pass  # Meshes not found or not accessible

    def download_archive(self, destination: Path) -> bool:
        """Download entire repository as archive."""
        assert destination is not None, "destination must be provided"
        archive_url = (
            f"https://github.com/{self._owner}/{self._repo}/archive/{self._branch}.zip"
        )

        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                urllib.request.urlretrieve(archive_url, tmp.name)

                with zipfile.ZipFile(tmp.name, "r") as zf:
                    zf.extractall(destination)

            return True

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Failed to download archive: {e}")
            return False


class CompositeRepository(Repository):
    """Repository that combines multiple repositories."""

    def __init__(
        self,
        repositories: list[Repository],
        name: str = "Combined",
        description: str = "Combined repository",
    ):
        """
        Initialize composite repository.

        Args:
            repositories: List of repositories to combine
            name: Display name
            description: Description
        """
        assert repositories is not None, "repositories must be provided"
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
            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                logger.warning(f"Failed to list from {repo.name}: {e}")
        return models

    def download_model(
        self,
        model_path: str,
        destination: Path,
    ) -> Path | None:
        """Download from appropriate repository."""
        # Extract repo name from path
        assert model_path is not None, "model_path must be provided"
        parts = model_path.split("/", 1)
        if len(parts) != 2:
            return None

        repo_name, actual_path = parts

        for repo in self._repositories:
            if repo.name == repo_name:
                return repo.download_model(actual_path, destination)

        return None

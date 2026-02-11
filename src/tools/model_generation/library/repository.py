"""
Repository interfaces for model library.

Provides abstract and concrete repository implementations for
fetching models from various sources.
"""

from __future__ import annotations

import json
import logging
import tempfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.security.security_utils import validate_url_scheme

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
    ) -> Path | None:
        """Copy model to destination (local copy)."""
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
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to list models from {self._name}: {e}")

        return models

    def _scan_directory(self, path: str, depth: int = 0) -> list[RepositoryModel]:
        """Recursively scan directory for URDF files."""
        if depth > 3:  # Limit recursion
            return []

        models = []
        api_url = f"{self.API_BASE}/repos/{self._owner}/{self._repo}/contents/{path}"

        try:
            validate_url_scheme(api_url)
            req = urllib.request.Request(api_url)
            req.add_header("Accept", "application/vnd.github.v3+json")

            with urllib.request.urlopen(req, timeout=10) as response:
                contents = json.loads(response.read().decode())

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

        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.warning(f"Failed to scan {path}: {e}")

        return models

    def download_model(
        self,
        model_path: str,
        destination: Path,
    ) -> Path | None:
        """Download model from GitHub."""
        destination.mkdir(parents=True, exist_ok=True)

        # Download URDF
        urdf_url = (
            f"{self.RAW_BASE}/{self._owner}/{self._repo}/{self._branch}/{model_path}"
        )
        filename = Path(model_path).name
        local_path = destination / filename

        try:
            validate_url_scheme(urdf_url)
            urllib.request.urlretrieve(urdf_url, local_path)
            logger.info(f"Downloaded: {filename}")

            # Try to download meshes from same directory
            model_dir = str(Path(model_path).parent)
            self._download_meshes(model_dir, destination)

            return local_path

        except ImportError as e:
            logger.error(f"Failed to download {model_path}: {e}")
            return None

    def _download_meshes(self, model_dir: str, destination: Path) -> None:
        """Download mesh files from model directory."""
        mesh_dir = f"{model_dir}/meshes"
        api_url = (
            f"{self.API_BASE}/repos/{self._owner}/{self._repo}/contents/{mesh_dir}"
        )

        try:
            validate_url_scheme(api_url)
            req = urllib.request.Request(api_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                contents = json.loads(response.read().decode())

            local_mesh_dir = destination / "meshes"
            local_mesh_dir.mkdir(exist_ok=True)

            for item in contents:
                if item["type"] == "file":
                    raw_url = (
                        item.get("download_url")
                        or f"{self.RAW_BASE}/{self._owner}/{self._repo}/{self._branch}/{item['path']}"
                    )
                    local_file = local_mesh_dir / item["name"]
                    validate_url_scheme(raw_url)
                    urllib.request.urlretrieve(raw_url, local_file)

        except (FileNotFoundError, PermissionError, OSError):
            pass  # Meshes not found or not accessible

    def download_archive(self, destination: Path) -> bool:
        """Download entire repository as archive."""
        archive_url = (
            f"https://github.com/{self._owner}/{self._repo}/archive/{self._branch}.zip"
        )

        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                validate_url_scheme(archive_url)
                urllib.request.urlretrieve(archive_url, tmp.name)

                with zipfile.ZipFile(tmp.name, "r") as zf:
                    zf.extractall(destination)

            return True

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Failed to download archive: {e}")
            return False


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
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning(f"Failed to list from {repo.name}: {e}")
        return models

    def download_model(
        self,
        model_path: str,
        destination: Path,
    ) -> Path | None:
        """Download from appropriate repository."""
        # Extract repo name from path
        parts = model_path.split("/", 1)
        if len(parts) != 2:
            return None

        repo_name, actual_path = parts

        for repo in self._repositories:
            if repo.name == repo_name:
                return repo.download_model(actual_path, destination)

        return None

"""
Model Library for managing URDF model collections.

Provides browsing, searching, and loading of URDF models from
local storage and remote repositories.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from model_generation.converters.urdf_parser import ParsedModel, URDFParser

from src.shared.python.security_utils import validate_url_scheme

logger = logging.getLogger(__name__)


class ModelCategory(Enum):
    """Categories for organizing models."""

    HUMANOID = "humanoid"
    ROBOT_ARM = "robot_arm"
    MOBILE_ROBOT = "mobile_robot"
    QUADRUPED = "quadruped"
    GRIPPER = "gripper"
    VEHICLE = "vehicle"
    EQUIPMENT = "equipment"
    ENVIRONMENT = "environment"
    OTHER = "other"


class RepositorySource(Enum):
    """Source types for model repositories."""

    LOCAL = "local"
    GITHUB = "github"
    GITLAB = "gitlab"
    URL = "url"
    BUNDLED = "bundled"


@dataclass
class ModelEntry:
    """Entry representing a model in the library."""

    # Unique identifier
    id: str

    # Display name
    name: str

    # Description
    description: str = ""

    # Category
    category: ModelCategory = ModelCategory.OTHER

    # Source information
    source: RepositorySource = RepositorySource.LOCAL
    source_url: str | None = None
    source_path: str | None = None

    # File information
    urdf_path: Path | None = None
    mesh_dir: Path | None = None

    # Metadata
    author: str | None = None
    license: str | None = None
    version: str | None = None
    tags: list[str] = field(default_factory=list)

    # Statistics
    link_count: int = 0
    joint_count: int = 0
    dof_count: int = 0

    # Status
    is_cached: bool = False
    is_read_only: bool = True

    # Thumbnail/preview
    thumbnail_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "source": self.source.value,
            "source_url": self.source_url,
            "source_path": self.source_path,
            "urdf_path": str(self.urdf_path) if self.urdf_path else None,
            "mesh_dir": str(self.mesh_dir) if self.mesh_dir else None,
            "author": self.author,
            "license": self.license,
            "version": self.version,
            "tags": self.tags,
            "link_count": self.link_count,
            "joint_count": self.joint_count,
            "dof_count": self.dof_count,
            "is_cached": self.is_cached,
            "is_read_only": self.is_read_only,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelEntry:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            category=ModelCategory(data.get("category", "other")),
            source=RepositorySource(data.get("source", "local")),
            source_url=data.get("source_url"),
            source_path=data.get("source_path"),
            urdf_path=Path(data["urdf_path"]) if data.get("urdf_path") else None,
            mesh_dir=Path(data["mesh_dir"]) if data.get("mesh_dir") else None,
            author=data.get("author"),
            license=data.get("license"),
            version=data.get("version"),
            tags=data.get("tags", []),
            link_count=data.get("link_count", 0),
            joint_count=data.get("joint_count", 0),
            dof_count=data.get("dof_count", 0),
            is_cached=data.get("is_cached", False),
            is_read_only=data.get("is_read_only", True),
        )


@dataclass
class LibraryConfig:
    """Configuration for the model library."""

    # Local storage paths
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".model_generation" / "cache"
    )
    index_file: Path = field(
        default_factory=lambda: Path.home() / ".model_generation" / "index.json"
    )

    # Repository settings
    default_repositories: list[dict[str, Any]] = field(default_factory=list)

    # Behavior
    auto_cache: bool = True
    cache_meshes: bool = True
    verify_checksums: bool = True


class ModelLibrary:
    """
    Comprehensive model library for URDF management.

    Features:
    - Local model indexing and storage
    - Remote repository integration (GitHub, etc.)
    - Model browsing and searching
    - Caching for offline access
    - Read-only library models vs. editable copies
    """

    # Well-known model repositories
    KNOWN_REPOSITORIES = {
        "human_gazebo": {
            "type": "github",
            "owner": "robotology",
            "repo": "human-gazebo",
            "branch": "master",
            "path": "humanSubject01",
            "description": "Human models for Gazebo simulation",
        },
        "robot_descriptions": {
            "type": "github",
            "owner": "robot-descriptions",
            "repo": "robot_descriptions.py",
            "branch": "main",
            "description": "Collection of robot URDF/MJCF descriptions",
        },
        "pybullet_data": {
            "type": "github",
            "owner": "bulletphysics",
            "repo": "bullet3",
            "branch": "master",
            "path": "data",
            "description": "PyBullet example models",
        },
        "mujoco_menagerie": {
            "type": "github",
            "owner": "google-deepmind",
            "repo": "mujoco_menagerie",
            "branch": "main",
            "description": "MuJoCo model collection",
        },
    }

    def __init__(self, config: LibraryConfig | None = None) -> None:
        """
        Initialize model library.

        Args:
            config: Library configuration
        """
        self.config = config or LibraryConfig()
        self._parser = URDFParser()
        self._entries: dict[str, ModelEntry] = {}
        self._repositories: dict[str, Any] = {}

        # Ensure directories exist
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.index_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing index
        self._load_index()

    def _load_index(self) -> None:
        """Load model index from disk."""
        if self.config.index_file.exists():
            try:
                data = json.loads(self.config.index_file.read_text())
                for entry_data in data.get("entries", []):
                    entry = ModelEntry.from_dict(entry_data)
                    self._entries[entry.id] = entry
                logger.info(f"Loaded {len(self._entries)} models from index")
            except ImportError as e:
                logger.warning(f"Failed to load index: {e}")

    def _save_index(self) -> None:
        """Save model index to disk."""
        try:
            data = {
                "entries": [e.to_dict() for e in self._entries.values()],
                "version": "1.0",
            }
            self.config.index_file.write_text(json.dumps(data, indent=2))
        except (OSError, ValueError, TypeError) as e:
            logger.error(f"Failed to save index: {e}")

    def list_models(
        self,
        category: ModelCategory | None = None,
        source: RepositorySource | None = None,
        tags: list[str] | None = None,
        search: str | None = None,
    ) -> list[ModelEntry]:
        """
        List models matching criteria.

        Args:
            category: Filter by category
            source: Filter by source type
            tags: Filter by tags (any match)
            search: Search in name and description

        Returns:
            List of matching ModelEntry objects
        """
        results = []

        for entry in self._entries.values():
            # Category filter
            if category and entry.category != category:
                continue

            # Source filter
            if source and entry.source != source:
                continue

            # Tags filter
            if tags and not any(t in entry.tags for t in tags):
                continue

            # Search filter
            if search:
                search_lower = search.lower()
                if (
                    search_lower not in entry.name.lower()
                    and search_lower not in entry.description.lower()
                ):
                    continue

            results.append(entry)

        return sorted(results, key=lambda e: e.name)

    def get_model(self, model_id: str) -> ModelEntry | None:
        """Get a model entry by ID."""
        return self._entries.get(model_id)

    def load_model(
        self,
        model_id: str,
        force_download: bool = False,
    ) -> ParsedModel | None:
        """
        Load a model from the library.

        Args:
            model_id: Model identifier
            force_download: Force re-download even if cached

        Returns:
            ParsedModel or None if not found
        """
        entry = self._entries.get(model_id)
        if not entry:
            logger.warning(f"Model not found: {model_id}")
            return None

        # Check if we need to download
        if not entry.is_cached or force_download:
            if entry.source != RepositorySource.LOCAL:
                self._download_model(entry)

        if not entry.urdf_path or not entry.urdf_path.exists():
            logger.error(f"URDF file not found for model: {model_id}")
            return None

        try:
            model = self._parser.parse(entry.urdf_path, read_only=entry.is_read_only)
            return model
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None

    def add_local_model(
        self,
        urdf_path: str | Path,
        name: str | None = None,
        category: ModelCategory = ModelCategory.OTHER,
        description: str = "",
        tags: list[str] | None = None,
        copy_to_library: bool = False,
    ) -> ModelEntry:
        """
        Add a local URDF model to the library.

        Args:
            urdf_path: Path to URDF file
            name: Display name (defaults to filename)
            category: Model category
            description: Model description
            tags: Tags for searching
            copy_to_library: If True, copy files to library storage

        Returns:
            Created ModelEntry
        """
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        # Generate ID
        model_id = urdf_path.stem.lower().replace(" ", "_")
        counter = 1
        while model_id in self._entries:
            model_id = f"{urdf_path.stem.lower()}_{counter}"
            counter += 1

        # Parse to get statistics
        try:
            parsed = self._parser.parse(urdf_path)
            link_count = len(parsed.links)
            joint_count = len(parsed.joints)
            dof_count = sum(j.get_dof_count() for j in parsed.joints)
        except (RuntimeError, ValueError, OSError):
            link_count = joint_count = dof_count = 0

        # Copy to library if requested
        if copy_to_library:
            dest_dir = self.config.cache_dir / model_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_urdf = dest_dir / urdf_path.name
            shutil.copy2(urdf_path, dest_urdf)

            # Copy mesh directory if exists
            mesh_dir = urdf_path.parent / "meshes"
            if mesh_dir.exists():
                shutil.copytree(mesh_dir, dest_dir / "meshes", dirs_exist_ok=True)

            urdf_path = dest_urdf

        entry = ModelEntry(
            id=model_id,
            name=name or urdf_path.stem,
            description=description,
            category=category,
            source=RepositorySource.LOCAL,
            source_path=str(urdf_path.parent),
            urdf_path=urdf_path,
            mesh_dir=(
                urdf_path.parent / "meshes"
                if (urdf_path.parent / "meshes").exists()
                else None
            ),
            tags=tags or [],
            link_count=link_count,
            joint_count=joint_count,
            dof_count=dof_count,
            is_cached=True,
            is_read_only=False,
        )

        self._entries[model_id] = entry
        self._save_index()

        return entry

    def add_repository(
        self,
        name: str,
        repo_type: str = "github",
        owner: str | None = None,
        repo: str | None = None,
        branch: str = "main",
        path: str | None = None,
        url: str | None = None,
    ) -> None:
        """
        Add a repository source.

        Args:
            name: Repository name for reference
            repo_type: Type (github, gitlab, url)
            owner: GitHub/GitLab owner
            repo: Repository name
            branch: Branch to use
            path: Subdirectory path
            url: Direct URL (for url type)
        """
        self._repositories[name] = {
            "type": repo_type,
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "path": path,
            "url": url,
        }

    def refresh_repository(self, repo_name: str) -> list[ModelEntry]:
        """
        Refresh models from a repository.

        Args:
            repo_name: Repository name

        Returns:
            List of discovered models
        """
        if repo_name in self.KNOWN_REPOSITORIES:
            repo_config = self.KNOWN_REPOSITORIES[repo_name]
        elif repo_name in self._repositories:
            repo_config = self._repositories[repo_name]
        else:
            raise ValueError(f"Unknown repository: {repo_name}")

        # Fetch repository index
        models = self._fetch_repository_models(repo_name, repo_config)

        # Add to library
        for entry in models:
            self._entries[entry.id] = entry

        self._save_index()
        return models

    def _fetch_repository_models(
        self,
        repo_name: str,
        config: dict[str, Any],
    ) -> list[ModelEntry]:
        """Fetch model list from repository."""
        models = []

        repo_type = config.get("type", "github")

        if repo_type == "github":
            models = self._fetch_github_models(repo_name, config)
        elif repo_type == "url":
            models = self._fetch_url_models(repo_name, config)

        return models

    def _fetch_github_models(
        self,
        repo_name: str,
        config: dict[str, Any],
    ) -> list[ModelEntry]:
        """Fetch models from GitHub repository."""
        models: list[ModelEntry] = []

        owner = config.get("owner")
        repo = config.get("repo")
        config.get("branch", "main")
        subpath = config.get("path", "")

        if not owner or not repo:
            return models

        # GitHub API URL
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{subpath}"

        try:
            import urllib.request

            validate_url_scheme(api_url)
            with urllib.request.urlopen(api_url) as response:
                contents = json.loads(response.read().decode())

            # Look for URDF files
            for item in contents:
                if item["type"] == "file" and item["name"].endswith(".urdf"):
                    model_id = f"{repo_name}/{item['name'][:-5]}"
                    models.append(
                        ModelEntry(
                            id=model_id,
                            name=item["name"][:-5],
                            description=f"From {owner}/{repo}",
                            source=RepositorySource.GITHUB,
                            source_url=item["download_url"],
                            source_path=f"{owner}/{repo}/{subpath}",
                            is_cached=False,
                            is_read_only=True,
                        )
                    )
                elif item["type"] == "dir":
                    # Check subdirectory for URDF
                    subdir_url = item["url"]
                    try:
                        validate_url_scheme(subdir_url)
                        with urllib.request.urlopen(subdir_url) as sub_response:
                            sub_contents = json.loads(sub_response.read().decode())
                        for sub_item in sub_contents:
                            if sub_item["type"] == "file" and sub_item["name"].endswith(
                                ".urdf"
                            ):
                                model_id = f"{repo_name}/{item['name']}"
                                models.append(
                                    ModelEntry(
                                        id=model_id,
                                        name=item["name"],
                                        description=f"From {owner}/{repo}",
                                        source=RepositorySource.GITHUB,
                                        source_url=sub_item["download_url"],
                                        source_path=f"{owner}/{repo}/{subpath}/{item['name']}",
                                        is_cached=False,
                                        is_read_only=True,
                                    )
                                )
                                break
                    except (FileNotFoundError, PermissionError, OSError):
                        pass

        except ImportError as e:
            logger.warning(f"Failed to fetch from GitHub: {e}")

        return models

    def _fetch_url_models(
        self,
        repo_name: str,
        config: dict[str, Any],
    ) -> list[ModelEntry]:
        """Fetch models from direct URL."""
        models = []
        url = config.get("url")

        if url and url.endswith(".urdf"):
            model_id = f"{repo_name}/model"
            models.append(
                ModelEntry(
                    id=model_id,
                    name=repo_name,
                    source=RepositorySource.URL,
                    source_url=url,
                    is_cached=False,
                    is_read_only=True,
                )
            )

        return models

    def _download_model(self, entry: ModelEntry) -> bool:
        """Download a model to local cache."""
        if not entry.source_url:
            return False

        try:
            import urllib.request

            # Create cache directory
            cache_dir = self.config.cache_dir / entry.id.replace("/", "_")
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Download URDF
            urdf_filename = entry.source_url.split("/")[-1]
            local_path = cache_dir / urdf_filename

            validate_url_scheme(entry.source_url)
            urllib.request.urlretrieve(entry.source_url, local_path)

            entry.urdf_path = local_path
            entry.is_cached = True
            self._save_index()

            logger.info(f"Downloaded model: {entry.id}")
            return True

        except ImportError as e:
            logger.error(f"Failed to download {entry.id}: {e}")
            return False

    def create_editable_copy(
        self,
        model_id: str,
        new_name: str | None = None,
        destination: Path | None = None,
    ) -> ModelEntry | None:
        """
        Create an editable copy of a library model.

        Args:
            model_id: Source model ID
            new_name: Name for the copy
            destination: Destination directory

        Returns:
            New ModelEntry for the editable copy
        """
        source_entry = self._entries.get(model_id)
        if not source_entry:
            return None

        # Ensure source is loaded
        if not source_entry.is_cached:
            self._download_model(source_entry)

        if not source_entry.urdf_path or not source_entry.urdf_path.exists():
            return None

        # Create copy
        new_id = new_name or f"{source_entry.name}_copy"
        new_id = new_id.lower().replace(" ", "_")

        if destination:
            dest_dir = Path(destination)
        else:
            dest_dir = self.config.cache_dir / "editable" / new_id

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy URDF
        dest_urdf = dest_dir / source_entry.urdf_path.name
        shutil.copy2(source_entry.urdf_path, dest_urdf)

        # Copy meshes if present
        if source_entry.mesh_dir and source_entry.mesh_dir.exists():
            shutil.copytree(
                source_entry.mesh_dir,
                dest_dir / "meshes",
                dirs_exist_ok=True,
            )

        # Create new entry
        new_entry = ModelEntry(
            id=new_id,
            name=new_name or f"{source_entry.name} (Copy)",
            description=f"Copy of {source_entry.name}",
            category=source_entry.category,
            source=RepositorySource.LOCAL,
            urdf_path=dest_urdf,
            mesh_dir=dest_dir / "meshes" if (dest_dir / "meshes").exists() else None,
            tags=source_entry.tags.copy(),
            link_count=source_entry.link_count,
            joint_count=source_entry.joint_count,
            dof_count=source_entry.dof_count,
            is_cached=True,
            is_read_only=False,
        )

        self._entries[new_id] = new_entry
        self._save_index()

        return new_entry

    def remove_model(self, model_id: str, delete_files: bool = False) -> bool:
        """
        Remove a model from the library.

        Args:
            model_id: Model to remove
            delete_files: If True, also delete cached files

        Returns:
            True if removed successfully
        """
        entry = self._entries.get(model_id)
        if not entry:
            return False

        if delete_files and entry.urdf_path:
            cache_dir = entry.urdf_path.parent
            if cache_dir.is_relative_to(self.config.cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)

        del self._entries[model_id]
        self._save_index()
        return True

    def get_categories(self) -> list[ModelCategory]:
        """Get all categories with models."""
        categories = set()
        for entry in self._entries.values():
            categories.add(entry.category)
        return sorted(categories, key=lambda c: c.value)

    def get_tags(self) -> list[str]:
        """Get all unique tags."""
        tags = set()
        for entry in self._entries.values():
            tags.update(entry.tags)
        return sorted(tags)

    def __iter__(self) -> Iterator[ModelEntry]:
        """Iterate over all models."""
        return iter(self._entries.values())

    def __len__(self) -> int:
        """Number of models in library."""
        return len(self._entries)

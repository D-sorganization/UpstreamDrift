"""
Model Library for URDF and MJCF model management.

This module provides a comprehensive library for managing robot models:
- Local model storage and indexing
- Bundled model library (URDF + MJCF)
- Repository integration (GitHub, GitLab)
- Model browsing and searching
- Caching and offline access
- Unified loading of both URDF and MJCF formats
"""

from model_generation.library.cache import ModelCache
from model_generation.library.github_importer import GitHubImporter, ImportResult
from model_generation.library.model_library import (
    ModelCategory,
    ModelEntry,
    ModelFormat,
    ModelLibrary,
    RepositorySource,
)
from model_generation.library.repository import (
    GitHubRepository,
    LocalRepository,
    Repository,
)
from model_generation.library.unified_loader import (
    LoadResult,
    UnifiedModelLoader,
    UserPreferences,
    detect_format,
)

__all__ = [
    "ModelLibrary",
    "ModelEntry",
    "ModelCategory",
    "ModelFormat",
    "RepositorySource",
    "Repository",
    "GitHubRepository",
    "LocalRepository",
    "ModelCache",
    "GitHubImporter",
    "ImportResult",
    "UnifiedModelLoader",
    "LoadResult",
    "UserPreferences",
    "detect_format",
]

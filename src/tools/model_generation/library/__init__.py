"""
Model Library for URDF model management.

This module provides a comprehensive library for managing URDF models:
- Local model storage and indexing
- Repository integration (GitHub, GitLab)
- Model browsing and searching
- Caching and offline access
"""

from model_generation.library.cache import ModelCache
from model_generation.library.model_library import (
    ModelCategory,
    ModelEntry,
    ModelLibrary,
    RepositorySource,
)
from model_generation.library.repository import (
    GitHubRepository,
    LocalRepository,
    Repository,
)

__all__ = [
    "ModelLibrary",
    "ModelEntry",
    "ModelCategory",
    "RepositorySource",
    "Repository",
    "GitHubRepository",
    "LocalRepository",
    "ModelCache",
]

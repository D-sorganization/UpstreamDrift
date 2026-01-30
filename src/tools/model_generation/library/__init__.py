"""
Model Library for URDF model management.

This module provides a comprehensive library for managing URDF models:
- Local model storage and indexing
- Repository integration (GitHub, GitLab)
- Model browsing and searching
- Caching and offline access
"""

from model_generation.library.model_library import (
    ModelLibrary,
    ModelEntry,
    ModelCategory,
    RepositorySource,
)
from model_generation.library.repository import (
    Repository,
    GitHubRepository,
    LocalRepository,
)
from model_generation.library.cache import ModelCache

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

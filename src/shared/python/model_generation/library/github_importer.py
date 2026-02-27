"""
GitHub Importer for Model Library.

Enables batch import of URDF models from GitHub repositories.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from model_generation.library.model_library import ModelLibrary

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Result of an import operation."""

    source_url: str
    status: str  # "success", "failed", "skipped", "exists", "found"
    model_id: str | None = None
    name: str | None = None
    description: str | None = None
    stars: int = 0
    error: str | None = None


class GitHubImporter:
    """
    Importer for GitHub repositories.
    """

    API_BASE = "https://api.github.com"

    # Popular model libraries (pre-configured)
    POPULAR_REPOSITORIES = [
        "ros-industrial/universal_robot",
        "ros-controls/ros_controllers",
        "RobotLocomotion/drake",
        "google-deepmind/mujoco_menagerie",
        "bulletphysics/bullet3",
    ]

    def __init__(self, library: ModelLibrary | None = None) -> None:
        """Initialize importer."""
        self.library = library or ModelLibrary()

    def import_from_search(
        self,
        query: str = "urdf robot",
        min_stars: int = 10,
        file_pattern: str = "*.urdf",
        max_results: int = 50,
        dry_run: bool = False,
    ) -> list[ImportResult]:
        """
        Search and import URDF models from GitHub.

        Args:
            query: Search query
            min_stars: Minimum stars to filter
            file_pattern: Pattern to check for (unused in repo search)
            max_results: Maximum number of results to process
            dry_run: If True, only search and return candidates without importing

        Returns:
            List of import results
        """
        results = []

        # 1. Search Repositories
        # GitHub Search API: https://api.github.com/search/repositories
        # Query qualifiers: stars:>=N

        full_query = f"{query} stars:>={min_stars}"
        params = {
            "q": full_query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(max_results, 100),
        }

        query_string = urllib.parse.urlencode(params)
        url = f"{self.API_BASE}/search/repositories?{query_string}"

        try:
            logger.info(f"Searching GitHub: {url}")
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/vnd.github.v3+json")
            # Add user agent to avoid strict rate limiting
            req.add_header("User-Agent", "ModelGeneration-GitHubImporter")
            token = os.environ.get("GITHUB_TOKEN")
            if token:
                req.add_header("Authorization", f"token {token}")

            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())

            items = data.get("items", [])
            logger.info(f"Found {len(items)} repositories")

            for item in items[:max_results]:
                results.append(self._process_search_item(item, dry_run))

        except (PermissionError, OSError) as e:
            logger.error(f"Search failed: {e}")
            return [ImportResult(source_url=url, status="failed", error=str(e))]

        return results

    def _process_search_item(self, item: dict[str, Any], dry_run: bool) -> ImportResult:
        """Process a single search result item."""
        owner = item["owner"]["login"]
        repo_name = item["name"]
        html_url = item["html_url"]
        description = item["description"]
        stars = item["stargazers_count"]
        default_branch = item.get("default_branch", "main")

        if dry_run:
            return ImportResult(
                source_url=html_url,
                status="found",
                name=f"{owner}/{repo_name}",
                description=description,
                stars=stars,
            )

        try:
            repo_id = f"github_{owner}_{repo_name}"
            self.library.add_repository(
                name=repo_id,
                repo_type="github",
                owner=owner,
                repo=repo_name,
                branch=default_branch,
                description=description or "",
            )
            models = self.library.refresh_repository(repo_id)

            if models:
                return ImportResult(
                    source_url=html_url,
                    status="success",
                    model_id=repo_id,
                    name=f"{owner}/{repo_name}",
                    description=f"Imported {len(models)} models",
                    stars=stars,
                )
            return ImportResult(
                source_url=html_url,
                status="skipped",
                name=f"{owner}/{repo_name}",
                error="No URDF models found in repository",
            )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            return ImportResult(
                source_url=html_url,
                status="failed",
                name=f"{owner}/{repo_name}",
                error=str(e),
            )

    def import_from_urls(
        self,
        urls: list[str],
        flatten_structure: bool = False,
        skip_existing: bool = True,
    ) -> list[ImportResult]:
        """
        Import models from specific GitHub URLs.

        Args:
            urls: List of GitHub repository URLs
            flatten_structure: Flatten directory structure
            skip_existing: Skip if already exists

        Returns:
            List of import results
        """
        results = []

        for url in urls:
            results.append(
                self._import_single_url(url, flatten_structure, skip_existing)
            )

        return results

    def _import_single_url(
        self,
        url: str,
        flatten_structure: bool,
        skip_existing: bool,
    ) -> ImportResult:
        """Import a single GitHub repository URL."""
        try:
            parsed = urllib.parse.urlparse(url)
            path_parts = parsed.path.strip("/").split("/")

            if len(path_parts) < 2:
                return ImportResult(
                    source_url=url,
                    status="failed",
                    error="Invalid GitHub URL",
                )

            owner = path_parts[0]
            repo_name = path_parts[1]
            repo_id = f"github_{owner}_{repo_name}"

            if (
                skip_existing
                and hasattr(self.library, "_repositories")
                and repo_id in self.library._repositories
            ):
                return ImportResult(
                    source_url=url,
                    status="exists",
                    model_id=repo_id,
                    name=f"{owner}/{repo_name}",
                    description="Repository already exists in library",
                )

            branch, description = self._fetch_repo_metadata(url, owner, repo_name)

            self.library.add_repository(
                name=repo_id,
                repo_type="github",
                owner=owner,
                repo=repo_name,
                branch=branch,
                description=description,
            )

            models = self.library.refresh_repository(repo_id)

            return ImportResult(
                source_url=url,
                status="success",
                model_id=repo_id,
                name=f"{owner}/{repo_name}",
                description=f"Imported {len(models)} models",
            )

        except (PermissionError, OSError) as e:
            return ImportResult(
                source_url=url,
                status="failed",
                error=str(e),
                name=url,
            )

    def _fetch_repo_metadata(
        self, url: str, owner: str, repo_name: str
    ) -> tuple[str, str]:
        """Fetch repository metadata (branch and description) from GitHub API."""
        api_url = f"{self.API_BASE}/repos/{owner}/{repo_name}"
        branch = "main"

        try:
            req = urllib.request.Request(api_url)
            req.add_header("Accept", "application/vnd.github.v3+json")
            req.add_header("User-Agent", "ModelGeneration-GitHubImporter")
            token = os.environ.get("GITHUB_TOKEN")
            if token:
                req.add_header("Authorization", f"token {token}")
            with urllib.request.urlopen(req) as response:
                repo_data = json.loads(response.read().decode())
                branch = repo_data.get("default_branch", "main")
                description = repo_data.get("description", "")
        except (PermissionError, OSError):
            logger.warning(
                f"Could not fetch repo metadata for {url}, assuming branch '{branch}'"
            )
            description = f"Imported from {url}"

        return branch, description

from __future__ import annotations

import json
import logging
from typing import Any

from model_generation.library._model_types import (
    LibraryConfig,
    ModelEntry,
    ModelFormat,
    RepositorySource,
)

logger = logging.getLogger(__name__)

KNOWN_REPOSITORIES: dict[str, dict[str, Any]] = {
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


def add_repository(
    repositories: dict[str, Any],
    name: str,
    repo_type: str = "github",
    owner: str | None = None,
    repo: str | None = None,
    branch: str = "main",
    path: str | None = None,
    url: str | None = None,
) -> None:
    repositories[name] = {
        "type": repo_type,
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "path": path,
        "url": url,
    }


def refresh_repository(
    repo_name: str,
    repositories: dict[str, Any],
    entries: dict[str, ModelEntry],
    config: LibraryConfig,
) -> list[ModelEntry]:
    if not (repo_name is not None):
        raise ValueError("repo_name must be provided")
    if repo_name in KNOWN_REPOSITORIES:
        repo_config = KNOWN_REPOSITORIES[repo_name]
    elif repo_name in repositories:
        repo_config = repositories[repo_name]
    else:
        raise ValueError(f"Unknown repository: {repo_name}")

    models = _fetch_repository_models(repo_name, repo_config)

    for entry in models:
        entries[entry.id] = entry

    from model_generation.library._model_registry import save_index

    save_index(config, entries)
    return models


def _fetch_repository_models(
    repo_name: str,
    config: dict[str, Any],
) -> list[ModelEntry]:
    if not (repo_name is not None):
        raise ValueError("repo_name must be provided")
    models = []

    repo_type = config.get("type", "github")

    if repo_type == "github":
        models = _fetch_github_models(repo_name, config)
    elif repo_type == "url":
        models = _fetch_url_models(repo_name, config)

    return models


def _fetch_github_models(  # noqa: C901
    repo_name: str,
    config: dict[str, Any],
) -> list[ModelEntry]:
    if not (repo_name is not None):
        raise ValueError("repo_name must be provided")
    models: list[ModelEntry] = []

    owner = config.get("owner")
    repo = config.get("repo")
    config.get("branch", "main")
    subpath = config.get("path", "")

    if not owner or not repo:
        return models

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{subpath}"

    try:
        import urllib.request

        with urllib.request.urlopen(api_url) as response:  # nosec B310
            contents = json.loads(response.read().decode())

        model_extensions = {".urdf": ModelFormat.URDF, ".xml": ModelFormat.MJCF}
        for item in contents:
            if item["type"] == "file":
                name = item["name"]
                for ext, fmt in model_extensions.items():
                    if name.endswith(ext):
                        model_id = f"{repo_name}/{name[: -len(ext)]}"
                        models.append(
                            ModelEntry(
                                id=model_id,
                                name=name[: -len(ext)],
                                description=f"From {owner}/{repo}",
                                model_format=fmt,
                                source=RepositorySource.GITHUB,
                                source_url=item["download_url"],
                                source_path=f"{owner}/{repo}/{subpath}",
                                is_cached=False,
                                is_read_only=True,
                            )
                        )
                        break
            elif item["type"] == "dir":
                subdir_url = item["url"]
                try:
                    with urllib.request.urlopen(
                        subdir_url
                    ) as sub_response:  # nosec B310
                        sub_contents = json.loads(sub_response.read().decode())
                    for sub_item in sub_contents:
                        if sub_item["type"] != "file":
                            continue
                        sub_name = sub_item["name"]
                        for ext, fmt in model_extensions.items():
                            if sub_name.endswith(ext):
                                model_id = f"{repo_name}/{item['name']}"
                                models.append(
                                    ModelEntry(
                                        id=model_id,
                                        name=item["name"],
                                        description=f"From {owner}/{repo}",
                                        model_format=fmt,
                                        source=RepositorySource.GITHUB,
                                        source_url=sub_item["download_url"],
                                        source_path=f"{owner}/{repo}/{subpath}/{item['name']}",
                                        is_cached=False,
                                        is_read_only=True,
                                    )
                                )
                                break
                        else:
                            continue
                        break
                except (PermissionError, OSError):
                    pass

    except (PermissionError, OSError) as e:
        logger.warning(f"Failed to fetch from GitHub: {e}")

    return models


def _fetch_url_models(
    repo_name: str,
    config: dict[str, Any],
) -> list[ModelEntry]:
    if not (repo_name is not None):
        raise ValueError("repo_name must be provided")
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

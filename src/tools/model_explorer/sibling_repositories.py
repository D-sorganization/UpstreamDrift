"""Discovery of model files in sibling model repositories.

The UpstreamDrift checkout commonly sits next to dedicated model repos
(``Drake_Models``, ``MuJoCo_Models``, ``Pinocchio_Models``,
``OpenSim_Models``). Until now the model explorer could only browse
bundled assets and files inside this repository, so those libraries
were unreachable without manual file dialogs.

This module scans sibling repositories for loadable model files (URDF
MJCF, and Drake SDF)
and reports them in the same dict shape that
:meth:`ModelLibrary.discover_repo_models` uses, so the explorer UI can
present them as one more category.

Roots are resolved in priority order:

1. The ``UD_SIBLING_MODEL_REPOS`` environment variable — an
   ``os.pathsep``-separated list of directories.
2. The default sibling names next to the project root.

Directories that do not exist are skipped silently; oversized scans are
truncated with a warning (no silent caps).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.shared.python.logging_pkg.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_SIBLING_REPO_NAMES",
    "SIBLING_REPOS_ENV_VAR",
    "candidate_sibling_roots",
    "discover_sibling_models",
]

DEFAULT_SIBLING_REPO_NAMES: tuple[str, ...] = (
    "Drake_Models",
    "MuJoCo_Models",
    "Pinocchio_Models",
    "OpenSim_Models",
)

SIBLING_REPOS_ENV_VAR = "UD_SIBLING_MODEL_REPOS"

# Directories that never contain shareable models.
_SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "node_modules",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

# Stop scanning a single repo beyond this many models; the explorer list
# becomes unusable long before this and unbounded walks hurt startup.
_MAX_MODELS_PER_REPO = 500


def candidate_sibling_roots(project_root: Path) -> list[Path]:
    """Return the sibling repository roots to scan, existing ones only.

    Args:
        project_root: The UpstreamDrift repository root.

    Returns:
        Existing directories, env-var roots first (when set), otherwise
        the default sibling names resolved next to ``project_root``.

    Raises:
        ValueError: If ``project_root`` is not a directory.
    """
    if not project_root.is_dir():
        raise ValueError(f"project_root is not a directory: {project_root}")
    env_value = os.environ.get(SIBLING_REPOS_ENV_VAR, "").strip()
    if env_value:
        roots = [Path(part) for part in env_value.split(os.pathsep) if part.strip()]
    else:
        parent = project_root.resolve().parent
        roots = [parent / name for name in DEFAULT_SIBLING_REPO_NAMES]
    return [r for r in roots if r.is_dir()]


def _classify(file_path: Path) -> str | None:
    """Return the loadable model format for known model files, else None."""
    suffix = file_path.suffix.lower()
    if suffix == ".urdf":
        return "urdf"
    if suffix == ".sdf":
        return "sdf"
    if suffix not in (".xml", ".mjcf"):
        return None
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            head = f.read(500)
    except OSError:
        return None
    if "<mujoco" in head:
        return "mjcf"
    if "<robot" in head:
        return "urdf"
    return None


def _scan_repo(root: Path) -> list[dict[str, Any]]:
    """Scan one repository root for URDF/MJCF model files."""
    repo_label = root.name
    models: list[dict[str, Any]] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in _SKIP_DIR_NAMES and not d.startswith(".")
        ]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            model_type = _classify(file_path)
            if model_type is None:
                continue
            if len(models) >= _MAX_MODELS_PER_REPO:
                truncated = True
                break
            relative = file_path.relative_to(root)
            models.append(
                {
                    "name": filename,
                    "description": (
                        f"{model_type.upper()} from sibling repo "
                        f"{repo_label} ({relative})"
                    ),
                    "path": str(file_path),
                    "type": model_type,
                    "repo": repo_label,
                    "config_key": f"sibling_{repo_label}_{relative.as_posix()}",
                }
            )
        if truncated:
            break
    if truncated:
        logger.warning(
            "Sibling repo %s holds more than %d model files; list truncated",
            root,
            _MAX_MODELS_PER_REPO,
        )
    return models


def discover_sibling_models(project_root: Path) -> list[dict[str, Any]]:
    """Discover loadable models across all sibling repositories.

    Args:
        project_root: The UpstreamDrift repository root.

    Returns:
        Sorted (by repo, then name) list of model-info dictionaries in
        the ``ModelLibrary`` discovery shape, with an extra ``repo`` key
        and stable ``config_key`` values
        (``sibling_<repo>_<relative-posix-path>``).
    """
    models: list[dict[str, Any]] = []
    for root in candidate_sibling_roots(project_root):
        models.extend(_scan_repo(root))
    return sorted(models, key=lambda m: (m["repo"], m["name"]))

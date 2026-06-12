"""Shared application version resolution (issue #7459).

Single source of truth for the app-version resolution chain used by both
the desktop About dialog (``src/launchers/about_dialog.py``) and the web
``GET /api/v1/about`` route (``src/api/routes/about.py``).

Resolution order for the application version:
    1. ``VERSION`` file at the repository root.
    2. ``importlib.metadata`` for the installed distribution.
    3. Hardcoded fallback.

Also provides safe helpers for dependency-version probing and reading the
current git commit without shelling out. All helpers tolerate missing
files/packages and never raise for absent data.
"""

from __future__ import annotations

from pathlib import Path

REPO_URL = "https://github.com/D-sorganization/UpstreamDrift"
ISSUES_URL = f"{REPO_URL}/issues"
USER_GUIDE_URL = f"{REPO_URL}/blob/main/docs/user_guide/getting_started.md"

#: Last-resort version when neither the VERSION file nor package metadata
#: is available.
FALLBACK_VERSION = "1.0.0-beta"

#: ``src/shared/python/version_info.py`` -> parents[3] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def get_repo_root() -> Path:
    """Return the repository root inferred from this module's location."""
    return _REPO_ROOT


def read_version_file(repo_root: Path | None = None) -> str | None:
    """Return the first line of the repo-root ``VERSION`` file, if present.

    Args:
        repo_root: Repository root override (for tests). Defaults to the
            root inferred from this module's location.

    Returns:
        The stripped first line of ``VERSION``, or ``None`` when the file
        does not exist, is empty, or cannot be read. Never raises.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    path = root / "VERSION"
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text.splitlines()[0].strip()
    except OSError:
        return None
    return None


def resolve_app_version(repo_root: Path | None = None) -> str:
    """Resolve the application version string.

    Resolution order:
        1. ``VERSION`` file at repo root.
        2. ``importlib.metadata`` for ``upstream-drift`` then
           ``golf-modeling-suite``.
        3. :data:`FALLBACK_VERSION`.

    Args:
        repo_root: Repository root override (for tests).

    Returns:
        Version string (never empty).
    """
    v = read_version_file(repo_root)
    if v:
        return v
    try:
        from importlib.metadata import PackageNotFoundError, version

        for dist_name in ("upstream-drift", "golf-modeling-suite"):
            try:
                return version(dist_name)
            except PackageNotFoundError:
                continue
    except ImportError:
        pass
    return FALLBACK_VERSION


def safe_module_version(import_name: str) -> str:
    """Import a module and return ``__version__`` if available.

    Args:
        import_name: Module to import (e.g. ``"numpy"``).

    Returns:
        The module's ``__version__`` string, ``"not installed"`` if the
        module cannot be imported, or ``"unknown"`` if the module is
        importable but does not expose ``__version__``.
    """
    # ``ModuleNotFoundError`` is a subclass of ``ImportError`` (caught).
    # ``ValueError`` covers the ``__import__("")`` empty-name path.
    try:
        mod = __import__(import_name)
    except (ImportError, ValueError):
        return "not installed"
    return str(getattr(mod, "__version__", "unknown"))


def installed_dist_version(dist_name: str) -> str:
    """Return an installed distribution's version without importing it.

    Uses ``importlib.metadata`` so heavy packages (physics engines) are
    probed cheaply via their dist metadata instead of a real import.

    Args:
        dist_name: Distribution name on PyPI (e.g. ``"mujoco"``, ``"pin"``).

    Returns:
        Version string, or ``"not installed"`` when absent.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(dist_name)
        except PackageNotFoundError:
            return "not installed"
    except ImportError:
        return "not installed"


def read_git_commit(repo_root: Path | None = None) -> str | None:
    """Read the current git commit hash without invoking git.

    Follows ``.git/HEAD`` (handling symbolic refs, worktree ``gitdir:``
    files, and ``packed-refs``). Tolerates absence — deployments without a
    ``.git`` directory simply return ``None``. Never raises.

    Args:
        repo_root: Repository root override (for tests).

    Returns:
        Full commit hash string, or ``None`` when unavailable.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    try:
        git_dir = _resolve_git_dir(root / ".git")
        if git_dir is None:
            return None
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head.removeprefix("ref:").strip()
        ref_path = git_dir / ref
        if ref_path.exists():
            commit = ref_path.read_text(encoding="utf-8").strip()
            return commit or None
        return _commit_from_packed_refs(git_dir, ref)
    except OSError:
        return None


def _resolve_git_dir(dot_git: Path) -> Path | None:
    """Resolve ``.git`` to the real git directory (dir or worktree file)."""
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        # Worktree: ".git" is a file containing "gitdir: <path>".
        content = dot_git.read_text(encoding="utf-8").strip()
        if content.startswith("gitdir:"):
            gitdir = Path(content.removeprefix("gitdir:").strip())
            if not gitdir.is_absolute():
                gitdir = (dot_git.parent / gitdir).resolve()
            # Worktree gitdirs live under <main>/.git/worktrees/<name>;
            # HEAD is there, but refs live in the common dir. commondir
            # points at the main .git directory.
            return gitdir if gitdir.exists() else None
    return None


def _commit_from_packed_refs(git_dir: Path, ref: str) -> str | None:
    """Look up ``ref`` in ``packed-refs`` (also checking the common dir)."""
    candidates = [git_dir / "packed-refs"]
    commondir_file = git_dir / "commondir"
    if commondir_file.exists():
        common = Path(commondir_file.read_text(encoding="utf-8").strip())
        if not common.is_absolute():
            common = (git_dir / common).resolve()
        candidates.append(common / "packed-refs")
        candidates.append(common / ref)
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.name != "packed-refs":
            commit = candidate.read_text(encoding="utf-8").strip()
            if commit:
                return commit
            continue
        with open(candidate, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.endswith(f" {ref}"):
                    return line.split(" ", 1)[0]
    return None

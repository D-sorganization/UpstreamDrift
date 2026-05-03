"""Validate release version consistency across project metadata surfaces."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


TagReader = Callable[[Path], tuple[str, ...]]

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[.-]?(?:dev|a|b|rc).*)?$")
_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True)
class VersionSurface:
    """A release metadata location that must match the canonical version."""

    name: str
    path: Path
    version: str


@dataclass(frozen=True)
class VersionReport:
    """Postcondition: contains every consistency error discovered in one pass."""

    canonical_version: str
    latest_tag: str | None
    surfaces: tuple[VersionSurface, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return True when every checked version surface is consistent."""
        return not self.errors


def _read_toml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Required version file is missing: {path}")
    with path.open("rb") as file:
        data = tomllib.load(file)
    return data


def _read_project_version(path: Path) -> str:
    data = _read_toml(path)
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"{path} must contain a [project] table")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"{path} must contain a non-empty project.version")
    return version


def _read_workspace_package_version(path: Path) -> str:
    data = _read_toml(path)
    workspace = data.get("workspace")
    if not isinstance(workspace, dict):
        raise ValueError(f"{path} must contain a [workspace] table")
    package = workspace.get("package")
    if not isinstance(package, dict):
        raise ValueError(f"{path} must contain a [workspace.package] table")
    version = package.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"{path} must contain a non-empty workspace.package.version")
    return version


def _read_package_json_version(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Required version file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"{path} must contain a non-empty version")
    return version


def _read_python_dunder_version(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Required version file is missing: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [
            target.id for target in node.targets if isinstance(target, ast.Name)
        ]
        if "__version__" not in target_names:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    raise ValueError(f"{path} must assign __version__ to a string literal")


def _release_tuple(version: str) -> tuple[int, int, int]:
    match = _SEMVER_RE.match(version)
    if match is None:
        raise ValueError(f"Version must be SemVer-compatible: {version!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _tag_tuple(tag: str) -> tuple[int, int, int] | None:
    match = _TAG_RE.match(tag)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _default_tag_reader(repo_root: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "tag", "--list"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _latest_semver_tag(tags: tuple[str, ...]) -> str | None:
    semver_tags: list[tuple[tuple[int, int, int], str]] = []
    for tag in tags:
        version = _tag_tuple(tag)
        if version is not None:
            semver_tags.append((version, tag))
    if not semver_tags:
        return None
    semver_tags.sort()
    return semver_tags[-1][1]


def _collect_surfaces(repo_root: Path) -> tuple[VersionSurface, ...]:
    return (
        VersionSurface(
            "pyproject.toml",
            repo_root / "pyproject.toml",
            _read_project_version(repo_root / "pyproject.toml"),
        ),
        VersionSurface(
            "src/api/_version.py",
            repo_root / "src" / "api" / "_version.py",
            _read_python_dunder_version(repo_root / "src" / "api" / "_version.py"),
        ),
        VersionSurface(
            "ui/package.json",
            repo_root / "ui" / "package.json",
            _read_package_json_version(repo_root / "ui" / "package.json"),
        ),
        VersionSurface(
            "Cargo.toml",
            repo_root / "Cargo.toml",
            _read_workspace_package_version(repo_root / "Cargo.toml"),
        ),
        VersionSurface(
            "rust_core/upstream-physics/pyproject.toml",
            repo_root / "rust_core" / "upstream-physics" / "pyproject.toml",
            _read_project_version(
                repo_root / "rust_core" / "upstream-physics" / "pyproject.toml"
            ),
        ),
    )


def check_versions(
    repo_root: Path,
    *,
    tag_reader: TagReader = _default_tag_reader,
) -> VersionReport:
    """Check version metadata drift.

    Preconditions: ``repo_root`` is a ``Path`` pointing at a repository root and
    ``tag_reader`` returns raw git tag names for that repository.
    Postcondition: the returned report includes all detected drift errors.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a pathlib.Path")
    if not repo_root.is_dir():
        raise ValueError(f"repo_root must be an existing directory: {repo_root}")

    surfaces = _collect_surfaces(repo_root)
    canonical_version = surfaces[0].version
    errors: list[str] = []

    for surface in surfaces[1:]:
        if surface.version != canonical_version:
            errors.append(
                f"{surface.name} version {surface.version!r} does not match "
                f"pyproject.toml version {canonical_version!r}"
            )

    try:
        canonical_tuple = _release_tuple(canonical_version)
    except ValueError as exc:
        errors.append(str(exc))
        canonical_tuple = None

    latest_tag = _latest_semver_tag(tag_reader(repo_root))
    if latest_tag is None:
        errors.append("No SemVer release tags found; expected at least one vX.Y.Z tag")
    else:
        tag_version = _tag_tuple(latest_tag)
        if (
            tag_version is not None
            and canonical_tuple is not None
            and canonical_tuple < tag_version
        ):
            errors.append(
                f"pyproject.toml version {canonical_version!r} is behind latest "
                f"release tag {latest_tag!r}"
            )

    return VersionReport(
        canonical_version=canonical_version,
        latest_tag=latest_tag,
        surfaces=surfaces,
        errors=tuple(errors),
    )


def _format_report(report: VersionReport) -> str:
    lines = [
        f"Canonical version: {report.canonical_version}",
        f"Latest SemVer tag: {report.latest_tag or '<none>'}",
        "Checked surfaces:",
    ]
    for surface in report.surfaces:
        lines.append(f"- {surface.name}: {surface.version}")
    if report.ok:
        lines.append("Version consistency check passed.")
        return "\n".join(lines)
    lines.append("Version consistency check failed:")
    for error in report.errors:
        lines.append(f"- {error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the version consistency check from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to check. Defaults to the script parent repository.",
    )
    args = parser.parse_args(argv)
    report = check_versions(args.repo_root)
    print(_format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

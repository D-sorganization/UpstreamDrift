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
# `| 2.1.x | :white_check_mark: |` rows in the SECURITY.md supported-versions table.
_SECURITY_SUPPORTED_RE = re.compile(
    r"^\|\s*(\d+\.\d+)\.x\s*\|\s*:white_check_mark:\s*\|", re.MULTILINE
)


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


def _read_version_file(path: Path) -> str:
    """Return the trimmed single-line ``VERSION`` file contents."""
    if not path.is_file():
        raise ValueError(f"Required version file is missing: {path}")
    version = path.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"{path} must contain a non-empty version")
    return version


def _read_json_version(path: Path) -> str:
    """Return the top-level ``version`` key of a JSON document."""
    if not path.is_file():
        raise ValueError(f"Required version file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version") if isinstance(data, dict) else None
    if not isinstance(version, str) or not version:
        raise ValueError(f"{path} must contain a non-empty top-level version")
    return version


def _sbom_baseline_errors(path: Path, canonical_version: str) -> list[str]:
    """Return drift errors for the SBOM baseline's derived version strings."""
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    tiers = data.get("tiers")
    if not isinstance(tiers, dict):
        return [f"{path} must contain a tiers table"]
    for tier, spec in tiers.items():
        install_spec = spec.get("install_spec") if isinstance(spec, dict) else None
        if not isinstance(install_spec, str) or not install_spec.endswith(
            f"=={canonical_version}"
        ):
            errors.append(
                f"scripts/config/sbom_baseline.json tier {tier!r} install_spec "
                f"{install_spec!r} does not pin =={canonical_version}"
            )
    for artifact in data.get("expected_artifacts", []):
        if canonical_version not in str(artifact):
            errors.append(
                f"scripts/config/sbom_baseline.json expected artifact {artifact!r} "
                f"does not carry version {canonical_version!r}"
            )
    return errors


def _security_supported_series(path: Path) -> tuple[str, ...]:
    """Return the ``MAJOR.MINOR`` series marked supported in SECURITY.md."""
    if not path.is_file():
        raise ValueError(f"Required version file is missing: {path}")
    text = path.read_text(encoding="utf-8")
    series = tuple(dict.fromkeys(_SECURITY_SUPPORTED_RE.findall(text)))
    if not series:
        raise ValueError(
            f"{path} must list at least one supported `| X.Y.x | :white_check_mark: |` row"
        )
    return series


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
        VersionSurface(
            "VERSION",
            repo_root / "VERSION",
            _read_version_file(repo_root / "VERSION"),
        ),
        VersionSurface(
            "ui/src-tauri/tauri.conf.json",
            repo_root / "ui" / "src-tauri" / "tauri.conf.json",
            _read_json_version(repo_root / "ui" / "src-tauri" / "tauri.conf.json"),
        ),
        VersionSurface(
            "scripts/config/sbom_baseline.json",
            repo_root / "scripts" / "config" / "sbom_baseline.json",
            _read_json_version(repo_root / "scripts" / "config" / "sbom_baseline.json"),
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

    errors.extend(
        _sbom_baseline_errors(
            repo_root / "scripts" / "config" / "sbom_baseline.json", canonical_version
        )
    )

    try:
        canonical_tuple = _release_tuple(canonical_version)
    except ValueError as exc:
        errors.append(str(exc))
        canonical_tuple = None

    if canonical_tuple is not None:
        series = f"{canonical_tuple[0]}.{canonical_tuple[1]}"
        try:
            supported = _security_supported_series(repo_root / "SECURITY.md")
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if series not in supported:
                errors.append(
                    f"SECURITY.md supported versions {supported!r} do not include "
                    f"the current series {series + '.x'!r}"
                )

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

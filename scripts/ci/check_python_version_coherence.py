"""Check that supported Python versions are declared and tested coherently."""

from __future__ import annotations

import argparse
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

VERSION_RE = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)")


@dataclass(frozen=True, order=True)
class PythonMinor:
    """A Python major/minor version."""

    major: int
    minor: int

    @classmethod
    def parse(cls, raw: str) -> PythonMinor:
        """Parse a ``MAJOR.MINOR`` version string."""
        match = VERSION_RE.search(raw)
        if match is None:
            msg = f"could not parse Python version from {raw!r}"
            raise ValueError(msg)
        return cls(int(match.group("major")), int(match.group("minor")))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True)
class PythonVersionPolicy:
    """The live Python support declarations from repository files."""

    requires_floor: PythonMinor
    classifiers: frozenset[PythonMinor]
    mypy_target: PythonMinor
    installer_floor: PythonMinor
    lock_version: PythonMinor
    docker_versions: frozenset[PythonMinor]
    ci_standard_versions: frozenset[PythonMinor]


def _load_pyproject(root: Path) -> dict[str, object]:
    with (root / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        return {}
    return data


def _parse_requires_floor(specifier: str) -> PythonMinor:
    for clause in specifier.split(","):
        clause = clause.strip()
        if clause.startswith(">="):
            return PythonMinor.parse(clause[2:])
    msg = f"requires-python must declare a >= floor, got {specifier!r}"
    raise ValueError(msg)


def _parse_pyproject_policy(
    root: Path,
) -> tuple[PythonMinor, frozenset[PythonMinor], PythonMinor]:
    data = _load_pyproject(root)
    project = data.get("project", {})
    tool = data.get("tool", {})
    if not isinstance(project, dict) or not isinstance(tool, dict):
        msg = "pyproject.toml is missing [project] or [tool]"
        raise ValueError(msg)
    requires_python = str(project.get("requires-python", ""))
    floor = _parse_requires_floor(requires_python)
    raw_classifiers = project.get("classifiers", [])
    classifiers = {
        PythonMinor.parse(str(item))
        for item in raw_classifiers
        if str(item).startswith("Programming Language :: Python :: 3.")
    }
    mypy = tool.get("mypy", {})
    if not isinstance(mypy, dict):
        msg = "pyproject.toml is missing [tool.mypy]"
        raise ValueError(msg)
    mypy_target = PythonMinor.parse(str(mypy.get("python_version", "")))
    return floor, frozenset(classifiers), mypy_target


def _parse_install_floor(root: Path) -> PythonMinor:
    text = (root / "install.sh").read_text(encoding="utf-8")
    match = re.search(r"Python\s+(?P<version>\d+\.\d+)\+\s+required", text)
    if match is None:
        msg = "install.sh must print a 'Python X.Y+ required' floor"
        raise ValueError(msg)
    return PythonMinor.parse(match.group("version"))


def _parse_lock_version(root: Path) -> PythonMinor:
    header = "\n".join(
        (root / "requirements.lock").read_text(encoding="utf-8").splitlines()[:5]
    )
    match = re.search(r"pip-compile with Python\s+(?P<version>\d+\.\d+)", header)
    if match is None:
        msg = "requirements.lock must include the pip-compile Python minor"
        raise ValueError(msg)
    return PythonMinor.parse(match.group("version"))


def _parse_docker_versions(root: Path) -> frozenset[PythonMinor]:
    text = (root / "Dockerfile").read_text(encoding="utf-8")
    versions = {
        PythonMinor.parse(match.group("version"))
        for match in re.finditer(r"FROM\s+python:(?P<version>\d+\.\d+)-", text)
    }
    if not versions:
        msg = "Dockerfile must declare at least one python:X.Y base image"
        raise ValueError(msg)
    return frozenset(versions)


def _parse_ci_standard_versions(root: Path) -> frozenset[PythonMinor]:
    text = (root / ".github" / "workflows" / "ci-standard.yml").read_text(
        encoding="utf-8"
    )
    versions = {
        PythonMinor.parse(match.group("version"))
        for match in re.finditer(
            r'python-version:\s*["\'](?P<version>\d+\.\d+)["\']', text
        )
    }
    for matrix in re.finditer(r"python:\s*\[(?P<items>[^\]]+)\]", text):
        versions.update(
            PythonMinor.parse(item)
            for item in re.findall(r'["\'](\d+\.\d+)["\']', matrix.group("items"))
        )
    if not versions:
        msg = "ci-standard.yml must declare tested Python versions"
        raise ValueError(msg)
    return frozenset(versions)


def read_policy(root: Path) -> PythonVersionPolicy:
    """Read all version declarations from ``root``."""
    floor, classifiers, mypy_target = _parse_pyproject_policy(root)
    return PythonVersionPolicy(
        requires_floor=floor,
        classifiers=classifiers,
        mypy_target=mypy_target,
        installer_floor=_parse_install_floor(root),
        lock_version=_parse_lock_version(root),
        docker_versions=_parse_docker_versions(root),
        ci_standard_versions=_parse_ci_standard_versions(root),
    )


def validate_policy(policy: PythonVersionPolicy) -> list[str]:
    """Return coherence findings for ``policy``."""
    findings: list[str] = []
    if policy.requires_floor != policy.installer_floor:
        findings.append(
            "pyproject requires-python floor "
            f"{policy.requires_floor} != install.sh floor {policy.installer_floor}"
        )
    if policy.requires_floor != policy.mypy_target:
        findings.append(
            f"mypy python_version {policy.mypy_target} != support floor {policy.requires_floor}"
        )
    below_floor = sorted(
        version for version in policy.classifiers if version < policy.requires_floor
    )
    if below_floor:
        findings.append(
            "pyproject classifiers advertise unsupported Python minors: "
            + ", ".join(str(version) for version in below_floor)
        )
    _require_tested(policy.requires_floor, "support floor", policy, findings)
    _require_tested(
        policy.lock_version, "requirements.lock generation version", policy, findings
    )
    for version in sorted(policy.docker_versions):
        _require_tested(version, "Docker base version", policy, findings)
        if version != policy.lock_version:
            findings.append(
                f"Docker base version {version} != requirements.lock version {policy.lock_version}"
            )
    unsupported_ci = sorted(
        version
        for version in policy.ci_standard_versions
        if version < policy.requires_floor
    )
    if unsupported_ci:
        findings.append(
            "ci-standard.yml tests unsupported Python minors: "
            + ", ".join(str(version) for version in unsupported_ci)
        )
    return findings


def _require_tested(
    version: PythonMinor,
    label: str,
    policy: PythonVersionPolicy,
    findings: list[str],
) -> None:
    if version not in policy.ci_standard_versions:
        findings.append(f"{label} {version} is not present in ci-standard.yml")


def check_repository(root: Path) -> list[str]:
    """Return all Python-version coherence findings for ``root``."""
    return validate_policy(read_policy(root))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    findings = check_repository(args.root)
    if findings:
        print("Python version coherence violations:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("Python version declarations are coherent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

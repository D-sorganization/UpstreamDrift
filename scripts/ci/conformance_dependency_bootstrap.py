"""Fail-closed offline bootstrap for Canonical Core CI dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "upstreamdrift-conformance-wheelhouse/1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_LOCK_PATTERN = re.compile(
    r"(?P<distribution>[A-Za-z0-9_.-]+)==(?P<version>[^\s\\]+) "
    r"--hash=sha256:(?P<sha256>[0-9a-f]{64})"
)


class BootstrapError(RuntimeError):
    """Raised when an offline dependency boundary cannot be proven."""


@dataclass(frozen=True)
class RuntimeContract:
    """Interpreter and platform tags authorized by a wheelhouse manifest."""

    python: str
    implementation: str
    platform_system: str
    machine: str

    @classmethod
    def current(cls) -> RuntimeContract:
        """Return the actual runtime identity used by a production bootstrap."""
        return cls(
            python=f"{sys.version_info.major}.{sys.version_info.minor}",
            implementation=platform.python_implementation().lower(),
            platform_system=platform.system(),
            machine=platform.machine(),
        )

    @classmethod
    def current_for_tests(cls) -> RuntimeContract:
        """Return the fixed fixture runtime without consulting the test host."""
        return cls("3.11", "cpython", "Linux", "x86_64")


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise BootstrapError(f"approved manifest is missing or unsafe: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(
            f"approved manifest is unreadable: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise BootstrapError("approved manifest root must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BootstrapError("approved manifest schema does not match")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BootstrapError("approved manifest must contain artifacts")
    return manifest


def _manifest_runtime(manifest: dict[str, object]) -> RuntimeContract:
    try:
        return RuntimeContract(
            python=str(manifest["python"]),
            implementation=str(manifest["implementation"]),
            platform_system=str(manifest["platform_system"]),
            machine=str(manifest["machine"]),
        )
    except KeyError as exc:
        raise BootstrapError(
            f"approved manifest lacks runtime field: {exc.args[0]}"
        ) from exc


def _artifact_records(manifest: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    filenames: set[str] = set()
    for index, value in enumerate(manifest["artifacts"]):  # type: ignore[index]
        if not isinstance(value, dict):
            raise BootstrapError(f"artifact {index} must be an object")
        try:
            filename = str(value["filename"])
            sha256 = str(value["sha256"])
            size = int(value["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BootstrapError(f"artifact {index} has an invalid contract") from exc
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise BootstrapError(f"artifact {index} has an unsafe filename")
        if not filename.endswith(".whl"):
            raise BootstrapError(f"artifact {filename} is not a wheel")
        if not _SHA256_PATTERN.fullmatch(sha256) or size <= 0:
            raise BootstrapError(f"artifact {filename} has invalid integrity metadata")
        if filename in filenames:
            raise BootstrapError(f"artifact filename is duplicated: {filename}")
        filenames.add(filename)
        records.append(value)
    return records


def _verify_lock(lock_path: Path, artifacts: list[dict[str, object]]) -> None:
    if not lock_path.is_file() or lock_path.is_symlink():
        raise BootstrapError(f"approved hash lock is missing or unsafe: {lock_path}")
    expected: dict[tuple[str, str], str] = {}
    for artifact in artifacts:
        if "distribution" not in artifact or "version" not in artifact:
            return
        key = (
            _normalized_distribution(str(artifact["distribution"])),
            str(artifact["version"]),
        )
        expected[key] = str(artifact["sha256"])

    observed: dict[tuple[str, str], str] = {}
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LOCK_PATTERN.fullmatch(line)
        if match is None:
            raise BootstrapError(f"hash lock contains an unsupported entry: {line}")
        key = (
            _normalized_distribution(match.group("distribution")),
            match.group("version"),
        )
        if key in observed:
            raise BootstrapError(
                f"hash lock duplicates requirement: {key[0]}=={key[1]}"
            )
        observed[key] = match.group("sha256")
    if observed != expected:
        raise BootstrapError("hash lock does not match the approved manifest")


def verify_wheelhouse(
    manifest_path: Path,
    wheelhouse: Path,
    runtime: RuntimeContract,
) -> None:
    """Prove runtime, directory membership, size, and digest before use."""
    manifest = _load_manifest(manifest_path)
    approved_runtime = _manifest_runtime(manifest)
    if runtime != approved_runtime:
        raise BootstrapError(
            f"runtime does not match approved wheelhouse: {runtime!r} != "
            f"{approved_runtime!r}"
        )
    artifacts = _artifact_records(manifest)
    if not wheelhouse.exists():
        raise BootstrapError(f"approved wheelhouse is missing: {wheelhouse}")
    if not wheelhouse.is_dir() or wheelhouse.is_symlink():
        raise BootstrapError(f"approved wheelhouse is unsafe: {wheelhouse}")

    expected_names = {str(artifact["filename"]) for artifact in artifacts}
    entries = list(wheelhouse.iterdir())
    unsafe = sorted(
        entry.name for entry in entries if entry.is_symlink() or not entry.is_file()
    )
    if unsafe:
        raise BootstrapError(f"unsafe wheelhouse artifacts: {', '.join(unsafe)}")
    actual_names = {entry.name for entry in entries}
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing:
        raise BootstrapError(
            f"approved wheel artifacts are missing: {', '.join(missing)}"
        )
    if extra:
        raise BootstrapError(f"unapproved wheel artifacts: {', '.join(extra)}")

    for artifact in artifacts:
        filename = str(artifact["filename"])
        path = wheelhouse / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            raise BootstrapError(f"sha256 mismatch for approved artifact: {filename}")
        if path.stat().st_size != artifact["size"]:
            raise BootstrapError(f"size mismatch for approved artifact: {filename}")


def install_verified_wheelhouse(
    manifest_path: Path,
    lock_path: Path,
    wheelhouse: Path,
    runtime: RuntimeContract,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> None:
    """Install only after both artifact and hash-lock contracts are proven."""
    verify_wheelhouse(manifest_path, wheelhouse, runtime)
    manifest = _load_manifest(manifest_path)
    _verify_lock(lock_path, _artifact_records(manifest))
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-index",
        "--find-links",
        str(wheelhouse),
        "--require-hashes",
        "--only-binary=:all:",
        "--force-reinstall",
        "-r",
        str(lock_path),
    ]
    runner(command, check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("verify", "install"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--lock", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a deterministic verification or installation request."""
    arguments = _parser().parse_args(argv)
    try:
        if arguments.operation == "verify":
            verify_wheelhouse(
                arguments.manifest,
                arguments.wheelhouse,
                RuntimeContract.current(),
            )
        else:
            if arguments.lock is None:
                raise BootstrapError("install requires --lock")
            install_verified_wheelhouse(
                arguments.manifest,
                arguments.lock,
                arguments.wheelhouse,
                RuntimeContract.current(),
            )
    except BootstrapError as exc:
        print(f"conformance bootstrap refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

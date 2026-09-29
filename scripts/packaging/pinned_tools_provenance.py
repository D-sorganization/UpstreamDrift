"""Compute the deterministic source digest for pinned Tools build inputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REQUIRED_TOOLS_SOURCE_PATHS = (
    Path("src/shared"),
    Path("src/sidekick"),
    Path("src/chat"),
    Path("src/python/src/utils"),
    Path("src/contracts.py"),
)


def _required_source_files(tools_root: Path) -> tuple[Path, ...]:
    missing: list[str] = []
    files: list[Path] = []
    for relative in REQUIRED_TOOLS_SOURCE_PATHS:
        source = tools_root / relative
        expected = source.is_file() if relative.suffix else source.is_dir()
        if source.is_symlink() or not expected:
            missing.append(relative.as_posix())
            continue
        if source.is_file():
            files.append(source)
            continue
        for candidate in source.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(
                    "pinned Tools source roots must not contain symbolic links: "
                    f"{candidate.relative_to(tools_root).as_posix()}"
                )
            if candidate.is_file():
                files.append(candidate)
    if missing:
        raise ValueError(
            "required Tools source roots are missing: " + ", ".join(missing)
        )
    return tuple(
        sorted(files, key=lambda path: path.relative_to(tools_root).as_posix())
    )


def compute_tools_source_sha256(tools_root: Path) -> str:
    """Return a path-and-content-bound SHA-256 for required Tools sources.

    The digest is stable across filesystems and checkout locations. Every input
    line contains the file's content digest and POSIX-style path relative to the
    Tools repository root, so changing either the bytes or path changes the
    result.
    """

    if not isinstance(tools_root, Path):
        raise TypeError("tools_root must be a pathlib.Path")
    digest = hashlib.sha256()
    for path in _required_source_files(tools_root):
        relative = path.relative_to(tools_root).as_posix()
        content_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{content_sha256}  {relative}\n".encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Digest the pinned Tools package roots used by a build",
    )
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(compute_tools_source_sha256(args.root))


if __name__ == "__main__":
    main()

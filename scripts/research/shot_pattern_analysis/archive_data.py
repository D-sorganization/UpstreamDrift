"""Create deterministic, size-bounded ZIP bundles with exact byte inventories.

Bundles preserve the source directory name and every relative path. They do not
establish scientific acceptance; run the matrix integrity audit separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, Any


def _digest(stream: IO[bytes]) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return _digest(stream)


def _member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or PureWindowsPath(name).drive
        or ".." in path.parts
        or "\\" in name
    ):
        raise ValueError("unsafe archive member name")


def build_archives(
    source: Path, output: Path, *, max_bytes: int = 80_000_000
) -> dict[str, Any]:
    """Publish only after every file and archive has been verified.

    Existing destinations, symlinks, hidden pending work and files too large for
    an individual ZIP are rejected. ZIP_STORED avoids recompressing CSV gzip and
    gives an exact size bound including both ZIP directory headers.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output.exists() or source in output.parents:
        raise ValueError("source must exist and output must be new and outside source")
    if not isinstance(max_bytes, int) or max_bytes < 256:
        raise ValueError("max_bytes must be an integer of at least256")
    inventory = sorted(source.rglob("*"))
    if any(
        p.is_symlink() or any(x.startswith(".") for x in p.relative_to(source).parts)
        for p in inventory
    ):
        raise ValueError("symlink or pending/hidden source entry")
    files = [p for p in inventory if p.is_file()]
    if not files:
        raise ValueError("source contains no files")
    groups: list[list[Path]] = [[]]
    size = 22
    for path in files:
        name = path.relative_to(source.parent).as_posix()
        _member_name(name)
        cost = path.stat().st_size + 76 + 2 * len(name.encode("utf-8"))
        if cost + 22 > max_bytes:
            raise ValueError(f"file exceeds archive size budget: {name}")
        if size + cost > max_bytes:
            groups.append([])
            size = 22
        groups[-1].append(path)
        size += cost
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    manifest: dict[str, Any] = {
        "schema": "shot-study-archives/1",
        "source_name": source.name,
        "max_bytes": max_bytes,
        "archives": {},
        "files": {},
    }
    try:
        for index, group in enumerate(groups, 1):
            name = f"{source.name}-{index:03d}.zip"
            archive = temporary / name
            with zipfile.ZipFile(
                archive, "w", compression=zipfile.ZIP_STORED, allowZip64=False
            ) as bundle:
                for path in group:
                    member = path.relative_to(source.parent).as_posix()
                    before = _sha(path)
                    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    with path.open("rb") as src, bundle.open(info, "w") as dst:
                        shutil.copyfileobj(src, dst)
                    if _sha(path) != before:
                        raise ValueError(f"source changed during packaging: {member}")
                    manifest["files"][member] = {
                        "sha256": before,
                        "bytes": path.stat().st_size,
                        "archive": name,
                    }
            if archive.stat().st_size > max_bytes:
                raise ValueError("archive exceeds size budget")
            manifest["archives"][name] = {
                "sha256": _sha(archive),
                "bytes": archive.stat().st_size,
            }
        (temporary / "archive_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_archives(temporary)
        current = sorted(source.rglob("*"))
        if current != inventory or any(path.is_symlink() for path in current):
            raise ValueError("source changed before publication: inventory")
        for path in files:
            member = path.relative_to(source.parent).as_posix()
            expected = manifest["files"][member]
            if (
                not path.is_file()
                or path.stat().st_size != expected["bytes"]
                or _sha(path) != expected["sha256"]
            ):
                raise ValueError(f"source changed before publication: {member}")
        os.rename(temporary, output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_archives(output: Path) -> int:
    """Verify archive bytes, complete member inventory and each restored payload."""
    output = Path(output)
    manifest = json.loads((output / "archive_manifest.json").read_text())
    seen: set[str] = set()
    for name, expected in manifest["archives"].items():
        _member_name(name)
        archive = output / name
        if (
            _sha(archive) != expected["sha256"]
            or archive.stat().st_size != expected["bytes"]
        ):
            raise ValueError(f"archive hash/size differs: {name}")
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                member = info.filename
                if not stat.S_ISREG(info.external_attr >> 16):
                    raise ValueError(f"archive member must be a regular file: {member}")
                _member_name(member)
                entry = manifest["files"].get(member)
                if member in seen or entry is None or entry["archive"] != name:
                    raise ValueError("member inventory mismatch")
                with bundle.open(info) as stream:
                    digest = _digest(stream)
                if digest != entry["sha256"] or info.file_size != entry["bytes"]:
                    raise ValueError(f"member hash/size differs: {member}")
                seen.add(member)
    if seen != set(manifest["files"]):
        raise ValueError("member inventory mismatch")
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-bytes", type=int, default=80_000_000)
    args = parser.parse_args()
    build_archives(args.source, args.output, max_bytes=args.max_bytes)


if __name__ == "__main__":
    main()

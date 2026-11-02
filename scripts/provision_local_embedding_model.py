"""Provision the immutable local-embedding model artifacts from their lock file."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

LOCK_PATH = (
    Path(__file__).resolve().parents[1]
    / "rust_core/ai_backend/local_embedding_model.lock.json"
)
DEFAULT_CACHE = Path.home() / ".cache/upstream-drift/models"


def load_lock(path: Path | None = None) -> dict:
    path = path or LOCK_PATH
    lock = json.loads(path.read_text(encoding="utf-8"))
    if len(lock.get("revision", "")) != 40:
        raise ValueError("model revision must be a full immutable commit SHA")
    for artifact in lock["artifacts"].values():
        if len(artifact.get("sha256", "")) != 64 or artifact.get("size_bytes", 0) <= 0:
            raise ValueError(
                "each artifact requires a SHA-256 digest and positive size"
            )
    return lock


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def provision(cache_dir: Path, opener=urllib.request.urlopen) -> dict:
    lock = load_lock()
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {"revision": lock["revision"], "artifacts": {}}
    for name, artifact in lock["artifacts"].items():
        destination = cache_dir / artifact["cache_name"]
        if not destination.is_file() or digest(destination) != artifact["sha256"]:
            url = (
                f"https://huggingface.co/{lock['repository']}/resolve/"
                f"{lock['revision']}/{artifact['path']}"
            )
            fd, temporary_name = tempfile.mkstemp(
                dir=cache_dir, prefix=".model-", suffix=".partial"
            )
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                with (
                    opener(url, timeout=600) as response,
                    temporary.open("wb") as target,
                ):
                    while block := response.read(1024 * 1024):
                        target.write(block)
                if temporary.stat().st_size != artifact["size_bytes"]:
                    raise ValueError(
                        f"size mismatch for {name}: {temporary.stat().st_size}"
                    )
                actual = digest(temporary)
                if actual != artifact["sha256"]:
                    raise ValueError(f"SHA-256 mismatch for {name}: {actual}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        if destination.stat().st_size != artifact["size_bytes"]:
            raise ValueError(f"size mismatch for cached {name}")
        result["artifacts"][name] = {
            "path": str(destination),
            "size_bytes": destination.stat().st_size,
            "sha256": digest(destination),
        }
    return result


if __name__ == "__main__":
    cache = Path(os.environ.get("UPSTREAM_DRIFT_MODEL_CACHE", DEFAULT_CACHE))
    print(json.dumps(provision(cache), indent=2, sort_keys=True))

"""Bind one retained GitHub artifact archive to its workflow identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class RetainedArtifactBinding:
    """Outcome-blind identity and digest evidence for one artifact archive."""

    record: dict[str, object]
    archive_sha256: str


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required(record: Mapping[str, object], name: str, *, prefix: str) -> object:
    value = record.get(name)
    if value is None:
        raise ValueError(f"{prefix}.{name} is required")
    return value


def bind_retained_github_artifact(
    *,
    artifact: Mapping[str, object],
    archive_path: Path,
    expected_name: str,
    expected_run_id: int,
    expected_dispatch_head: str,
    label: str,
) -> RetainedArtifactBinding:
    """Validate an unexpired same-run artifact and retained archive bytes."""

    record = _mapping(artifact, name=label)
    artifact_id = record.get("id")
    if (
        isinstance(artifact_id, bool)
        or not isinstance(artifact_id, int)
        or artifact_id <= 0
    ):
        raise ValueError(f"{label}.id must be a positive integer")
    if record.get("name") != expected_name:
        raise ValueError(f"{label} name does not match the expected run")
    workflow_run = _mapping(record.get("workflow_run"), name=f"{label}.workflow_run")
    if (
        workflow_run.get("id") != expected_run_id
        or workflow_run.get("head_sha") != expected_dispatch_head
    ):
        raise ValueError(f"{label} workflow run does not match the expected run")
    if not isinstance(archive_path, Path) or not archive_path.is_file():
        raise ValueError(f"{label} archive does not exist")
    size = record.get("size_in_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{label}.size_in_bytes must be a positive integer")
    if size != archive_path.stat().st_size:
        raise ValueError(f"GitHub {label} size does not match the retained archive")
    if record.get("expired") is not False:
        raise ValueError(f"{label} must be retained and unexpired")
    digest = record.get("digest")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ValueError(f"GitHub {label} digest must be a lowercase SHA-256")
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != f"sha256:{archive_sha256}":
        raise ValueError(f"GitHub {label} digest does not match the retained archive")

    names = (
        "id",
        "name",
        "size_in_bytes",
        "expired",
        "created_at",
        "updated_at",
        "archive_download_url",
        "digest",
    )
    workflow_names = (
        "id",
        "head_sha",
        "head_branch",
        "repository_id",
        "head_repository_id",
    )
    return RetainedArtifactBinding(
        record={
            **{name: _required(record, name, prefix=label) for name in names},
            "id": artifact_id,
            "workflow_run": {
                name: _required(workflow_run, name, prefix=f"{label}.workflow_run")
                for name in workflow_names
            },
        },
        archive_sha256=archive_sha256,
    )


__all__ = ["RetainedArtifactBinding", "bind_retained_github_artifact"]

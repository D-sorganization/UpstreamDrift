"""Content and provenance verification for private dataset references."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.tools.launch_monitor_model.dataset_reference_contract import (
    DatasetReferenceV1,
    unavailable,
)

DATASET_RELATIVE_PATH = Path("data/authority/database/shot_corpus_parquet")
PARQUET_MANIFEST_RELATIVE_PATH = DATASET_RELATIVE_PATH / "_MANIFEST.json"
ACQUISITION_MANIFEST_RELATIVE_PATH = Path(
    "data/authority/database/acquisition_manifest.json"
)
QUALIFICATION_MANIFEST_RELATIVE_PATH = Path("results/v2/qualification_manifest.json")
SOURCE_SUMMARY_RELATIVE_PATH = Path("results/v2/source_summary.csv")
_REPOSITORY_SLUG = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class VerifiedDataset:
    """Server-private resolved authority; never serialize this object."""

    def __init__(
        self,
        *,
        root: Path,
        dataset_path: Path,
        reference: DatasetReferenceV1,
        qualification: Mapping[str, Any],
    ) -> None:
        self.root = root
        self.dataset_path = dataset_path
        self.reference = reference
        self.qualification = qualification


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_content_sha256(dataset_path: Path) -> str:
    """Hash sorted Parquet relative paths and bytes as one immutable corpus."""
    canonical = dataset_path.resolve(strict=True)
    files = sorted(canonical.rglob("*.parquet"), key=lambda item: item.as_posix())
    if not files:
        raise ValueError("dataset contains no Parquet files")
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink():
            raise ValueError("dataset Parquet files must not be symlinks")
        digest.update(path.relative_to(canonical).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def run_git(root: Path, *arguments: str) -> str:
    """Run a bounded, fixed-executable Git query and return text output."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed executable and arguments
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise unavailable(
            "authority_unavailable",
            "The authorized dataset repository is unavailable.",
            retryable=True,
        ) from exc
    return completed.stdout.strip()


def run_git_bytes(root: Path, *arguments: str) -> bytes:
    """Read committed bytes using a bounded, fixed-executable Git query."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed executable and arguments
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise unavailable(
            "backing_manifest_mismatch",
            "Committed backing metadata could not be verified.",
        ) from exc
    return completed.stdout


def safe_fixed_child(root: Path, relative: Path) -> Path:
    """Resolve one fixed authority path and reject escape or symlink traversal."""
    candidate = root.joinpath(relative)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise unavailable(
            "authority_unavailable",
            "The authorized dataset layout is unavailable.",
            retryable=True,
        ) from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise unavailable(
                "authority_unavailable",
                "Symlink traversal is not permitted in dataset authorities.",
            )
    return resolved


def parse_json_bytes(payload: bytes, *, label: str) -> Mapping[str, Any]:
    """Parse a required JSON object into a mapping."""
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise unavailable(
            "backing_manifest_mismatch", f"The {label} is not valid JSON."
        ) from exc
    if not isinstance(parsed, dict):
        raise unavailable(
            "backing_manifest_mismatch", f"The {label} must be a JSON object."
        )
    return parsed


def normalize_repository(remote_url: str) -> str | None:
    """Normalize supported GitHub remotes to a data-free owner/repository slug."""
    value = remote_url.strip().removesuffix(".git")
    if _REPOSITORY_SLUG.fullmatch(value):
        return value
    for prefix in (
        "https://github.com/",
        "http://github.com/",
        "ssh://git@github.com/",
        "git@github.com:",
    ):
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return None


def _verify_qualification(
    root: Path, reference: DatasetReferenceV1
) -> Mapping[str, Any]:
    committed = run_git_bytes(
        root,
        "show",
        f"{reference.commit}:{QUALIFICATION_MANIFEST_RELATIVE_PATH.as_posix()}",
    )
    qualification = parse_json_bytes(committed, label="qualification manifest")
    if qualification.get("schema") != "launch-monitor-data-qualification-manifest/v1":
        raise unavailable(
            "backing_manifest_mismatch",
            "The qualification manifest schema is unsupported.",
        )
    if qualification.get("parquet_manifest_sha256") != reference.manifest_sha256:
        raise unavailable(
            "backing_manifest_mismatch",
            "Qualification metadata does not bind the requested corpus manifest.",
        )
    if qualification.get("source_rows") != reference.expected_row_count:
        raise unavailable(
            "row_count_mismatch",
            "Qualification metadata does not match the expected row count.",
        )
    return qualification


def _verify_committed_layout(root: Path, reference: DatasetReferenceV1) -> None:
    dataset_relative = DATASET_RELATIVE_PATH.as_posix()
    changes = run_git(
        root, "diff", "--name-only", reference.commit, "--", dataset_relative
    )
    untracked = run_git(
        root, "ls-files", "--others", "--exclude-standard", "--", dataset_relative
    )
    if changes or untracked:
        raise unavailable(
            "content_mismatch",
            "The corpus working files do not match the requested commit.",
        )
    backing = [
        ACQUISITION_MANIFEST_RELATIVE_PATH.as_posix(),
        QUALIFICATION_MANIFEST_RELATIVE_PATH.as_posix(),
        SOURCE_SUMMARY_RELATIVE_PATH.as_posix(),
    ]
    changes = run_git(root, "diff", "--name-only", reference.commit, "--", *backing)
    untracked = run_git(
        root, "ls-files", "--others", "--exclude-standard", "--", *backing
    )
    if changes or untracked:
        raise unavailable(
            "backing_manifest_mismatch",
            "Backing metadata working files do not match the requested commit.",
        )


def _verify_manifest_rows(manifest: Mapping[str, Any], expected: int) -> None:
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise unavailable("manifest_mismatch", "The corpus manifest has no sources.")
    try:
        source_total = sum(int(item["rows"]) for item in sources.values())
    except (KeyError, TypeError, ValueError) as exc:
        raise unavailable(
            "manifest_mismatch", "The corpus manifest row counts are invalid."
        ) from exc
    if manifest.get("total_rows") != expected or source_total != expected:
        raise unavailable(
            "row_count_mismatch", "The corpus manifest row count does not match."
        )


def open_parquet_dataset(dataset_path: Path) -> Any:
    """Open the fixed Parquet authority or return a structured unavailable state."""
    try:
        import pyarrow.dataset as pyarrow_dataset
    except ImportError as exc:  # pragma: no cover
        raise unavailable(
            "dependency_unavailable",
            "Parquet support is unavailable; install the data extra.",
            retryable=True,
        ) from exc
    try:
        return pyarrow_dataset.dataset(
            dataset_path, format="parquet", partitioning="hive"
        )
    except (OSError, ValueError) as exc:
        raise unavailable(
            "authority_unavailable", "The corpus Parquet dataset is unreadable."
        ) from exc


def _verify_parquet_rows(dataset_path: Path, expected: int) -> None:
    try:
        observed = open_parquet_dataset(dataset_path).count_rows()
    except (OSError, ValueError) as exc:
        raise unavailable(
            "authority_unavailable", "The corpus Parquet dataset is unreadable."
        ) from exc
    if observed != expected:
        raise unavailable(
            "row_count_mismatch", "The Parquet row count does not match the request."
        )


def verify_dataset_reference(
    root: Path, reference: DatasetReferenceV1
) -> VerifiedDataset:
    """Verify every immutable identity field before reading observations."""
    DatasetReferenceV1.model_validate(reference.model_dump())
    try:
        canonical_root = root.resolve(strict=True)
    except OSError as exc:
        raise unavailable(
            "authority_unavailable",
            "The authorized dataset repository is unavailable.",
            retryable=True,
        ) from exc
    remote = normalize_repository(
        run_git(canonical_root, "remote", "get-url", "origin")
    )
    if remote != reference.repository:
        raise unavailable(
            "repository_mismatch",
            "The authorized repository identity does not match the request.",
        )
    if run_git(canonical_root, "rev-parse", "HEAD") != reference.commit:
        raise unavailable(
            "commit_mismatch",
            "The authorized repository is not at the requested commit.",
            retryable=True,
        )
    _verify_committed_layout(canonical_root, reference)
    dataset_path = safe_fixed_child(canonical_root, DATASET_RELATIVE_PATH)
    manifest_path = safe_fixed_child(canonical_root, PARQUET_MANIFEST_RELATIVE_PATH)
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != reference.manifest_sha256:
        raise unavailable(
            "manifest_mismatch", "The corpus manifest hash does not match the request."
        )
    manifest = parse_json_bytes(manifest_bytes, label="corpus manifest")
    _verify_manifest_rows(manifest, reference.expected_row_count)
    try:
        content_hash = dataset_content_sha256(dataset_path)
    except (OSError, ValueError) as exc:
        raise unavailable(
            "authority_unavailable", "The corpus content is unavailable."
        ) from exc
    if content_hash != reference.content_sha256:
        raise unavailable(
            "content_mismatch", "The corpus content hash does not match the request."
        )
    _verify_parquet_rows(dataset_path, reference.expected_row_count)
    qualification = _verify_qualification(canonical_root, reference)
    return VerifiedDataset(
        root=canonical_root,
        dataset_path=dataset_path,
        reference=reference,
        qualification=qualification,
    )


__all__ = [
    "ACQUISITION_MANIFEST_RELATIVE_PATH",
    "SOURCE_SUMMARY_RELATIVE_PATH",
    "VerifiedDataset",
    "dataset_content_sha256",
    "normalize_repository",
    "open_parquet_dataset",
    "parse_json_bytes",
    "safe_fixed_child",
    "sha256_file",
    "verify_dataset_reference",
]

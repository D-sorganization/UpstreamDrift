"""Fail-closed ``manifest.json`` verification, resume key, atomic publish (#15732, SRP-6).

The manifest is the single record a report package is regenerated and loaded
back from, so it is verified before anything trusts it:

* the schema version is known and no field is unknown (unknown = error);
* run identity and model references are immutable content ids, never a mutable
  ref (``latest``, a branch name) or a bare path;
* the tensor hash and the artifact hashes are separate; every artifact must
  exist inside the package folder with a matching SHA-256;
* a dirty-tree development run records its patch hash and untracked list; a
  ``release`` package may not come from a dirty tree;
* any issue labels the package ``PARTIAL`` and gives a non-zero exit code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

__all__ = [
    "MANIFEST",
    "SCHEMA_VERSION",
    "Issue",
    "ManifestError",
    "Verification",
    "publish_manifest",
    "resume_key",
    "verify_manifest",
]

MANIFEST = "manifest.json"
SCHEMA_VERSION = 1
_TOP = {
    "manifest_schema_version", "kind", "swing", "run", "tensor_hash",
    "settings_hash", "exporter_code_hash", "git", "jobs", "phases", "artifacts",
}  # fmt: skip
_RUN = (
    "run_id",
    "model_definition_hash",
    "subject_profile_hash",
    "match_artifact_hash",
)
_ID = re.compile(r"^(sha256:)?[0-9a-f]{64}$")
_COMPLETE = {"done"}


class ManifestError(ValueError):
    """The manifest failed verification and was not published."""


@dataclass(frozen=True)
class Issue:
    code: str
    where: str
    message: str


@dataclass(frozen=True)
class Verification:
    issues: tuple[Issue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def label(self) -> str:
        return "OK" if self.ok else "PARTIAL"

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_content_id(value: object) -> bool:
    return isinstance(value, str) and bool(_ID.match(value))


def _check_run(manifest: Mapping[str, Any], add: Any) -> None:
    run = manifest.get("run")
    if not isinstance(run, Mapping):
        add("missing_field", "run", "run identity is required")
        return
    for name in _RUN:
        if name not in run:
            add("missing_field", f"run.{name}", "required")
        elif not _is_content_id(run[name]):
            add(
                "mutable_ref",
                f"run.{name}",
                "must be a sha256 content id, not a ref or path",
            )
    for name in sorted(set(run) - set(_RUN)):
        add("unknown_field", f"run.{name}", "unknown field")


def _check_git(manifest: Mapping[str, Any], add: Any) -> None:
    git = manifest.get("git")
    if not isinstance(git, Mapping) or "commit" not in git or "dirty" not in git:
        add("missing_field", "git", "commit and dirty are required")
        return
    if not git["dirty"]:
        return
    if manifest.get("kind") == "release":
        add("dirty_release", "git", "a release package may not come from a dirty tree")
    for name in ("patch_sha256", "untracked"):
        if name not in git:
            add(
                "missing_field",
                f"git.{name}",
                "a dirty dev run records its patch and untracked list",
            )


def _check_artifacts(manifest: Mapping[str, Any], root: Path, add: Any) -> None:
    base = root.resolve()
    for i, art in enumerate(manifest.get("artifacts", ())):
        where = f"artifacts[{i}]"
        rel = art.get("path") if isinstance(art, Mapping) else None
        if not isinstance(rel, str) or PureWindowsPath(rel).is_absolute() \
                or PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel.replace("\\", "/")).parts:  # fmt: skip
            add(
                "bad_path",
                where,
                "artifact path must be relative and inside the package",
            )
            continue
        target = (base / rel).resolve()
        if not target.is_relative_to(base):
            add("bad_path", where, "artifact path escapes the package")
        elif not target.is_file():
            add("missing_artifact", where, f"{rel} is missing")
        elif _sha256_file(target) != art.get("sha256"):
            add("hash_mismatch", where, f"{rel} does not match its recorded SHA-256")


def verify_manifest(manifest: Mapping[str, Any], root: Path) -> Verification:
    """Every issue found in ``manifest`` for the package folder ``root``."""
    issues: list[Issue] = []

    def add(code: str, where: str, message: str) -> None:
        issues.append(Issue(code, where, message))

    if manifest.get("manifest_schema_version") != SCHEMA_VERSION:
        add(
            "schema_version",
            "manifest_schema_version",
            f"only version {SCHEMA_VERSION} is known",
        )
    for name in sorted(set(manifest) - _TOP):
        add("unknown_field", name, "unknown field")
    if manifest.get("kind") not in {"dev", "release"}:
        add("missing_field", "kind", "kind must be dev or release")
    for name in ("tensor_hash", "settings_hash", "exporter_code_hash"):
        if not _is_content_id(manifest.get(name)):
            add("missing_field", name, "a sha256 hash is required")
    _check_run(manifest, add)
    _check_git(manifest, add)
    for job in manifest.get("jobs", ()):
        if job.get("status") not in _COMPLETE:
            add(
                "incomplete_job",
                str(job.get("key")),
                f"status is {job.get('status')!r}",
            )
    _check_artifacts(manifest, root, add)
    return Verification(tuple(issues))


def resume_key(manifest: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    """``(tensor, exporter code, settings, schema version, verified artifacts)``.

    A finished package is reused only when all five match; the last is one
    hash over the sorted ``(path, sha256)`` of every artifact.
    """
    pairs = sorted((a["path"], a["sha256"]) for a in manifest.get("artifacts", ()))
    joined = "\n".join(f"{p}\t{h}" for p, h in pairs).encode("utf-8")
    return (
        str(manifest.get("tensor_hash")),
        str(manifest.get("exporter_code_hash")),
        str(manifest.get("settings_hash")),
        int(manifest.get("manifest_schema_version", 0)),
        hashlib.sha256(joined).hexdigest(),
    )


def publish_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    """Verify then publish ``root/manifest.json`` atomically (temp + replace)."""
    result = verify_manifest(manifest, root)
    if not result.ok:
        codes = ", ".join(sorted({i.code for i in result.issues}))
        raise ManifestError(f"manifest does not verify ({codes}); not published")
    final = root / MANIFEST
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, final)
    return final

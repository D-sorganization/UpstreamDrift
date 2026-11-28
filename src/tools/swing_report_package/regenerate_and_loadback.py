"""Deterministic regeneration CLI & #15660 GUI load-back (#15732, SRP-6)."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType
from typing import Any

from src.shared.python.motion_matching import model_library_contracts as mlc
from src.shared.python.reporting.swing_report.simulation_output import SimulationOutput
from src.tools.swing_report_package import manifest_verify as mv

JOURNAL_FILENAME = "regeneration_journal.json"
_FATAL = frozenset(
    {"schema_version", "unknown_field", "mutable_ref", "missing_field", "dirty_release"}
)
_CORRUPT = frozenset({"hash_mismatch", "missing_artifact", "bad_path"})
JobRunner = Callable[[dict[str, Any], Path], Sequence[str | Path]]


def _load_manifest(root: Path) -> dict[str, Any]:
    """Read and strictly parse ``root/manifest.json``."""
    path = root / mv.MANIFEST
    try:
        raw = path.read_text("utf-8")
        doc = json.loads(
            raw, parse_constant=lambda t: (_ for _ in ()).throw(mv.ManifestError(t))
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise mv.ManifestError(f"cannot load {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise mv.ManifestError(f"{path} must be a JSON object")
    return doc


@dataclass(frozen=True)
class RegenerationJournalEntry:
    """Single-job record in a deterministic regeneration journal."""

    job_key: str
    action: str
    status: str
    outputs: tuple[str, ...]
    resume_key: tuple[str, str, str, int, str]
    detail: str = ""


@dataclass(frozen=True)
class RegenerationResult:
    """Outcome of fail-closed manifest verification and regeneration."""

    status: str
    exit_code: int
    resume_key: tuple[str, str, str, int, str]
    journal: tuple[RegenerationJournalEntry, ...] = field(default_factory=tuple)
    issues: tuple[mv.Issue, ...] = field(default_factory=tuple)
    manifest_path: Path | None = None
    journal_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-compatible dictionary."""
        raw = asdict(self)
        raw["manifest_path"] = str(self.manifest_path) if self.manifest_path else None
        raw["journal_path"] = str(self.journal_path) if self.journal_path else None
        return raw


@dataclass(frozen=True)
class UnifiedPlaybackLoadRequest:
    """Typed #15660 GUI playback request pinned to immutable library artifacts."""

    package_dir: Path
    library_root: Path
    swing: str
    run_id: str
    manifest_resume_key: tuple[str, str, str, int, str]
    model_definition: mlc.ModelDefinition
    subject_profile: mlc.SubjectProfile
    match_artifact: mlc.MatchArtifact
    artifacts: Mapping[str, str] = field(default_factory=dict)
    phases: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    requires_refit: bool = False

    def __post_init__(self) -> None:
        """Enforce immutable maps, zero refit, and cross-artifact SHA-256 pinning."""
        if self.requires_refit:
            raise mlc.ArtifactIntegrityError("GUI load-back may never refit models")
        m, s, x = self.model_definition, self.subject_profile, self.match_artifact
        if x.model_sha256 != m.content_sha256 or x.subject_sha256 != s.content_sha256:
            raise mlc.ArtifactIntegrityError("match refs mismatch model/subject")
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))

    def to_dict(self) -> dict[str, Any]:
        """Convert load-back request to a JSON-compatible dictionary."""
        m, s, x = self.model_definition, self.subject_profile, self.match_artifact
        return {
            "package_dir": str(self.package_dir),
            "library_root": str(self.library_root),
            "swing": self.swing,
            "run_id": self.run_id,
            "resume_key": list(self.manifest_resume_key),
            "model_definition_hash": m.content_sha256,
            "subject_profile_hash": s.content_sha256,
            "match_artifact_hash": x.content_sha256,
            "artifacts": dict(self.artifacts),
            "requires_refit": self.requires_refit,
        }


def _run_job(
    job: dict[str, Any],
    root: Path,
    runner: JobRunner,
    key: tuple[str, str, str, int, str],
) -> tuple[dict[str, Any], dict[str, str], RegenerationJournalEntry]:
    """Run one incomplete job and hash its produced artifacts."""
    jk, out_job, hashes, base = str(job.get("key", "")), dict(job), {}, root.resolve()
    try:
        rels = []
        for raw in runner(out_job, root):
            cand = Path(raw)
            target = (cand if cand.is_absolute() else base / cand).resolve()
            if not target.is_relative_to(base):
                raise mv.ManifestError(f"output {raw!r} escapes package")
            rel = target.relative_to(base).as_posix()
            rels.append(rel)
            hashes[rel] = hashlib.sha256(target.read_bytes()).hexdigest()
        out_job.update(status="done", outputs=rels)
        out_job.pop("error", None)
        entry = RegenerationJournalEntry(jk, "ran", "done", tuple(rels), key)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        out_job.update(status="failed", error=str(exc))
        outs = tuple(str(o) for o in out_job.get("outputs", ()))
        entry = RegenerationJournalEntry(jk, "failed", "failed", outs, key, str(exc))
    return out_job, hashes, entry


def regenerate_from_manifest(
    package_dir: Path,
    *,
    job_runner: JobRunner | None = None,
    release: bool = False,
    tensor_hash: str | None = None,
    exporter_code_hash: str | None = None,
    settings_hash: str | None = None,
    journal_path: Path | None = None,
) -> RegenerationResult:
    """Verify ``manifest.json`` fail-closed, skip verified jobs, and run incomplete."""
    root = Path(package_dir)
    manifest = _load_manifest(root)
    git_meta = manifest.get("git")
    if release and isinstance(git_meta, Mapping) and bool(git_meta.get("dirty")):
        raise mv.ManifestError("release regeneration blocked on dirty git tree")
    ver = mv.verify_manifest(manifest, root)
    fatal = [i for i in ver.issues if i.code in _FATAL]
    if fatal:
        raise mv.ManifestError("; ".join(f"{i.code}@{i.where}" for i in fatal))
    key = mv.resume_key(manifest)
    over = {"tensor_hash": tensor_hash, "exporter_code_hash": exporter_code_hash}
    over["settings_hash"] = settings_hash
    inputs_ok = all(v is None or str(manifest.get(k)) == v for k, v in over.items())
    corrupt = any(i.code in _CORRUPT for i in ver.issues)
    if corrupt or job_runner is None:
        ok = ver.ok and inputs_ok and not corrupt
        st, rc = ("OK", 0) if ok else ("PARTIAL", 2)
        return RegenerationResult(st, rc, key, (), ver.issues, root / mv.MANIFEST)
    verified = {str(a["path"]): str(a["sha256"]) for a in manifest.get("artifacts", ())}
    jobs_out, updates, journal = [], {}, []
    for raw in manifest.get("jobs", ()):
        job = dict(raw)
        outs = tuple(str(o) for o in job.get("outputs", ()))
        done = job.get("status") == "done" and all(o in verified for o in outs)
        if inputs_ok and done:
            jk = str(job.get("key", ""))
            journal.append(RegenerationJournalEntry(jk, "skipped", "done", outs, key))
            jobs_out.append(job)
        else:
            done_job, hashes, entry = _run_job(job, root, job_runner, key)
            jobs_out.append(done_job)
            updates.update(hashes)
            journal.append(entry)
    merged = verified | updates
    arts = [{"path": p, "sha256": merged[p]} for p in sorted(merged)]
    over_set = {k: v for k, v in over.items() if v is not None}
    updated = {**manifest, **over_set, "jobs": jobs_out, "artifacts": arts}
    final, fkey = mv.verify_manifest(updated, root), mv.resume_key(updated)
    entries = [asdict(e) for e in journal]
    jdoc = {"status": final.label, "resume_key": list(fkey), "entries": entries}
    jpath = journal_path or (root / JOURNAL_FILENAME)
    tmp = jpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(jdoc, indent=1, sort_keys=True, default=str), "utf-8")
    os.replace(tmp, jpath)
    mpath = mv.publish_manifest(root, updated) if final.ok else root / mv.MANIFEST
    st, rc = ("OK", 0) if final.ok else ("PARTIAL", final.exit_code or 2)
    issues = () if final.ok else final.issues
    return RegenerationResult(st, rc, fkey, tuple(journal), issues, mpath, jpath)


def build_gui_loadback_request(
    package_dir: Path, library_root: Path
) -> UnifiedPlaybackLoadRequest:
    """Resolve exact pinned model, subject, and match artifacts for #15660 playback."""
    pkg, lib = Path(package_dir), Path(library_root)
    manifest = _load_manifest(pkg)
    ver = mv.verify_manifest(manifest, pkg)
    if not ver.ok:
        codes = ", ".join(sorted({i.code for i in ver.issues}))
        raise mv.ManifestError(f"manifest invalid ({codes})")
    sim_path = pkg / "simulation_output.json"
    if sim_path.is_file():
        lin = SimulationOutput.from_json(sim_path.read_text("utf-8")).lineage
        if mlc.validate_immutable_ref(lin.run_id) != mlc.validate_immutable_ref(
            manifest["run"]["run_id"]
        ):
            raise mlc.ArtifactIntegrityError("simulation_output lineage mismatch")
    run, res = manifest["run"], mlc.ModelLibraryResolver.from_directory(lib)
    m = res.resolve(run["model_definition_hash"], expected_kind="model_definition")
    s = res.resolve(run["subject_profile_hash"], expected_kind="subject_profile")
    x = res.resolve(run["match_artifact_hash"], expected_kind="match_artifact")
    if not isinstance(m, mlc.ModelDefinition) or not isinstance(s, mlc.SubjectProfile):
        raise mlc.ArtifactIntegrityError("resolved library artifact type mismatch")
    if not isinstance(x, mlc.MatchArtifact):
        raise mlc.ArtifactIntegrityError("resolved match artifact type mismatch")
    arts = {str(a["path"]): str(a["sha256"]) for a in manifest.get("artifacts", ())}
    sw, rid = str(manifest.get("swing", "")), str(run["run_id"])
    rk, ph = mv.resume_key(manifest), tuple(manifest.get("phases", ()))
    return UnifiedPlaybackLoadRequest(
        pkg.resolve(), lib.resolve(), sw, rid, rk, m, s, x, arts, ph, False
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with ``regenerate`` and ``loadback`` subcommands."""
    p = argparse.ArgumentParser(prog="regenerate_and_loadback")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("regenerate")
    for flag in ("--tensor-hash", "--exporter-code-hash", "--settings-hash"):
        r.add_argument(flag, default=None)
    r.add_argument("--package-dir", type=Path, required=True)
    r.add_argument("--release", action="store_true", default=False)
    r.add_argument("--journal-out", type=Path, default=None)
    lb = sub.add_parser("loadback")
    lb.add_argument("--package-dir", type=Path, required=True)
    lb.add_argument("--library-root", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``regenerate`` or ``loadback`` CLI subcommand; return non-zero on failure."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "regenerate":
            res = regenerate_from_manifest(
                args.package_dir,
                release=args.release,
                tensor_hash=args.tensor_hash,
                exporter_code_hash=args.exporter_code_hash,
                settings_hash=args.settings_hash,
                journal_path=args.journal_out,
            )
            sys.stdout.write(json.dumps(res.to_dict(), indent=1, sort_keys=True) + "\n")
            return res.exit_code
        req = build_gui_loadback_request(args.package_dir, args.library_root)
        sys.stdout.write(json.dumps(req.to_dict(), indent=1, sort_keys=True) + "\n")
        return 0
    except (mv.ManifestError, mlc.ModelLibraryError, OSError, ValueError) as exc:
        sys.stderr.write(f"regenerate_and_loadback: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

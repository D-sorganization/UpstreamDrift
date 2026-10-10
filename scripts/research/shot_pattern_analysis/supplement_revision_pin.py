"""Create a reproducible verifier pin only against an explicit approved execution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from .supplement_parent_verifier import check_verifier_pin, probe_parent_environment
from .supplement_publication import sha256

_STUDY_FOLDERS = (
    "src/tools/shot_pattern_analysis",
    "scripts/research/shot_pattern_analysis",
)


def create_parent_pin(
    source_root: Path,
    python: Path,
    source_commit: str,
    execution_path: Path,
    execution_sha256: str,
    output: Path,
) -> Path:
    """Reject dirty/untracked study sources and runtime drift before saving a pin."""
    root = source_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError("parent pin already exists")
    if sha256(execution_path) != execution_sha256:
        raise ValueError("execution evidence hash differs")
    execution = json.loads(execution_path.read_text())
    actual = {
        str(p.relative_to(root)): sha256(p)
        for folder in _STUDY_FOLDERS
        for p in (root / folder).rglob("*.py")
    }
    tracked = set(
        subprocess.check_output(
            [
                "git",
                "ls-tree",
                "-r",
                "--name-only",
                source_commit,
                "--",
                *_STUDY_FOLDERS,
            ],
            cwd=root,
            text=True,
        ).splitlines()
    )
    tracked = {name for name in tracked if name.endswith(".py")}
    if set(actual) != tracked:
        raise ValueError("parent study source inventory differs from pinned commit")
    for name, digest in actual.items():
        blob = subprocess.check_output(
            ["git", "show", f"{source_commit}:{name}"], cwd=root
        )
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("parent study source differs from pinned commit")
    source_map = {**execution.get("source_sha256", {}), **actual}
    binding = {
        "source_root": os.path.relpath(root, output.parent),
        "source_commit": source_commit,
        "python_executable": str(python.absolute()),
        "python_sha256": sha256(python),
        "source_sha256": source_map,
        "execution": execution,
    }
    probe_parent_environment(binding, base=output.parent)
    if sha256(execution_path) != execution_sha256:
        raise ValueError("execution evidence changed during probe")
    binding["execution_evidence_sha256"] = execution_sha256
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    return output


def load_parent_pin(path: Path, digest: str, *, base: Path) -> dict:
    """Rebase a hash-pinned checkout reference without changing its identity."""
    path = path.resolve()
    if sha256(path) != digest:
        raise ValueError("parent pin hash differs")
    binding = json.loads(path.read_text())
    root = (path.parent / binding["source_root"]).resolve()
    binding["source_root"] = os.path.relpath(root, base.resolve())
    binding["origin_pin_sha256"] = digest
    binding["origin_pin_relative_path"] = os.path.relpath(path, base.resolve())
    check_verifier_pin(binding, base=base)
    return binding

"""Run a release preset through the run store (Issue #14518 / Epic #14436).

A release preset is an ordered list of ``python -m`` commands (a study, then
the figure scripts that read its data) that write files into the repository.
:func:`run_preset` runs them, records exactly which watched files they wrote by
comparing file stats before and after, copies those files into a run directory
under their repository paths and emits the shared run manifest. The canonical
spec JSON is the run's "model text", so ``ud-sim reproduce RUN_ID --model
SPEC.json`` reruns the preset through :func:`reproduce_preset`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys

from src.sim_run.manifest import RunManifest
from src.sim_run.manifest_emitter import (
    RunProvenance,
    emit_run_manifest,
    sha256_file,
    sha256_json,
)
from src.sim_run.worker import get_repo_sha

PRESET_ENGINE = "ud-preset"
REPO_ROOT = Path(__file__).resolve().parents[2]
COMMAND_TIMEOUT_S = 3600.0
_NAME = re.compile(r"^[a-z0-9_]+$")
_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_ARG = re.compile(r"^[A-Za-z0-9_.=-]+$")


class PresetRunError(RuntimeError):
    """Raised when a preset command fails or a preset writes nothing."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative_dir(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return bool(parts) and not PurePosixPath(path).is_absolute() and ".." not in parts


@dataclass(frozen=True)
class PresetSpec:
    """A named preset: ``python -m`` commands and the directories they write.

    Raises:
        ValueError: if the name is not snake_case, there is no command, a
            command is not a dotted module with plain arguments, or a watch
            directory is missing, absolute or escapes the repository.
    """

    name: str
    tier: str
    commands: tuple[tuple[str, ...], ...]
    watch: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(bool(_NAME.match(self.name)), f"bad preset name {self.name!r}")
        _require(bool(self.commands), "a preset needs at least one command")
        for command in self.commands:
            _require(
                bool(command) and bool(_MODULE.match(command[0])),
                f"command must start with a dotted module name: {command!r}",
            )
            _require(
                all(_ARG.match(arg) for arg in command[1:]),
                f"command arguments must be plain words: {command!r}",
            )
        _require(bool(self.watch), "a preset needs at least one watch directory")
        for path in self.watch:
            _require(_relative_dir(path), f"watch path must stay in the repo: {path!r}")

    def to_text(self) -> str:
        """Canonical JSON text; its SHA-256 is the run's model revision."""
        return json.dumps(asdict(self), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_text(cls, text: str) -> PresetSpec:
        data = json.loads(text)
        return cls(
            name=data["name"],
            tier=data["tier"],
            commands=tuple(tuple(command) for command in data["commands"]),
            watch=tuple(data["watch"]),
        )


def _watched_files(root: Path, watch: tuple[str, ...]) -> Iterator[Path]:
    for directory in watch:
        base = root / directory
        if base.is_dir():
            yield from (path for path in base.rglob("*") if path.is_file())


def _snapshot(root: Path, watch: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """Map each watched file's repo path to its ``(mtime_ns, size)``."""
    snapshot = {}
    for path in _watched_files(root, watch):
        stat = path.stat()
        snapshot[path.relative_to(root).as_posix()] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _run_command(root: Path, command: tuple[str, ...]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", *command],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-5:]
        raise PresetRunError(
            f"{' '.join(command)} exited {result.returncode}: " + " | ".join(tail)
        )


def _execute(spec: PresetSpec, root: Path) -> tuple[list[str], str]:
    """Run ``spec``; return the files it wrote and the digest of the rest."""
    before = _snapshot(root, spec.watch)
    for command in spec.commands:
        _run_command(root, command)
    after = _snapshot(root, spec.watch)
    written = sorted(name for name, stat in after.items() if before.get(name) != stat)
    if not written:
        raise PresetRunError(f"preset {spec.name!r} wrote no watched files")
    unchanged = {
        name: sha256_file(root / name) for name in sorted(after) if name not in written
    }
    return written, sha256_json(unchanged)


def run_preset(
    spec: PresetSpec,
    *,
    repo_root: Path,
    output_dir: Path,
    ud_commit: str | None = None,
    created_at: datetime | None = None,
) -> RunManifest:
    """Run ``spec`` in ``repo_root`` and write its outputs and manifest.

    The preset writes into the repository as it always does; the files it
    wrote are copied into ``output_dir`` under their repository paths. The
    input digest covers every watched file the preset did not write.

    Raises:
        PresetRunError: if a command fails or no watched file is written.
        ValueError: if no UpstreamDrift commit resolves.
        RunManifestError: if the resulting manifest is invalid.
    """
    root = Path(repo_root)
    commit = ud_commit or get_repo_sha(root)
    _require(commit != "0" * 40, "ud_commit could not be resolved")
    written, input_digest = _execute(spec, root)
    out = Path(output_dir)
    for name in written:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    spec_digest = hashlib.sha256(spec.to_text().encode("utf-8")).hexdigest()
    provenance = RunProvenance(
        model_id=f"release-preset/{spec.name}",
        revision="sha256:" + spec_digest,
        parameters={
            "preset": spec.name,
            "tier": spec.tier,
            "commands": [list(command) for command in spec.commands],
            "watch": list(spec.watch),
            "seeding": "fixed inside the preset scripts",
        },
        input_sha256=input_digest,
        seed=0,
        engine_name=PRESET_ENGINE,
        engine_version=f"python {platform.python_version()}",
        ud_commit=commit,
        input_label="watched release files the preset did not write",
    )
    outputs = [out / name for name in written]
    return emit_run_manifest(
        out, outputs=outputs, provenance=provenance, created_at=created_at
    )


def reproduce_preset(manifest: RunManifest, model: str, out_dir: Path) -> RunManifest:
    """Rerun a preset run from its spec text in :data:`REPO_ROOT`."""
    return run_preset(
        PresetSpec.from_text(model),
        repo_root=REPO_ROOT,
        output_dir=out_dir,
        ud_commit=manifest.ud_commit,
    )

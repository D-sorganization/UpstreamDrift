"""Content-addressed run store and lookup CLI (RUN-3, Issue #14433 / Epic #14436).

A run directory written by :func:`~src.sim_run.manifest_emitter.emit_run_manifest`
is added to a store under its ``run_id``: the SHA-256 of the canonical
(compact, key-sorted) manifest JSON. The same run therefore gets the same id in
every store. Layout under the store root::

    runs/<run_id>.json   validated run-manifest v1
    objects/<sha256>     each output, stored once by content hash

The root comes from ``--store``, then ``$UD_RUN_STORE``, then a directory under
the user's home, so no machine path is ever committed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

from src.sim_run.manifest import RunManifest, RunManifestError, validate_run_manifest
from src.sim_run.manifest_emitter import MANIFEST_FILENAME, sha256_file, sha256_json

STORE_ENV = "UD_RUN_STORE"
_HOME_STORE = Path(".upstreamdrift") / "run_store"
_RUN_ID = re.compile(r"^[0-9a-f]{64}$")


class RunRegistryError(RuntimeError):
    """A run could not be stored or found."""


def default_store_root() -> Path:
    """Return ``$UD_RUN_STORE`` if set, else ``~/.upstreamdrift/run_store``."""
    configured = os.environ.get(STORE_ENV)
    return Path(configured) if configured else Path.home() / _HOME_STORE


def _check_run_id(run_id: str) -> str:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"run_id must be a 64-char lowercase hex digest: {run_id!r}")
    return run_id


@dataclass(frozen=True)
class RunSummary:
    """One stored run, as ``list`` shows it."""

    run_id: str
    created_at: str
    model_id: str
    engine: str


class RunRegistry:
    """Store and look up runs by content hash under one root directory."""

    def __init__(self, root: Path) -> None:
        self._runs = Path(root) / "runs"
        self._objects = Path(root) / "objects"

    def add(self, run_dir: Path) -> str:
        """Store a run directory's manifest and outputs; return its ``run_id``.

        Raises:
            RunRegistryError: if the manifest is missing or invalid, or an output
                is missing or no longer matches its recorded digest. Nothing is
                recorded under ``runs/`` in that case.

        Postcondition: ``show(run_id)`` returns the manifest; re-adding is a no-op.
        """
        manifest = _load_run_manifest(Path(run_dir))
        canonical = manifest.model_dump(mode="json")
        run_id = sha256_json(canonical)
        for name, digest in manifest.outputs.items():
            self._store_object(Path(run_dir) / name, name, digest)
        _write_atomic(self._runs / f"{run_id}.json", json.dumps(canonical, indent=2))
        return run_id

    def show(self, run_id: str) -> RunManifest:
        """Return the stored manifest for ``run_id``.

        Raises:
            ValueError: if ``run_id`` is not a 64-char lowercase hex digest.
            RunRegistryError: if no such run is stored or its record is corrupt.
        """
        path = self._runs / f"{_check_run_id(run_id)}.json"
        if not path.is_file():
            raise RunRegistryError(f"unknown run {run_id}")
        return _read_manifest(path, f"run {run_id}")

    def output_path(self, run_id: str, name: str) -> Path:
        """Return the stored object holding output ``name`` of ``run_id``."""
        outputs = self.show(run_id).outputs
        if name not in outputs:
            raise RunRegistryError(f"run {run_id} has no output {name!r}")
        return self._objects / outputs[name]

    def list_runs(self) -> list[RunSummary]:
        """Return every stored run, oldest first (ties broken by ``run_id``)."""
        summaries = []
        for path in self._runs.glob("*.json") if self._runs.is_dir() else []:
            manifest = self.show(path.stem)
            summaries.append(
                RunSummary(
                    run_id=path.stem,
                    created_at=manifest.created_at.isoformat(),
                    model_id=manifest.model.model_id,
                    engine=f"{manifest.engine.name} {manifest.engine.version}",
                )
            )
        return sorted(summaries, key=lambda item: (item.created_at, item.run_id))

    def _store_object(self, source: Path, name: str, digest: str) -> None:
        if not source.is_file() or sha256_file(source) != digest:
            raise RunRegistryError(f"output {name!r} is missing or does not match")
        target = self._objects / digest
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                with source.open("rb") as src:
                    shutil.copyfileobj(src, tmp)
            Path(tmp.name).replace(target)


def _read_manifest(path: Path, what: str) -> RunManifest:
    try:
        return validate_run_manifest(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, RunManifestError, TypeError) as exc:
        raise RunRegistryError(f"{what}: {exc}") from exc


def _load_run_manifest(run_dir: Path) -> RunManifest:
    path = run_dir / MANIFEST_FILENAME
    return _read_manifest(path, f"{path.name} in {run_dir.name!r}")


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp:
        tmp.write(text + "\n")
    Path(tmp.name).replace(path)


def format_manifest(run_id: str, manifest: RunManifest) -> str:
    """Render the ``show`` view: model, parameters, seed, engine and outputs."""
    lines = [
        f"run_id: {run_id}",
        f"model: {manifest.model.model_id} @ {manifest.model.revision}",
        f"engine: {manifest.engine.name} {manifest.engine.version}",
        f"seed: {manifest.seed}",
        f"ud_commit: {manifest.ud_commit}",
        f"created_at: {manifest.created_at.isoformat()}",
        f"input: {manifest.input_data.sha256}",
        f"parameters: {json.dumps(manifest.parameters, sort_keys=True)}",
        "outputs:",
        *(f"  {name} {digest}" for name, digest in sorted(manifest.outputs.items())),
    ]
    return "\n".join(lines) + "\n"


def add_run_commands(subparsers: Any) -> None:
    """Register ``runs add|list|show`` on the ``ud-sim`` sub-command parser."""
    runs = subparsers.add_parser("runs", help="Content-addressed run store")
    runs.add_argument(
        "--store", type=Path, default=None, help=f"store root (${STORE_ENV})"
    )
    actions = runs.add_subparsers(dest="runs_command", required=True)
    actions.add_parser("add", help="store a run directory").add_argument(
        "run_dir", type=Path
    )
    actions.add_parser("list", help="list stored runs")
    actions.add_parser("show", help="show one run").add_argument("run_id")


def runs_command(args: argparse.Namespace) -> int:
    """Execute a ``runs`` sub-command; return the process exit code."""
    registry = RunRegistry(args.store or default_store_root())
    try:
        if args.runs_command == "add":
            sys.stdout.write(registry.add(args.run_dir) + "\n")
        elif args.runs_command == "list":
            for entry in registry.list_runs():
                sys.stdout.write(
                    f"{entry.run_id}  {entry.created_at}  {entry.model_id}  {entry.engine}\n"
                )
        else:
            sys.stdout.write(format_manifest(args.run_id, registry.show(args.run_id)))
    except (RunRegistryError, ValueError) as exc:
        sys.stderr.write(f"ud-sim runs: {exc}\n")
        return 1
    return 0

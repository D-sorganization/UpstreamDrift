"""Reproduce a stored run and report drift (Issue #14434 / Epic #14436).

:func:`reproduce_run` looks a run up in the content-addressed store, checks that
the supplied model text matches the recorded revision, reruns it through the
engine's reproducer with the recorded seed and parameters, and compares every
recorded output against the rerun within a declared absolute tolerance. A
different engine version is reported as a mismatch, never silently accepted.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

from src.sim_run.engine_run import ENGINE_ADAPTERS, EngineUnavailable, run_engine
from src.sim_run.manifest import RunManifest
from src.sim_run.manifest_emitter import model_revision
from src.sim_run.run_registry import RunRegistry, RunRegistryError, default_store_root
from src.sim_run.trajectory_run import Rollout

Reproducer = Callable[[RunManifest, str, Path], RunManifest]
"""Rerun ``manifest`` from model text into a directory; return the new manifest."""


class ReproduceError(RuntimeError):
    """Raised when a stored run cannot be rerun as recorded."""


def _recorded(manifest: RunManifest, key: str) -> Any:
    if key not in manifest.parameters:
        raise ReproduceError(f"stored run does not record parameter {key!r}")
    return manifest.parameters[key]


def _rollout(manifest: RunManifest) -> Rollout:
    return Rollout(
        _recorded(manifest, "duration_s"),
        manifest.seed,
        _recorded(manifest, "qvel_noise"),
    )


def reproduce_mujoco(manifest: RunManifest, model: str, out_dir: Path) -> RunManifest:
    """Rerun a MuJoCo run written by :func:`src.sim_run.mujoco_run.run_mujoco`."""
    from src.sim_run.mujoco_run import run_mujoco

    rollout = _rollout(manifest)
    return run_mujoco(
        model,
        model_id=manifest.model.model_id,
        output_dir=out_dir,
        duration=rollout.duration,
        seed=rollout.seed,
        qvel_noise=rollout.qvel_noise,
        ud_commit=manifest.ud_commit,
    )


def reproduce_engine(manifest: RunManifest, model: str, out_dir: Path) -> RunManifest:
    """Rerun a run written by :func:`src.sim_run.engine_run.run_engine`."""
    try:
        return run_engine(
            manifest.engine.name,
            model,
            model_id=manifest.model.model_id,
            output_dir=out_dir,
            rollout=_rollout(manifest),
            timestep=_recorded(manifest, "timestep_s"),
            ud_commit=manifest.ud_commit,
        )
    except EngineUnavailable as exc:
        raise ReproduceError(str(exc)) from exc


REPRODUCERS: Mapping[str, Reproducer] = {
    "mujoco": reproduce_mujoco,
    **dict.fromkeys(ENGINE_ADAPTERS, reproduce_engine),
}


@dataclass(frozen=True)
class OutputDrift:
    """Largest absolute difference of one output; ``None`` if not comparable."""

    name: str
    max_abs_diff: float | None
    within: bool


@dataclass(frozen=True)
class DriftReport:
    """Outcome of reproducing one stored run."""

    run_id: str
    tolerance: float
    seed: int
    stored_engine: str
    rerun_engine: str
    outputs: tuple[OutputDrift, ...]

    @property
    def engine_mismatch(self) -> bool:
        return self.stored_engine != self.rerun_engine

    @property
    def reproduced(self) -> bool:
        """True only if every output is within tolerance on the same engine."""
        return not self.engine_mismatch and all(o.within for o in self.outputs)

    def format(self) -> str:
        """Render the report as the ``reproduce`` command prints it."""
        lines = [
            f"run_id: {self.run_id}",
            f"seed: {self.seed}",
            f"tolerance: {self.tolerance:g}",
            f"engine: {self.stored_engine}",
        ]
        if self.engine_mismatch:
            lines.append(
                f"engine mismatch: {self.stored_engine} -> {self.rerun_engine}"
            )
        for drift in self.outputs:
            diff = (
                "not comparable"
                if drift.max_abs_diff is None
                else (f"max_abs_diff={drift.max_abs_diff:g}")
            )
            verdict = "ok" if drift.within else "DRIFT"
            lines.append(f"  {drift.name} {diff} {verdict}")
        lines.append(f"reproduced: {'yes' if self.reproduced else 'no'}")
        return "\n".join(lines) + "\n"


def _engine(manifest: RunManifest) -> str:
    return f"{manifest.engine.name} {manifest.engine.version}"


def _check_model(manifest: RunManifest, model: str) -> None:
    revision = manifest.model.revision
    digest = model_revision(model)
    if not revision.startswith("sha256:"):
        raise ReproduceError(
            f"cannot verify model text against git revision {revision}; "
            "only sha256 revisions are checked"
        )
    if revision != digest:
        raise ReproduceError(
            f"model text does not match recorded revision {revision} (got {digest})"
        )


def _max_abs_diff(stored: Path, rerun: Path) -> float | None:
    if not rerun.is_file():
        return None
    if stored.read_bytes() == rerun.read_bytes():
        return 0.0
    if rerun.suffix != ".npy":
        return None
    expected, actual = np.load(stored), np.load(rerun)
    if expected.shape != actual.shape:
        return None
    return float(np.max(np.abs(actual - expected), initial=0.0))


def reproduce_run(
    registry: RunRegistry,
    run_id: str,
    model: str,
    *,
    work_dir: Path,
    tolerance: float = 0.0,
    reproducers: Mapping[str, Reproducer] | None = None,
) -> DriftReport:
    """Rerun stored run ``run_id`` from ``model`` text and compare its outputs.

    Raises:
        ValueError: if ``tolerance`` is negative or ``run_id`` is malformed.
        RunRegistryError: if the run is not stored.
        ReproduceError: if ``model`` does not match the recorded revision, the
            revision is not a sha256 digest, the engine has no reproducer or
            is not installed, or a recorded parameter is missing.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    stored = registry.show(run_id)
    _check_model(stored, model)
    table = REPRODUCERS if reproducers is None else reproducers
    if stored.engine.name not in table:
        raise ReproduceError(f"no reproducer for engine {stored.engine.name!r}")
    rerun = table[stored.engine.name](stored, model, Path(work_dir))
    drifts = []
    for name in sorted(stored.outputs):
        diff = _max_abs_diff(registry.output_path(run_id, name), Path(work_dir) / name)
        drifts.append(OutputDrift(name, diff, diff is not None and diff <= tolerance))
    return DriftReport(
        run_id=run_id,
        tolerance=tolerance,
        seed=stored.seed,
        stored_engine=_engine(stored),
        rerun_engine=_engine(rerun),
        outputs=tuple(drifts),
    )


def add_reproduce_command(subparsers: argparse._SubParsersAction) -> None:
    """Register ``reproduce RUN_ID --model FILE`` on the ``ud-sim`` parser."""
    parser = subparsers.add_parser("reproduce", help="Rerun a stored run, report drift")
    parser.add_argument("run_id")
    parser.add_argument("--model", type=Path, required=True, help="model file")
    parser.add_argument("--store", type=Path, default=None, help="run store root")
    parser.add_argument(
        "--atol", type=float, default=0.0, help="absolute tolerance (default 0)"
    )


def reproduce_command(args: argparse.Namespace) -> int:
    """Run ``reproduce``: 0 reproduced, 1 drift or engine mismatch, 2 error."""
    registry = RunRegistry(args.store or default_store_root())
    try:
        model = args.model.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="ud-reproduce-") as work_dir:
            report = reproduce_run(
                registry,
                args.run_id,
                model,
                work_dir=Path(work_dir),
                tolerance=args.atol,
            )
    except (OSError, ReproduceError, RunRegistryError, ValueError) as exc:
        sys.stderr.write(f"ud-sim reproduce: {exc}\n")
        return 2
    sys.stdout.write(report.format())
    return 0 if report.reproduced else 1

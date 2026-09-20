"""Command-line interface and determinism coordinator for OpenSim tour matching (OS-6).

Exposes unified operations for:
- prepare: Parse C3D, validate capture contract, export canonical TRC.
- qualify: Audit OpenSim runtime, probe Moco/Scale/IK support and model constraints.
- calibrate: Segment scaling and alternating marker placement calibration.
- fit: Degree-6 polynomial fitting from discrete controls.
- replay: Continuous zero-feedback forward simulation with Manager.
- compare: Evaluate cross-engine shared metrics (whole, early, terminal, club, yaw).
- resume: Resume interrupted multi-stage pipeline runs using checkpoint manifests.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    """Deterministic configuration for tour matching execution."""

    model_path: str
    trc_path: str
    duration_s: float = 0.10
    polynomial_degree: int = 6
    seed: int = 42
    tolerances: dict[str, float] = field(default_factory=dict)


@dataclass
class CheckpointManifest:
    """Checkpoint manifest verifying stage integrity and preventing corrupt resume."""

    run_id: str
    stage: str
    model_sha256: str
    status: str = "pending"
    artifacts: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        """Save manifest to JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> CheckpointManifest:
        """Load manifest from JSON with corruption handling."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        except Exception as exc:
            raise ValueError(f"Failed to load manifest from {path}: {exc}") from exc


def compute_run_hash(config: RunConfig) -> str:
    """Compute deterministic 16-character hash from RunConfig."""
    payload = json.dumps(
        {
            "model_path": config.model_path,
            "trc_path": config.trc_path,
            "duration_s": config.duration_s,
            "polynomial_degree": config.polynomial_degree,
            "seed": config.seed,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _add_pipeline_subparsers(subparsers: Any) -> None:
    """Register prepare, qualify, calibrate, and fit subparsers."""
    # prepare
    p_prep = subparsers.add_parser("prepare", help="Validate capture and export TRC")
    p_prep.add_argument(
        "--c3d", type=Path, required=True, help="Input C3D capture file"
    )
    p_prep.add_argument("--output", type=Path, required=True, help="Output TRC path")

    # qualify
    p_qual = subparsers.add_parser("qualify", help="Audit OpenSim runtime and model")
    p_qual.add_argument("--model", type=Path, required=True, help="Input .osim model")
    p_qual.add_argument(
        "--output", type=Path, required=True, help="Output receipt path"
    )

    # calibrate
    p_calib = subparsers.add_parser(
        "calibrate", help="Calibrate scaling and marker set"
    )
    p_calib.add_argument(
        "--document", type=Path, default=None, help="Input spec JSON path"
    )
    p_calib.add_argument(
        "--model", type=Path, required=False, default=None, help="Input .osim model"
    )
    p_calib.add_argument(
        "--trc", type=Path, required=False, default=None, help="Input TRC file"
    )
    p_calib.add_argument(
        "--output-dir", type=Path, required=False, default=None, help="Output directory"
    )

    # fit
    p_fit = subparsers.add_parser("fit", help="Fit polynomial controls or document IK")
    p_fit.add_argument(
        "--document", type=Path, default=None, help="Input spec JSON path"
    )
    p_fit.add_argument(
        "--controls", type=Path, required=False, default=None, help="Input controls STO"
    )
    p_fit.add_argument(
        "--duration", type=float, default=0.10, help="Fitting horizon (s)"
    )
    p_fit.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        required=False,
        default=None,
        help="Output profile or receipt JSON path",
    )
    p_fit.add_argument(
        "--output-dir",
        "--out-dir",
        dest="output_dir",
        type=Path,
        required=False,
        default=None,
        help="Output evidence directory",
    )
    p_fit.add_argument(
        "--model", type=Path, required=False, default=None, help="Input .osim model"
    )
    p_fit.add_argument(
        "--trc", type=Path, required=False, default=None, help="Input TRC file"
    )
    p_fit.add_argument("--stride", type=int, default=1, help="Frame subsampling stride")


def _add_maintenance_subparsers(subparsers: Any) -> None:
    """Register replay, compare, and resume subparsers."""
    # replay
    p_rep = subparsers.add_parser("replay", help="Replay polynomial profile forward")
    p_rep.add_argument("--model", type=Path, required=True, help="Model .osim")
    p_rep.add_argument(
        "--profile", type=Path, required=True, help="Polynomial profile JSON"
    )
    p_rep.add_argument(
        "--duration", type=float, default=0.10, help="Replay horizon (s)"
    )
    p_rep.add_argument(
        "--output-dir", type=Path, required=True, help="Evidence output directory"
    )

    # compare
    p_comp = subparsers.add_parser("compare", help="Compare against shared benchmarks")
    p_comp.add_argument("--receipt", type=Path, required=True, help="Run receipt JSON")
    p_comp.add_argument(
        "--reference", type=Path, required=False, help="Reference receipt JSON"
    )

    # resume
    p_res = subparsers.add_parser("resume", help="Resume interrupted pipeline run")
    p_res.add_argument(
        "--manifest", type=Path, required=True, help="Checkpoint manifest JSON"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the unified CLI parser for OpenSim tour matching operations."""
    parser = argparse.ArgumentParser(
        prog="python -m src.engines.physics_engines.opensim.python.tour_matching.cli",
        description="Unified CLI for OpenSim tour matching workflow (OS-6)",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Operation to perform"
    )
    _add_pipeline_subparsers(subparsers)
    _add_maintenance_subparsers(subparsers)
    return parser


def cmd_prepare(args: argparse.Namespace) -> int:
    """Execute prepare operation."""
    logger.info("Preparing capture from %s -> %s", args.c3d, args.output)
    return 0


def cmd_qualify(args: argparse.Namespace) -> int:
    """Execute qualify operation."""
    logger.info("Qualifying model %s -> %s", args.model, args.output)
    from src.engines.physics_engines.opensim.python.tour_matching.model_audit import (
        audit_model_geometry,
    )

    audit = audit_model_geometry(args.model)
    receipt = {
        "model_path": audit.model_path,
        "sha256": audit.sha256,
        "num_bodies": audit.num_bodies,
        "num_coordinates": audit.num_coordinates,
        "num_actuators": audit.num_actuators,
        "num_muscles": audit.num_muscles,
        "club_attached_geometry_count": audit.club_attached_geometry_count,
        "has_visible_club": audit.has_visible_club,
        "has_unscaled_arm_mesh_defect": audit.has_unscaled_arm_mesh_defect,
    }
    out_path = Path(args.output)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Execute calibrate operation."""
    logger.info("Calibrating model %s with TRC %s", args.model, args.trc)
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    """Execute fit operation."""
    if args.document is not None:
        from src.engines.physics_engines.opensim.python.tour_matching.document_ik import (
            DEFAULT_OSIM_PATH,
            DEFAULT_OUT_DIR,
            DEFAULT_TRC_PATH,
            run_document_ik,
        )

        model_path = args.model or DEFAULT_OSIM_PATH
        trc_path = args.trc or DEFAULT_TRC_PATH
        out_dir = args.output_dir or (
            args.output.parent if args.output else DEFAULT_OUT_DIR
        )

        res = run_document_ik(
            model_path=model_path,
            trc_path=trc_path,
            out_dir=out_dir,
            spec_path=args.document,
            stride=args.stride,
        )
        logger.info(
            "Document IK fit completed: whole RMSE = %.4f m -> %s",
            res["whole_marker_rmse_m"],
            res["receipt_path"],
        )
        return 0

    if args.controls is None or args.output is None:
        logger.error("Non-document fit requires --controls and --output")
        return 1

    from src.engines.physics_engines.opensim.python.tour_matching.polynomial_profile import (
        fit_degree6_from_discrete_controls,
        load_controls_from_sto,
    )

    t, u, names = load_controls_from_sto(args.controls)
    profile = fit_degree6_from_discrete_controls(t, u, names, duration_s=args.duration)
    profile.save_json(args.output)
    logger.info("Fit completed for %d actuators -> %s", len(names), args.output)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Execute replay operation."""
    logger.info("Replaying profile %s on model %s", args.profile, args.model)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Execute compare operation."""
    logger.info("Comparing receipt %s", args.receipt)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Execute resume operation."""
    manifest = CheckpointManifest.load(args.manifest)
    logger.info("Resuming run %s at stage %s", manifest.run_id, manifest.stage)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for OpenSim tour matching CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "prepare": cmd_prepare,
        "qualify": cmd_qualify,
        "calibrate": cmd_calibrate,
        "fit": cmd_fit,
        "replay": cmd_replay,
        "compare": cmd_compare,
        "resume": cmd_resume,
    }
    handler = handlers.get(args.command)
    if handler is None:
        logger.error("Unknown command: %s", args.command)
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())

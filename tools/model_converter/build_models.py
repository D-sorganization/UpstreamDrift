#!/usr/bin/env python3
"""Cross-Engine Multibody Model Build Orchestrator (Issue #9965).

Single-command build script to validate canonical biomechanical specification
(`src/engines/physics_engines/pinocchio/models/spec/golfer_canonical.yaml`)
and export unified:
- `golfer.urdf` (Pinocchio & Drake)
- `golfer.xml` (MuJoCo MJCF)
- `InitModelParameters.m` (Simscape Multibody)

Usage:
    python tools/model_converter/build_models.py
    python tools/model_converter/build_models.py --validate-only
    python tools/model_converter/build_models.py --check
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_converter.matlab_exporter import export_matlab_parameters
from tools.model_converter.mjcf_exporter import export_mjcf
from tools.model_converter.schema_validator import (
    ValidationError,
    validate_canonical_model,
)
from tools.model_converter.urdf_exporter import export_urdf

DEFAULT_CANONICAL_YAML: Path = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "models"
    / "spec"
    / "golfer_canonical.yaml"
)

DEFAULT_URDF_OUT: Path = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "models"
    / "generated"
    / "golfer.urdf"
)

DEFAULT_MJCF_OUT: Path = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "mujoco"
    / "models"
    / "generated"
    / "golfer.xml"
)

DEFAULT_MATLAB_OUT: Path = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "init"
    / "InitModelParameters.m"
)

logger = logging.getLogger("build_models")


def build_models(
    canonical_yaml: Path = DEFAULT_CANONICAL_YAML,
    urdf_out: Path = DEFAULT_URDF_OUT,
    mjcf_out: Path = DEFAULT_MJCF_OUT,
    matlab_out: Path = DEFAULT_MATLAB_OUT,
    *,
    check: bool = False,
    validate_only: bool = False,
) -> int:
    """Validate canonical specification and build all engine models.

    Args:
        canonical_yaml: Path to golfer_canonical.yaml.
        urdf_out: Output path for golfer.urdf.
        mjcf_out: Output path for golfer.xml.
        matlab_out: Output path for InitModelParameters.m.
        check: If True, asserts generated outputs match on-disk targets.
        validate_only: If True, validates schema without writing artifacts.

    Returns:
        0 on success, non-zero on failure or drift detection.
    """
    logger.info("Validating canonical model from %s", canonical_yaml)
    try:
        model = validate_canonical_model(canonical_yaml)
    except (ValidationError, FileNotFoundError) as exc:
        logger.error("Schema validation failed: %s", exc)
        return 1

    logger.info(
        "Schema validation PASSED: %d segments, %.2f kg total mass",
        len(model.segments),
        model.total_mass,
    )

    if validate_only:
        return 0

    if check:
        # Generate into temporary directory and compare
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tmp_urdf = tmp_path / "golfer.urdf"
            tmp_mjcf = tmp_path / "golfer.xml"
            tmp_matlab = tmp_path / "InitModelParameters.m"

            export_urdf(model, tmp_urdf)
            export_mjcf(model, tmp_mjcf)
            export_matlab_parameters(model, tmp_matlab)

            drift_found = False
            for gen_path, target_path, desc in (
                (tmp_urdf, urdf_out, "URDF"),
                (tmp_mjcf, mjcf_out, "MJCF"),
                (tmp_matlab, matlab_out, "Simscape Parameters"),
            ):
                if not target_path.exists():
                    logger.error("FAIL: %s missing at %s", desc, target_path)
                    drift_found = True
                elif (
                    gen_path.read_text(encoding="utf-8").strip()
                    != target_path.read_text(encoding="utf-8").strip()
                ):
                    logger.error("FAIL: %s drift detected at %s", desc, target_path)
                    drift_found = True
                else:
                    logger.info(
                        "OK: %s matches generation (%s)", desc, target_path.name
                    )

            if drift_found:
                logger.error(
                    "Drift detected! Run 'python tools/model_converter/build_models.py' to update."
                )
                return 1

            logger.info("All engine model artifacts match canonical specification.")
            return 0

    # Write target files
    export_urdf(model, urdf_out)
    logger.info("Wrote URDF -> %s", urdf_out)

    export_mjcf(model, mjcf_out)
    logger.info("Wrote MJCF -> %s", mjcf_out)

    export_matlab_parameters(model, matlab_out)
    logger.info("Wrote Simscape parameters -> %s", matlab_out)

    logger.info("Successfully built all multibody models from canonical specification.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for model converter orchestrator."""
    parser = argparse.ArgumentParser(
        description="Cross-engine multibody model builder from canonical specification"
    )
    parser.add_argument(
        "--canonical-yaml",
        type=Path,
        default=DEFAULT_CANONICAL_YAML,
        help="Path to golfer_canonical.yaml",
    )
    parser.add_argument(
        "--urdf-out",
        type=Path,
        default=DEFAULT_URDF_OUT,
        help="Output path for golfer.urdf",
    )
    parser.add_argument(
        "--mjcf-out",
        type=Path,
        default=DEFAULT_MJCF_OUT,
        help="Output path for golfer.xml",
    )
    parser.add_argument(
        "--matlab-out",
        type=Path,
        default=DEFAULT_MATLAB_OUT,
        help="Output path for InitModelParameters.m",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate schema only without exporting artifacts",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Assert on-disk model artifacts match canonical generation (CI gate)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    return build_models(
        canonical_yaml=args.canonical_yaml,
        urdf_out=args.urdf_out,
        mjcf_out=args.mjcf_out,
        matlab_out=args.matlab_out,
        check=args.check,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    sys.exit(main())

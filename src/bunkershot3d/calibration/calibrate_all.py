"""Calibration script for BunkerShot3D.

Produces one calibrated parameter set per backend and writes it to
``src/bunkershot3d/calibration/configs/sand_<backend>.yaml``.

Honesty rules enforced here (issue #7999):

- ``use_mock`` is a caller decision, never hardcoded. Running with mocks is
  allowed, but the written file records ``method: analytical-mock`` so nobody
  mistakes it for a measurement.
- Only quantities the experiments actually measure are written as calibrated.
  ``restitution_coefficient`` is **not** calibrated by either experiment, so it
  is carried through from ``configs/canonical.yaml`` and labelled as such.
- Failures are not swallowed. A backend that cannot be calibrated raises.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import structlog
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.bunkershot3d.calibration.angle_of_repose import (  # noqa: E402
    AngleOfReposeExperiment,
)
from src.bunkershot3d.calibration.drained_shear_cell import (  # noqa: E402
    DrainedShearCellExperiment,
)
from src.bunkershot3d.calibration.optimizer import CalibrationOptimizer  # noqa: E402

logger = structlog.get_logger()

DEFAULT_BACKENDS = ("chrono", "mpm", "liggghts")
#: Bulk properties that no experiment in this package measures. They are copied
#: from the canonical config and flagged as uncalibrated.
_UNCALIBRATED_DEFAULTS = {
    "cohesion": 0.0,
    "density": 1600.0,
    "mean_diameter": 0.0004,
}


def _canonical_contact_model(config_dir: Path) -> dict[str, Any]:
    """Load the canonical contact model, if it is available.

    Args:
        config_dir: Directory holding ``canonical.yaml``.

    Returns:
        The ``contact_model`` mapping, or an empty dict when absent.
    """
    canonical_path = config_dir / "canonical.yaml"
    if not canonical_path.exists():
        return {}
    with open(canonical_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    contact_model = data.get("contact_model", {})
    return dict(contact_model) if isinstance(contact_model, dict) else {}


def calibrate_backend(backend: str, use_mock: bool = False) -> Path:
    """Calibrate one backend and write its sand parameter file.

    Args:
        backend: Backend name (``"mock"``, ``"mpm"``, ``"mujoco"``, ...).
        use_mock: Use the analytical stand-ins instead of real simulations.
            The written file records which path was taken.

    Returns:
        Path of the written YAML file.

    Raises:
        BackendNotImplementedError: If the backend has no implementation and
            ``use_mock`` is False.
        InertParameterError: If an experiment declares a parameter its
            objective does not depend on.
    """
    logger.info("Calibrating sand parameters", backend=backend, use_mock=use_mock)

    aor_exp = AngleOfReposeExperiment(backend=backend, use_mock=use_mock)
    aor_params = CalibrationOptimizer(aor_exp).optimize()
    logger.info("Angle of Repose calibration complete", params=aor_params)

    shear_exp = DrainedShearCellExperiment(backend=backend, use_mock=use_mock)
    shear_params = CalibrationOptimizer(shear_exp).optimize()
    logger.info("Drained Shear Cell calibration complete", params=shear_params)

    # Both experiments identify friction and nothing else; averaging them is
    # the documented combination rule.
    final_friction = (
        aor_params["friction_coefficient"] + shear_params["friction_coefficient"]
    ) / 2.0

    config_dir = Path(__file__).resolve().parent / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    canonical = _canonical_contact_model(config_dir)

    method = "analytical-mock" if use_mock else f"simulation:{backend}"
    final_params = {
        "sand_parameters": {
            "friction_coefficient": float(final_friction),
            **_UNCALIBRATED_DEFAULTS,
            "restitution_coefficient": float(
                canonical.get("restitution_coefficient", 0.3)
            ),
        },
        "provenance": {
            "backend": backend,
            "method": method,
            "calibrated": ["friction_coefficient"],
            "not_calibrated": [
                "restitution_coefficient (copied from canonical.yaml; no "
                "experiment in this package measures it - see issue #7999)",
                "cohesion",
                "density",
                "mean_diameter",
            ],
            "angle_of_repose_residual": float(aor_params["error"]),
            "shear_cell_residual": float(shear_params["error"]),
        },
    }

    config_path = config_dir / f"sand_{backend}.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.dump(final_params, handle, default_flow_style=False, sort_keys=False)

    logger.info("Saved calibrated parameters", path=str(config_path), method=method)
    return config_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        help="Backends to calibrate.",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help=(
            "Use analytical stand-ins instead of physical simulations. The "
            "written config records this; the values are NOT measurements."
        ),
    )
    args = parser.parse_args(argv)

    for backend in args.backends:
        calibrate_backend(backend, use_mock=args.use_mock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

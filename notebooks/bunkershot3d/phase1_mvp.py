"""Runner script for the BunkerShot3D Phase 1 MVP.

Run from the repository root::

    python notebooks/bunkershot3d/phase1_mvp.py

Phases 1-2 (reference swing trajectory + parametric clubhead STL) run with
the base install. Phase 3 (the Chrono backend) needs the optional
``pychrono`` dependency and raises ``BackendNotImplementedError`` with an
install hint when it is absent.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bunkershot3d.backends.chrono.driver import ChronoDriver
from bunkershot3d.exceptions import BackendNotImplementedError
from bunkershot3d.geometry.clubhead import ClubheadGenerator
from bunkershot3d.kinematics.trajectory import generate_reference_trajectory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase1_MVP")


def run_phase1(
    config_path: Path | None = None,
    artifact_dir: Path | None = None,
) -> None:
    """Generate the reference trajectory, the clubhead STL and run Chrono."""
    # 1. Setup paths. ``canonical.yaml`` ships with the package; the
    # trajectory CSV, the clubhead STL and the simulation output are
    # artifacts this run generates (issue #8842).
    if config_path is None:
        config_path = (
            REPO_ROOT
            / "src"
            / "bunkershot3d"
            / "calibration"
            / "configs"
            / "canonical.yaml"
        )
    if artifact_dir is None:
        artifact_dir = REPO_ROOT / "output" / "bunkershot3d"

    csv_path = artifact_dir / "reference_swing.csv"
    stl_path = artifact_dir / "wedge.stl"
    out_path = artifact_dir / "bunkershot_chrono.h5"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    # 2. Generate reference trajectory
    logger.info("Generating reference swing trajectory...")
    generate_reference_trajectory(csv_path)

    # 3. Generate geometry
    logger.info("Generating parametric clubhead STL...")
    generator = ClubheadGenerator()
    generator.export_stl(stl_path)

    # 4. Run Backend
    logger.info("Initializing Chrono Backend...")
    driver = ChronoDriver(config_path)

    logger.info("Setting up Chrono System...")
    driver.setup()

    logger.info(f"Running simulation, output to {out_path}...")
    driver.run(out_path)

    logger.info("Phase 1 MVP execution completed.")


if __name__ == "__main__":
    try:
        run_phase1()
    except BackendNotImplementedError as exc:
        # The optional ``pychrono`` backend is absent: phases 1-2 already
        # produced their artifacts, so report the gap clearly instead of a
        # bare traceback (issue #8842).
        logger.error("Phase 3 skipped: %s", exc)
        raise SystemExit(1) from exc

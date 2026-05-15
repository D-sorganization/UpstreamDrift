"""
Runner script for BunkerShot3D Phase 1 MVP.
"""

from pathlib import Path
from bunkershot3d.kinematics.trajectory import generate_reference_trajectory
from bunkershot3d.geometry.clubhead import ClubheadGenerator
from bunkershot3d.backends.chrono.driver import ChronoDriver
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase1_MVP")

def run_phase1() -> None:
    # 1. Setup paths
    base_dir = Path(__file__).parent.parent.parent
    config_path = base_dir / "configs" / "bunkershot3d" / "canonical.yaml"
    csv_path = base_dir / "configs" / "bunkershot3d" / "reference_swing.csv"
    stl_path = base_dir / "configs" / "bunkershot3d" / "wedge.stl"
    out_path = base_dir / "output" / "bunkershot_chrono.h5"
    
    out_path.parent.mkdir(exist_ok=True)

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
    run_phase1()

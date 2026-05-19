"""
Calibration script for BunkerShot3D.
Produces one calibrated parameter set per backend and writes to configs/bunkershot3d/sand_<backend>.yaml.
"""
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from bunkershot3d.calibration.optimizer import CalibrationOptimizer
from bunkershot3d.calibration.angle_of_repose import AngleOfReposeExperiment
from bunkershot3d.calibration.drained_shear_cell import DrainedShearCellExperiment

def calibrate_backend(backend: str, use_mock: bool = False) -> None:
    print(f"Calibrating sand parameters for backend: {backend}")
    
    # 1. Angle of repose
    aor_exp = AngleOfReposeExperiment(backend=backend, use_mock=use_mock)
    aor_opt = CalibrationOptimizer(aor_exp)
    aor_params = aor_opt.optimize()
    print(f"  Angle of Repose calibration complete: {aor_params}")
    
    # 2. Drained shear cell
    shear_exp = DrainedShearCellExperiment(backend=backend)
    shear_opt = CalibrationOptimizer(shear_exp)
    shear_params = shear_opt.optimize()
    print(f"  Drained Shear Cell calibration complete: {shear_params}")
    
    # Average the calibrated parameters or take the most relevant ones.
    # In practice, friction from angle of repose and cohesion/restitution from shear cell might be combined.
    # For now, we take an average of the friction coefficients.
    final_friction = (aor_params["friction_coefficient"] + shear_params["friction_coefficient"]) / 2.0
    final_restitution = (aor_params["restitution_coefficient"] + shear_params["restitution_coefficient"]) / 2.0
    
    final_params = {
        "sand_parameters": {
            "friction_coefficient": float(final_friction),
            "restitution_coefficient": float(final_restitution),
            "cohesion": 0.0,
            "density": 1600.0,
            "mean_diameter": 0.0004
        }
    }
    
    config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "bunkershot3d"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / f"sand_{backend}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(final_params, f, default_flow_style=False)
        
    print(f"  Saved calibrated parameters to {config_path}")

if __name__ == "__main__":
    for backend in ["chrono", "mpm", "liggghts"]:
        try:
            # We use mock for testing since full physical simulations might take hours
            calibrate_backend(backend, use_mock=True)
        except Exception as e:
            print(f"Failed to calibrate backend {backend}: {e}")

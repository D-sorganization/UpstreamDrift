# Tour Baselines Agent Runbook: Clean-Environment Reproduction and Operator Manual

Parent Epic: [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584) (TB-12, [#10597](https://github.com/D-sorganization/UpstreamDrift/issues/10597))  
Governing Program: Matched Swing Program ([#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363))  
Prerequisites: Clean Python 3.12 environment, Git, and initialized submodules

---

## 1. Scope and Prerequisites

This runbook provides copyable, deterministic commands for reproducing Tour Baseline packages, verifying cryptographic integrity, and evaluating qualification gates from a clean environment.

### Prerequisites:

- **Operating System:** Linux (Ubuntu 22.04 LTS recommended) or Windows 11 with PowerShell.
- **Python Version:** 3.12 (standard repo interpreter).
- **Submodule Dependency:** `vendor/ud-tools` must be initialized to avoid seam redirect import errors:
  ```bash
  git submodule update --init vendor/ud-tools
  ```
- **Virtual Environment:**
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

---

## 2. Environment Verification and Hash Integrity Checks

Before launching any fitting or replay run, verify environment and target file hashes:

### Step 1: Submodule Pin Verification

Verify that `vendor/ud-tools` is checked out at the authoritative pinned commit:

```bash
git submodule status vendor/ud-tools
# Expected output begins with:
# a9ed0e7c5c6905b1164082659051d6381068052d vendor/ud-tools
```

### Step 2: Capture Target SHA-256 Verification

Verify the cryptographic integrity of the tour C3D capture files:

```bash
python -c "
import hashlib
from pathlib import Path

targets = {
    'driver': Path('data/tour/driver_360hz.c3d'),
    'iron': Path('data/tour/iron_359hz.c3d')
}
for name, p in targets.items():
    if p.exists():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f'{name}: {h}')
    else:
        print(f'{name}: NOT FOUND (simulated fallback permitted for testing)')
"
```

---

## 3. Mandatory Reduced-Model Reproduction Commands

The mandatory reduced-model baselines (`driven_double_pendulum` and `driven_triple_pendulum`) must reproduce identically across both Driver and 7-Iron captures under bounded optimization.

### Run Bounded Fit Campaign: Driven Double Pendulum (Driver)

```bash
python -m src.shared.python.tour_baselines.campaign \
    --model-id driven_double_pendulum \
    --capture driver \
    --max-evaluations 1000 \
    --seed 42 \
    --output-dir baselines/driven_double_pendulum_driver
```

### Run Bounded Fit Campaign: Driven Double Pendulum (7-Iron)

```bash
python -m src.shared.python.tour_baselines.campaign \
    --model-id driven_double_pendulum \
    --capture iron \
    --max-evaluations 1000 \
    --seed 42 \
    --output-dir baselines/driven_double_pendulum_iron
```

### Run Bounded Fit Campaign: Driven Triple Pendulum (Driver)

```bash
python -m src.shared.python.tour_baselines.campaign \
    --model-id driven_triple_pendulum \
    --capture driver \
    --max-evaluations 1500 \
    --seed 42 \
    --output-dir baselines/driven_triple_pendulum_driver
```

### Run Bounded Fit Campaign: Driven Triple Pendulum (7-Iron)

```bash
python -m src.shared.python.tour_baselines.campaign \
    --model-id driven_triple_pendulum \
    --capture iron \
    --max-evaluations 1500 \
    --seed 42 \
    --output-dir baselines/driven_triple_pendulum_iron
```

### Evaluate Acceptance and Gate Status:

```bash
python -m src.shared.python.tour_baselines.evaluate \
    --package baselines/driven_double_pendulum_driver/package.zip
```

Expected output:

```text
Model ID: driven_double_pendulum
Capture: driver
Solver Convergence: CONVERGED
Kinematic Accuracy: WITHIN_TOLERANCE
Dynamic Feasibility: PHYSICALLY_FEASIBLE
Scientific Qualification: QUALIFIED
Product Promotion: PROMOTED
Whole Marker RMSE: 14.8 mm (Gate: <= 18.0 mm)
Max Error: 28.2 mm
```

---

## 4. Full-Body Engine Reproduction and Environment Requirements

Full-body models require specific engine environments. Missing engines must fail closed without blocking non-full-body operations.

### 1. Simscape (`full_body_simscape`)

- **Explicit Requirement:** MATLAB R2025b with Simscape Multibody.
- **Operating Note:** Historical run receipts are pinned to MATLAB R2025b under `#10440`. Runs from older or newer MATLAB releases trigger runtime version mismatch warnings.
- **Reproduction Command:**
  ```bash
  matlab -batch "addpath('src/engines/physics_engines/simscape'); run_simscape_tour_match('driver'); exit;"
  ```
- **Status:** Driver terminal error 40.3 mm (> 35.0 mm gate); marked as `NATIVE_CANDIDATE` pending final parameter optimization under [#10440](https://github.com/D-sorganization/UpstreamDrift/issues/10440).

### 2. MuJoCo (`full_body_mujoco`)

- **Requirement:** `mujoco>=3.1.0`.
- **Reproduction Command:**
  ```bash
  python -m src.engines.physics_engines.mujoco.fit_swing \
      --capture driver \
      --static-seeds \
      --ground-support
  ```
- **Status:** Driver static calibration passed (5.1 mm address RMS, 27.3 mm IK RMS); forward shooting dynamics under active review in [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363).

### 3. Pinocchio / Crocoddyl (`full_body_pinocchio`)

- **Requirement:** `pinocchio>=3.0.0`, `crocoddyl>=2.0.0`.
- **Reproduction Command:**
  ```bash
  python -m src.engines.physics_engines.pinocchio.run_trajectory_optimization \
      --capture driver \
      --max-iters 100
  ```
- **Status:** Rejected on driver due to dynamic constraint divergence; tracked under [#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378).

### 4. OpenSim Moco (`full_body_opensim`)

- **Requirement:** OpenSim 4.5+ C++ Python bindings.
- **Reproduction Command:**
  ```bash
  python -m src.engines.physics_engines.opensim.run_moco_tracking \
      --model src/engines/physics_engines/opensim/models/golfer.osim \
      --capture driver
  ```
- **Status:** Placeholder model registered; full tracking pending adapter completion under [#10003](https://github.com/D-sorganization/UpstreamDrift/issues/10003).

### 5. MyoSuite (`full_body_myosuite`)

- **Status:** `UNAVAILABLE`.
- **Note:** Bundled `myobody` URDF assets are labeled placeholders. Fail-closed per MS-50 until native neuromuscular environment is integrated.

---

## 5. Baseline Package Export, Import, and Validation

### Package Export in Python:

```python
from pathlib import Path
from src.shared.python.tour_baselines import export_baseline_package

export_baseline_package(
    package=my_package,
    output_path=Path("dist/baselines/driven_double_pendulum_driver.zip"),
    deterministic=True,
)
```

### Package Import & Tamper Verification:

```python
from pathlib import Path
from src.shared.python.tour_baselines import import_baseline_package

pkg = import_baseline_package(
    Path("dist/baselines/driven_double_pendulum_driver.zip")
)
print(f"Loaded: {pkg.identity.model_id} on {pkg.identity.capture}")
print(f"RMSE: {pkg.metrics.whole_marker_rmse_m * 1000:.1f} mm")
```

If any file within the ZIP archive is modified, `import_baseline_package` raises `ValueError: Checksum mismatch for array 'trajectories/q'`, failing closed.

---

## 6. Troubleshooting and Operator Runbook

| Symptom                                                                | Probable Cause                            | Action                                                                                                                                                          |
| ---------------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'src.shared.python.logging_pkg'` | Submodule `vendor/ud-tools` uninitialized | Run `git submodule update --init vendor/ud-tools`                                                                                                               |
| `ValueError: Unknown model_id '...' for capture '...'`                 | Unregistered model alias or typo          | Run `python -c "from src.shared.python.tour_baselines.registry import list_golf_models; print([m.model_id for m in list_golf_models()])"` to list canonical IDs |
| `ChecksumMismatchError` on loading package                             | Package archive tampered or truncated     | Re-export package using deterministic export                                                                                                                    |
| Optimization reaches max evaluations without convergence               | Tight bounds or step size too small       | Increase `--max-evaluations` or review parameter feasibility bounds in `fit_campaign.py`                                                                        |
| UI crashes on `import PyQt6`                                           | Headless Linux display missing            | Run headless or set `export QT_QPA_PLATFORM=offscreen`                                                                                                          |

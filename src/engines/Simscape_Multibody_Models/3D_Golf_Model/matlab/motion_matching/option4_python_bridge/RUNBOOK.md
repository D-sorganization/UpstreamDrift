# Runbook — Option 4

Literal commands for the day-to-day operations on this bridge. Windows-first per [INSTALLATION.md](INSTALLATION.md). Linux/macOS users substitute path separators where obvious.

For first-time install (Python env, matlabengine pip package, MATLAB toolboxes, license setup), see [INSTALLATION.md](INSTALLATION.md).

All commands assume the repo is checked out at `C:\Users\diete\Repositories\UpstreamDrift` and the working directory is the repo root.

## 0. Pre-flight checks

```powershell
# Confirm Python can find the matlabengine package
python -c "import matlab.engine; print('matlabengine OK')"

# Confirm MATLAB starts headlessly without prompting
matlab -nodesktop -nosplash -r "disp('matlab OK'); quit"

# Confirm the .slx model is on disk
ls src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\model\GolfSwing3D_Kinetic.slx
```

If any of those three fail, stop and run [INSTALLATION.md](INSTALLATION.md) end to end.

## 1. Smoke test (issue #4077 surface)

The simplest "is the bridge alive" check uses the test suite that ships with #4077:

```powershell
python -m pytest src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\tests -v
```

Expected output (with MATLAB Engine for Python installed and Simscape Multibody licensed):

```
test_simulate_rejects_bad_theta_shape           PASSED
test_simout_dataclass_is_frozen                 PASSED
test_fit_options_defaults                       PASSED
test_fit_swing_jax_is_explicitly_unimplemented  PASSED
test_fit_swing_scipy_rejects_bad_theta0         PASSED
test_engine_starts_and_stops_cleanly            PASSED  (~5–30 s)
test_get_polynomial_bounds_shape                PASSED
test_simulate_with_coefficients_returns_canonical_simout PASSED  (slow)
test_round_trip_against_direct_matlab_call      PASSED  (slow; ~15 s)
test_recover_theta_from_synthetic_target        PASSED  (slow; ~30–120 s)
```

Without MATLAB Engine for Python on the active interpreter (`pip show matlabengine` returns nothing), every `requires_matlab_engine` test auto-skips with a loud reason pointing at [INSTALLATION.md](INSTALLATION.md). The pure-Python tests (`test_simulate_rejects_bad_theta_shape`, `test_simout_dataclass_is_frozen`, `test_fit_options_defaults`, `test_fit_swing_jax_is_explicitly_unimplemented`, `test_fit_swing_scipy_rejects_bad_theta0`) still run.

If MATLAB Engine for Python is installed but Simscape Multibody is not licensed (`license('test', 'Simscape_Multibody')` returns 0 inside MATLAB), the forward-sim tests skip with a Simscape-license reason and the lifecycle / bounds tests still pass — useful for partial validation in license-constrained CI.

## 2. Manual smoke (one-liner)

```powershell
python -c @"
from option4_python_bridge.simscape_adapter import SimscapeAdapter
import numpy as np
with SimscapeAdapter() as a:
    n = a.get_n_joints(default=28)
    lb, ub = a.get_polynomial_bounds(n)
    theta = np.zeros(lb.size)
    sim = a.simulate_with_coefficients(theta)
    print('grip shape:', sim.grip.shape, 'status:', sim.solver_status)
"@
```

The `option4_python_bridge` package directory must be on `sys.path` (e.g. via `PYTHONPATH=src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching`).

## 3. Run a Python-driven `system_identification` against the adapter

The headline workflow. Requires Options #036, #037, #040 closed.

```powershell
python -c "from pathlib import Path; \
from src.engines.loaders import load_matlab_3d_engine; \
from src.learning.sim2real.system_identification import SystemIdentifier; \
from src.shared.python.data_io.swing_capture_import import SwingCaptureImporter; \
suite_root = Path('.').resolve(); \
engine = load_matlab_3d_engine(suite_root); \
importer = SwingCaptureImporter(); \
demo = importer.import_excel( \
    'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx', \
    sheet='TW_ProV1'); \
identifier = SystemIdentifier(model=engine); \
result = identifier.identify_from_trajectory(demo.trajectory, max_iters=50); \
print(f'Converged: {result.converged}, residual: {result.residual_error:.6f}'); \
engine.close()"
```

(In practice, write this as a script — `scripts/run_sysid_against_simscape.py` — rather than a one-liner.)

## 4. Generate a Simscape parquet shard from Python

Headline workflow #2. Requires #036, #037, #039 closed and `dataset_generator/core.py` accepting a `PhysicsEngineProtocol` argument.

```powershell
python -c "from pathlib import Path; \
from src.engines.loaders import load_matlab_3d_engine; \
from src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.src.functions.dataset_generator.core import generate_shard; \
suite_root = Path('.').resolve(); \
engine = load_matlab_3d_engine(suite_root); \
shard = generate_shard(engine=engine, n_trials=100, seed=42, \
                       output_dir=Path('data/sweeps/option4_smoke')); \
print(f'Wrote {shard}'); \
engine.close()"
```

Wall-clock for 100 trials on a single engine: **~30–60 s** (warm-engine throughput). For 10⁴ trials, use the pool — see § 5.

## 5. Run a parallel sweep with `SimscapeAdapterPool`

Requires #038.

```powershell
python -c "from pathlib import Path; \
from src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.motion_matching.option4_python_bridge.simscape_adapter_pool import SimscapeAdapterPool; \
import numpy as np; \
slx = 'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/GolfSwing3D_Kinetic.slx'; \
rng = np.random.default_rng(42); \
thetas = [rng.uniform(-25, 25, size=16*7) for _ in range(20)]; \
with SimscapeAdapterPool(pool_size=4, model_path=slx) as pool: \
    outs = pool.map_simulate(thetas); \
print(f'Got {len(outs)} simulations')"
```

License pool warning: the pool will block on `__init__` waiting for licenses if the host has fewer than `pool_size` licenses available. Reduce `pool_size` to match the deployment.

## 6. Swap the adapter into a humanoid RL env (eval rollout only)

Requires #036, #037 closed. **Inner-loop training is not supported by Option 4 alone — use a surrogate.**

```python
from pathlib import Path
from src.engines.loaders import load_matlab_3d_engine
from src.learning.rl.humanoid_envs import HumanoidWalkEnv

engine = load_matlab_3d_engine(Path(".").resolve())
env = HumanoidWalkEnv(
    engine=engine,
    model_path="src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/GolfSwing3D_Kinetic.slx",
)

obs, _ = env.reset()
for _ in range(100):  # short eval rollout — 5–20 s wall-clock
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

engine.close()
```

This is the **slow path** (`step()` per call). Long training runs need the surrogate.

## 7. Use the bridge as Options 2 / 3's round-trip oracle

Validate that a surrogate's predicted coefficients reproduce the target via the canonical Simscape forward sim.

```python
from pathlib import Path
import numpy as np
from src.engines.loaders import load_matlab_3d_engine
# (Option 2 / 3 imports omitted — see their RUNBOOK files.)

engine = load_matlab_3d_engine(Path(".").resolve())

# theta_hat came from Option 2 surrogate or Option 3 inverse network
theta_hat: np.ndarray = ...  # shape (n_joints*7,)

simscape_out = engine.simulate_with_coefficients(theta_hat)

# Compare against measured target (the same target the surrogate was given)
position_rmse = np.sqrt(np.mean(
    (simscape_out.clubhead - target.clubhead) ** 2
))
print(f"Round-trip RMSE: {position_rmse*1000:.2f} mm")

engine.close()
```

If the round-trip RMSE is much larger than the surrogate's predicted RMSE, the surrogate is extrapolating — reject the fit.

## 8. Regenerate the latency / throughput / cache visualizations

```powershell
python -m src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.motion_matching.option4_python_bridge.visualization.benchmark_pool_throughput
python -m src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.motion_matching.option4_python_bridge.visualization.dashboard
```

Outputs go to `option4_python_bridge/visualization/output/` (gitignored).

## 9. Cleanup: kill orphan MATLAB processes

If a test crashed and left an engine running:

```powershell
Get-Process matlab -ErrorAction SilentlyContinue | Stop-Process -Force
```

The test fixtures use try/finally + `__exit__` to avoid this, but the safety net is occasionally useful.

## 10. Diagnose engine startup failures

If `python -c "import matlab.engine; matlab.engine.start_matlab()"` hangs or fails:

1. Check the MATLAB activation: `matlab -nodesktop -nosplash -r "license('inuse'); quit"` should print at least `MATLAB`, `Simulink`, `Simulink_Toolbox` (the Simscape Multibody license).
2. Check the `matlabengine` Python package version matches MATLAB: `python -c "import matlab.engine; print(matlab.engine.__version__)"` and `matlab -batch "ver"`.
3. Check the Python interpreter version is on the MathWorks compatibility matrix: `python --version`. See [INSTALLATION.md § version-pinning](INSTALLATION.md#version-pinning).
4. Try a longer startup timeout: `SimscapeAdapter(startup_timeout_s=120)`.
5. As a last resort, reinstall: `python -m pip uninstall matlabengine` then re-run [INSTALLATION.md](INSTALLATION.md) § 3.

## 11. Useful environment variables

| Variable                          | Purpose                                                                 | Default                                          |
| --------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------ |
| `GOLF_SWING_3D_SLX_PATH`          | Override the .slx path used by the test fixtures.                       | `<repo>/src/engines/.../GolfSwing3D_Kinetic.slx` |
| `SIMSCAPE_ADAPTER_TIMING`         | If `1`, every adapter call appends to `_timing_log`.                    | unset                                            |
| `SIMSCAPE_ADAPTER_CACHE_DISABLED` | If `1`, force `cache_enabled=False` regardless of constructor argument. | unset                                            |
| `MATLAB_ENGINE_LICENSE_TIMEOUT_S` | Per-engine license-checkout timeout.                                    | `60`                                             |

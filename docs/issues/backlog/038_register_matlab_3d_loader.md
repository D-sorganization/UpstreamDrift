# Issue: Register load_matlab_3d_engine in src/engines/loaders.py (Option 4)

## Summary

Wire `SimscapeAdapter` into the existing engine-registry pipeline:
implement `load_matlab_3d_engine` and add it to `LOADER_MAP` keyed by
`EngineType.MATLAB_3D`. After this, `EngineRegistry.create("MATLAB_3D")`
returns a working `SimscapeAdapter`.

## Motivation

See `motion_matching/option4_python_bridge/INTERFACES.md` §"Loader function"
and §"Registration patch for `src/engines/loaders.py`". This is what makes
Option 4 "just another engine" from the perspective of `system_identification`,
`domain_randomization`, and the rest of the existing fleet.

## Dependencies

- #036 (skeleton) — adapter class exists.
- #037 (simulate) — adapter actually works (loader smoke test calls it).

## File targets

- New: `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\loader.py`
- Modify: `C:\Users\diete\Repositories\UpstreamDrift\src\engines\loaders.py` (add `load_matlab_3d_engine` and entry to `LOADER_MAP`)
- New: `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\test_loader.py`
- New: `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\test_loaders_registry_integration.py`
- Modify: `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\INSTALLATION.md` (link from `loaders.py` docstring)

## Public API

Verbatim from `INTERFACES.md`:

```python
from pathlib import Path
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.data_io.common_utils import GolfModelingError

DEFAULT_SLX_RELPATH = Path(
    "engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/"
    "GolfSwing3D_Kinetic.slx"
)


def load_matlab_3d_engine(suite_root: Path) -> PhysicsEngine:
    """Factory for SimscapeAdapter wired into the registry as MATLAB_3D.

    Postcondition (DbC): returned engine is non-None.

    Raises:
        GolfModelingError: matlabengine is not installed, license missing,
            or the default .slx is not on disk.
    """
```

`src/engines/loaders.py` adds:

```python
def load_matlab_3d_engine(suite_root: Path) -> PhysicsEngine:
    """Load Simscape Multibody (MATLAB_3D) engine via the Python bridge.

    Postcondition: returned engine is non-None (DbC).
    """
    from src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.\
        motion_matching.option4_python_bridge.loader import (
        load_matlab_3d_engine as _load,
    )
    return _load(suite_root)


LOADER_MAP[EngineType.MATLAB_3D] = load_matlab_3d_engine
```

## Required tests (TDD)

- `test_loader_returns_simscape_adapter_for_valid_suite_root`
- `test_loader_returns_engine_with_non_empty_model_name_post_load`
- `test_loader_raises_GolfModelingError_when_matlabengine_not_installed`
- `test_loader_raises_GolfModelingError_when_default_slx_missing_from_suite_root`
- `test_loader_postcondition_returned_engine_is_non_none`
- `test_loader_smoke_calls_simulate_with_coefficients_on_returned_engine`
- `test_loader_logs_info_message_on_successful_load`
- `test_loaders_registry_LOADER_MAP_contains_MATLAB_3D_key`
- `test_engine_registry_create_MATLAB_3D_returns_working_simscape_adapter`
- `test_engine_registry_works_inside_pytest_marker_live_simulation`

Live tests marked `@pytest.mark.live_simulation`; offline tests stub
`SimscapeAdapter` with `unittest.mock`.

## DbC contract

Preconditions:

- `suite_root.exists()`.

Postconditions:

- Returned engine is non-None.
- `engine.model_name != ""`.

## Acceptance Criteria

- [ ] `load_matlab_3d_engine` exists in `src/engines/loaders.py` and in
      `option4_python_bridge/loader.py`; the former delegates to the latter.
- [ ] `LOADER_MAP[EngineType.MATLAB_3D]` is wired up.
- [ ] `EngineType.MATLAB_3D` already exists in `engine_registry.py` (line 23) —
      verified, no enum change needed.
- [ ] `__all__` in `loaders.py` updated to include `load_matlab_3d_engine`.
- [ ] All listed tests pass.
- [ ] DbC decorators applied; postconditions enforced.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines (loaders.py is currently large; check budget).
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option4`, `python`, `infra`, `tdd`, `dbc`

## Effort estimate

S (≤1 day) once #037 lands.

# Coding Standards — Motion Matching

These are non-negotiable rules. CI enforces them. PRs that don't comply will be sent back.

## TDD (Test-Driven Development)

**Every PR includes the test in the same diff as the implementation.** Tests are written **first** and committed first when feasible. Coverage must not decrease.

### MATLAB

Use `matlab.unittest.TestCase`. Tests live alongside the source:

```
option1_direct_optimization/
├── fit_swing_fmincon.m
└── tests/
    └── test_fit_swing_fmincon.m
```

Skeleton:

```matlab
classdef test_fit_swing_fmincon < matlab.unittest.TestCase
    methods (Test)
        function fits_synthetic_swing_to_within_tolerance(testCase)
            target = synthesize_swing_from_known_coefficients();
            result = fit_swing_fmincon(target, default_options());
            testCase.verifyLessThan(result.final_rmse_m, 0.01);
        end

        function rejects_target_with_wrong_dimensions(testCase)
            bad = struct('butt', zeros(10,2));  % wants Nx3
            testCase.verifyError( ...
                @() fit_swing_fmincon(bad, default_options()), ...
                'MATLAB:validation:IncompatibleSize');
        end
    end
end
```

Run from the repo root:

```matlab
results = runtests('motion_matching/option1_direct_optimization/tests');
```

Do **not** use the legacy `functiontests` form for new code.

### Python

`pytest` with markers from [pyproject.toml](../../../../../../../pyproject.toml). Place tests under `tests/motion_matching/` mirroring the source layout.

```python
import pytest

@pytest.mark.unit
def test_surrogate_predicts_within_tolerance(trained_surrogate, validation_batch):
    pred = trained_surrogate(validation_batch.coeffs)
    err = np.mean(np.abs(pred - validation_batch.kinematics))
    assert err < 0.01
```

## DbC (Design by Contract)

Pre/post conditions are explicit and checked at runtime — not buried in docstrings.

### MATLAB

Preconditions go in an `arguments` block:

```matlab
function result = fit_swing_fmincon(target, options)
    arguments
        target (1,1) struct {mustHaveFields(target, ["butt","clubhead","time"])}
        options (1,1) struct = default_options()
    end
    % ... body ...
    assert(isfield(result, "final_rmse_m"), ...
        "Postcondition: result must contain final_rmse_m");
    assert(result.final_rmse_m >= 0, ...
        "Postcondition: RMSE must be non-negative");
end
```

Custom validators live in `shared/+validators/` (a MATLAB package folder):

```matlab
function mustHaveFields(s, names)
    missing = setdiff(names, string(fieldnames(s)));
    if ~isempty(missing)
        error("validator:missingField", ...
              "Struct missing required fields: %s", strjoin(missing, ", "));
    end
end
```

### Python

Use the existing decorators:

```python
from src.shared.python.core.contracts import precondition, postcondition, invariant

@precondition(lambda coeffs: coeffs.ndim == 1, "coeffs must be 1-D")
@postcondition(lambda result: result.shape[1] == 3, "output must be Nx3")
def simulate_clubhead(coeffs: np.ndarray) -> np.ndarray:
    ...
```

## DRY

No duplicated logic blocks > 5 lines. If two options need the same operation:

1. The implementation goes under `motion_matching/shared/` (MATLAB) or `src/shared/python/motion_matching/` (Python).
2. Both options import or call it.
3. PRs that copy-paste from another option without consolidating to `shared/` will be rejected.

The cost function, club IK, dataset loader, and visualization helpers are **explicitly shared**. See [README.md](README.md).

## LOD (Law of Demeter)

No method chains deeper than 2 levels.

```matlab
% Bad:
result.swing.club.butt.position(1)

% Good (add a delegating accessor):
butt_x = get_butt_x(result);
```

```python
# Bad:
config.engine.model.body[3].mass

# Good:
mass = config.get_body_mass(3)
```

## File size

1200 lines max per `.m` or `.py`. If you're approaching the limit, **stop and refactor before adding more**. Helpers, formatting, and visualization functions belong in their own files.

## Naming

- MATLAB functions and files: `snake_case` for new code under `motion_matching/` (matches Python style and the existing dataset_generator code).
- Classes: `PascalCase` (MATLAB classdef and Python class).
- Private helpers in MATLAB: prefix with `private/` folder (auto-scoped) or use `+package` packages.
- Public types in `shared/`: documented in [README.md](README.md).

## Logging

- MATLAB: use `fprintf` with a verbosity check from `options.verbosity` (`'Silent' | 'Normal' | 'Verbose' | 'Debug'`). Do **not** use `disp` in production code paths.
- Python: `from src.shared.python.logging_pkg.logging_config import get_logger`. Never use `print()` in `src/`.

## Provenance and reproducibility

Every fit run produces a result struct/dict containing at minimum:

```
result
├── coefficients          (n_joints × 7 double)
├── final_rmse_m          (scalar, club-tip RMSE in metres)
├── final_total_work_J    (scalar, sum of |τ·ω|·dt across joints)
├── solver                (string: "fmincon" | "surrogateopt" | ...)
├── solver_options        (struct: full options used)
├── target_hash           (sha256 of the target trajectory; for cache invalidation)
├── git_commit            (string)
├── matlab_version        (string)
├── duration_s            (scalar wall-clock)
└── timestamp_utc         (string ISO-8601)
```

This is consumed by the visualization dashboard and by leaderboard comparison across the four options.

## Docs-currency policy

The top-level docs in this tree are the source of truth for scope and architecture. They drift fast if PRs don't keep them in sync.

**Rule:** every PR that meaningfully changes scope or architecture under `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/` must touch at least one of:

- [`PROJECT_SPEC.md`](../../PROJECT_SPEC.md)
- [`MATLAB_GOLF_MODEL_GUIDE.md`](../../matlab/MATLAB_GOLF_MODEL_GUIDE.md)
- [`GRIP_FIT_PLAYBOOK.md`](GRIP_FIT_PLAYBOOK.md)

**Opt-out:** if the change genuinely doesn't affect scope or architecture (a typo fix, a test-only change, a refactor with no behavioural delta), include the marker `[no-docs-needed]` in the PR description.

**Enforcement:** `.github/workflows/docs-currency-warning.yml` posts an advisory comment on PRs that touch this tree without touching docs/tests and without the opt-out marker. The check is **advisory only** — it does not block merges. Reviewers may still request docs updates.

"Meaningfully changes scope or architecture" means anything a future contributor would want to read about in the top-level docs: new options, new shared interfaces, changed cost-function semantics, changed dataset schema, new external dependencies, etc. Pure bug fixes, performance tweaks, and test additions do not require a docs touch.

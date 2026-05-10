# test(motion-matching): bring `src/shared/python/motion_matching/` to ≥85% line coverage

## Goal

Raise unit-level line coverage of `src/shared/python/motion_matching/` to **≥85% line, ≥75% branch**. Test-only PR — no production code changes.

## Current state

74 production `.py` files, 61 test files. Sub-areas with thin coverage are typically:
- `surrogate/` (model + invert + validate)
- `inverse/` (regressor, cvae, training, predict)
- `inverse_timestep/` (filter, model, predict, training)
- `dataset/` (sweep, synthetic, _validate)
- `hybrid.py`, `engine_init_profiler.py`, `align_to_simulation_grid.py`, `final_cost.py`, `validators.py`, `validate_theta.py`
- `loaders/` private helpers (`_align.py`, `_marker_clusters.py`, `_quaternion.py`, `_machinelearning_compat.py`)

## Process

1. Baseline:

   ```bash
   python3 -m pytest tests/unit/motion_matching/ -p no:cacheprovider \
     --cov=src/shared/python/motion_matching \
     --cov-report=term-missing --cov-branch -n auto --timeout=60 \
     2>&1 | tee coverage_baseline_motion_matching.txt
   ```

2. From the term-missing output, list every file under 85%. Focus on highest-impact files (smallest delta to 85% × highest line count) first.

3. Add unit tests in `tests/unit/motion_matching/`. One file per production file when natural. For very small modules, share a test file with siblings in the same subpackage.

4. For each uncovered line group, write one test that exercises that path, with a clear `"""Pin: <one-line description>"""` docstring identifying which uncovered line/branch it covers.

5. Use synthetic data, `pytest.fixture` for shared setup, and `numpy.testing.assert_allclose(rtol=…, atol=…)` with sensible tolerances.

6. Re-run coverage; iterate until target met. Final coverage report goes in the PR body.

## Constraints

- **Test-only PR.** If you find a real bug, file a separate issue with a failing test marked `xfail(reason=...)` and continue.
- DbC pre/post-conditions on production code are a feature; failing tests against malformed inputs ARE the right tests to write.
- Generic naming policy. No vendor / lab / person names anywhere.
- mypy + ruff + file-size budget clean.

## Out of scope

- Heavy-integration tests requiring real engine wheels — those have their own `tests/heavy_integration/` slot.
- The `motion_pipeline` subdir (different ownership; covered separately).
- Any production behaviour change.

## Files touched

- New / extended: `tests/unit/motion_matching/test_*.py`
- New / extended: `tests/unit/motion_matching/{surrogate,inverse,inverse_timestep,dataset,loaders}/test_*.py`

## Acceptance

- [ ] `pytest tests/unit/motion_matching/ --cov=src/shared/python/motion_matching --cov-report=term-missing --cov-branch` reports **≥85% line, ≥75% branch**.
- [ ] PR body lists per-file coverage delta (before / after).
- [ ] All new tests pass; no production code changes.
- [ ] mypy + ruff + file-size budget clean.

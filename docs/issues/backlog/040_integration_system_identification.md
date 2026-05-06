# Issue: Integration Test — End-to-End system_identification.py with SimscapeAdapter (Option 4)

## Summary

Add an integration test that exercises the existing
`src/learning/sim2real/system_identification.py` module against the
`SimscapeAdapter` registered in #038. The test confirms Option 4 is a true
plug-in to the existing fleet — not a parallel branch.

## Motivation

See `motion_matching/README.md` "Why four options in parallel" — Option 4's
selling point is "reuses existing system_identification, RL, retargeter stack".
Without this integration test that claim is unverified. The test is what closes
the Option 4 epic.

## Dependencies

- #036 (skeleton).
- #037 (working simulate).
- #038 (registry wiring).
- #039 (pool, optional — if absent, the test runs serial).

## File targets

- New: `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\integration\test_system_identification_integration.py`
- New: `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\integration\test_simscape_with_domain_randomization.py`
- New: `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\integration\fixtures\sysid_target_synthetic.json` (synthetic target captured from a known mass-set perturbation)
- Maybe modify: `C:\Users\diete\Repositories\UpstreamDrift\src\learning\sim2real\system_identification.py` (only if the test exposes a contract gap; do **not** make changes that aren't motivated by a test)

## Public API

This issue does not introduce new public API; it consumes:

```python
from src.learning.sim2real.system_identification import (
    SystemIdentifier,
    SystemIdentificationConfig,
)
from src.engines.loaders import load_matlab_3d_engine

engine = load_matlab_3d_engine(suite_root)
identifier = SystemIdentifier(engine, config=SystemIdentificationConfig(...))
result = identifier.identify(target_kinematics)
```

The test shape is:

```python
@pytest.mark.live_simulation
@pytest.mark.integration
def test_system_identification_recovers_known_mass_perturbation_via_simscape():
    suite_root = Path(__file__).parents[5]
    engine = load_matlab_3d_engine(suite_root)
    truth_masses = engine.get_link_masses()
    perturbed = truth_masses * np.array([1.0, 1.05, 0.95, ...])  # known shift
    engine.set_link_masses(perturbed)
    target = engine.simulate_with_coefficients(theta_nominal)
    engine.set_link_masses(truth_masses)  # reset
    identifier = SystemIdentifier(engine, default_sysid_config())
    result = identifier.identify(target)
    np.testing.assert_allclose(result.identified_masses, perturbed, rtol=0.05)
    engine.close()
```

## Required tests (TDD)

- `test_system_identification_recovers_known_mass_perturbation_via_simscape`
- `test_system_identification_recovers_known_damping_perturbation_via_simscape`
- `test_system_identification_handles_simscape_simulation_error_with_retry`
- `test_system_identification_completes_within_acceptable_wall_time_serial`
- `test_system_identification_completes_within_acceptable_wall_time_with_pool`
- `test_simscape_engine_passes_protocol_compliance_used_by_system_identification`
- `test_simscape_with_domain_randomization_runs_n_episodes_without_engine_leak`
- `test_simscape_engine_releases_matlab_license_on_close_after_integration_test`
- `test_integration_test_skipped_gracefully_when_matlab_engine_unavailable`
- `test_integration_uses_load_matlab_3d_engine_from_loaders_not_a_direct_import`

## DbC contract

This issue does not add new DbC decorators; it relies on those from
#036–#039. The test asserts they are not violated end-to-end.

## Acceptance Criteria

- [ ] At least two integration tests pass under `pytest -m live_simulation`.
- [ ] Tests skip gracefully (with informative message) when MATLAB Engine is
      unavailable; CI-without-MATLAB still passes.
- [ ] Domain randomization round-trip verified (engine accepts sequential
      `set_link_masses` calls without leaking memory or resetting the cache
      incorrectly).
- [ ] Wall-time budget recorded for serial and pooled paths; results in test
      output.
- [ ] If a contract gap in `system_identification.py` is exposed, file a
      follow-up issue rather than expanding scope here.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option4`, `python`, `infra`, `tdd`

## Effort estimate

M (1-3 days) once #036–#039 land.

# CI Optional Stack Runbook (#14227)

This note records what is **mechanically fixed on `main`** versus what still
needs an **owner decision** in the scheduled
[CI Optional Stack](https://github.com/D-sorganization/UpstreamDrift/blob/main/.github/workflows/ci-optional-stack.yml)
lane.

## Slice 1 — Pinocchio Gravity Matches the Shared Constant (Mechanical, on Main)

Issue [#13799](https://github.com/D-sorganization/UpstreamDrift/pull/13799) moved
the CasADi swing model to the repository gravity constant
`GRAVITY_M_S2` (`src/shared/python/core/physics_constants.py`). Pinocchio's
builder previously kept the library default (9.81 m/s²), so symbolic RNEA checks
and `pin.rnea` disagreed in Optional Stack trajectory-optimization steps.

**Fix (merged; completes Pinocchio's side of #13799):**

- `build_pinocchio_model` in `src/shared/python/optimization/model_provider.py`
  sets `model.gravity.linear = [0, 0, -GRAVITY_M_S2]` after
  `pin.buildModelFromXML(...)`.

**Regression lock:**

- `test_pinocchio_model_uses_the_shared_gravity_constant` in
  `tests/unit/optimization/test_model_provider.py` (hermetic mock of pinocchio).

CI contract tests in `tests/ci/test_optional_stack_gravity_contract_14227.py`
assert that this runbook, the builder, the constant, and the regression test
stay wired.

## Slice 2 — Manufactured Authority and Rolling (Mechanical, on Main)

Optional Stack jobs `articulated-manufactured-authority` and
`articulated-manufactured-rolling` regressed when spatial model canonical hashes
drifted (#13039 / #13839). **Agents must not fabricate or silently re-commit
authority artifacts** to turn CI green.

**Fix (merged on `main`):**

- [#15069](https://github.com/D-sorganization/UpstreamDrift/pull/15069) stabilizes
  spatial canonical hashing so deterministic authority candidates match the
  committed publication authority again.
- Scheduled run
  [43383564368](https://github.com/D-sorganization/UpstreamDrift/actions/runs/43383564368)
  (2026-11-13) and later show both manufactured jobs **green** on `main`.

If the authority lane regresses again, regeneration from the pinned environment
in #14227 remains the owner publication path — not an agent baseline swap.

## Related — Multiple-Shooting Dynamics Premise (#14227 Item 3)

After gravity alignment (#13799), the default six-node shooting benchmark no
longer exceeds the historical adaptive floors from #9830 (Optional Stack
2026-11-12 measured ~0.023 rad position and ~1.37 rad/s velocity). The live
test re-picks material floors (0.01 rad, 1.0 rad/s) plus own-grid separation
factors instead of the pre-#13799 absolute 0.1 rad / 10 rad/s ceilings.
Further changes need a fresh measurement on current `main`, not blind threshold
widening.

## Slice 3 — Optional-Stack Launcher Processes API Mock Contract (Mechanical)

Commit #14956 (`Part of #14949`) updated `LauncherService.get_running_processes()`
in `src/api/services/launcher_service.py` to prefer
`process_manager.list_processes()` when callable, which broke un-spec'd
`MagicMock` `ProcessManager` fixtures in `tests/api/test_launcher_launch_api.py`
and `tests/unit/api/services/test_launcher_service.py` (an auto-created
`MagicMock.list_processes()` returned an empty iterable and shadowed
`running_processes`), causing 5 Optional Stack API test failures in
`TestProcessesEndpoint` and `TestLaunchLifecycle`.

**Fix (merged on `main` via [#15089](https://github.com/D-sorganization/UpstreamDrift/pull/15089)):**

- `LauncherService.get_running_processes()` only ingests sequence rows from
  `list_processes()` whose `slot_name` is a `str`, and populates any remaining
  entries from `running_processes` without re-polling slots already returned by
  `list_processes()`.
- Regression coverage in `tests/unit/api/test_launcher_service_processes_14227.py`
  and existing tests in `tests/api/test_launcher_launch_api.py` and
  `tests/unit/api/services/test_launcher_service.py`.

## Scheduled Lane Status (Evidence on `main`)

| Run | `optional-stack-check` | `articulated-manufactured-authority` | `articulated-manufactured-rolling` |
| --- | --- | --- | --- |
| [43688504401](https://github.com/D-sorganization/UpstreamDrift/actions/runs/43688504401) (2026-11-15, scheduled) | Green — post-#15089 `#14227` slice 3 on `main` (`d7bdbc569` at schedule time) | Green | Green |
| [43569598549](https://github.com/D-sorganization/UpstreamDrift/actions/runs/43569598549) (2026-11-14) | Red — 5 API failures in `test_launcher_launch_api.py` (pre-#15089 merge) | Green | Green |

Scheduled acceptance for #14227 is the 2026-11-15 green run above. Earlier
failures are retained as history only.

The run metadata and all three lane checkout logs identify source commit
`d7bdbc569964ca2f85af6f3f1e05bd0362b87e7b`. Acceptance remains scoped to that
historical source commit.

# F09k Owned SDK Forecast Turnover

Issue #12009; parent #11793 and epic #11784. Branch
`feat/f09k-owned-sdk-forecast-12009` starts at F09j
`43e925dd7171f12593d6c8149e3468c720048bc7`, draft PR #12005. The parent and exact
Tools feature dependency remain separate from actual main acceptance.

## Implementation and Contracts

`myosuite_project_forecast.py` exposes `ProjectTaskForecaster` and
`ProjectTaskForecast`. Allocate one owned source-identical SDK task, then restore
the entire current integration state before each candidate. Reuse
`record_project_task_commands`; no second SDK stepping loop is introduced.
Construction and prediction pin model/data handles, compiled model, complete
source closure and SDK lineage. Guard observation validation and numeric command
conversion before owned execution and verify live preservation on every exit.
Current-state consistency does not authenticate snapshot origin.

Nonzero absolute epochs, activation, controls and warmstart must survive exact
restoration, including native forward preparation. A→B→A and failed-candidate→A
must equal a freshly owned A. Concurrent/reentrant use rejects. Idempotent close
owns only the candidate; returned candidates are closed on constructor admission
failure. No new guarantee covers failures internal to the existing factory before
it returns. Callers retain ownership of and serialize mutations to the live task.
Failure detects mutations without rollback or successful partial history.

Adapter source provenance is separate from producer, controller and replay
identity. `construction_seconds` and `last_attempt_seconds` include admission and
restoration overhead; failed attempts are timed. Replay cost remains separate.
This is an offline shooting primitive without an optimizer or deadline claim.

## TDD and Evidence

Actual pinned MyoSuite 3.0/MuJoCo 3.6 first produced ten API-absence failures,
zero skips, retained as `f09k-native-api-red.xml` in workspace planning evidence.
The first implementation campaign passes 16 actual SDK cases with zero skips
in 12.00 seconds (`f09k-native-first.xml`). These include full-state warmstart
repeatability, stale/inconsistent observations, changed compiled models,
deferred-conversion live mutation, invalid-candidate recovery, reentrancy,
post-close refusal and constructor-owned cleanup. A lookahead feedback fixture
exports and independently replays its complete history with feedback disabled.
Later review regressions and the final combined campaign remain separate from
these historical first-attempt reports.

Astra identified a replay-scope gap: producer forecasts permitted nonzero
external loads that independent replay refuses. Two actual SDK regressions
failed before repair. Reusing the existing replay restoration/admission check
now refuses those loads before any owned reset or step. The complete five-module
campaign passes 109 tests, zero failures/errors/skips, in 39.89 seconds. This
includes original-source driver and iron suffix forecasts with nonzero epoch,
activation and warmstart, independently replayed complete states and commands.
All twelve executed source/test hashes match this checkout. Retain
`f09k-external-load-red.xml`, `f09k-native-full-campaign-final.xml` and
`f09k-native-source-hashes-final.json` separately. The earlier 109-pass campaign
predates a two-local-variable LoD refactor and is preserved as historical evidence.

## Reproduction and Remaining Acceptance

Use the reviewed MyoSuite 3.0/MuJoCo 3.6 environment and exact vendored Tools
revision, with `--noconftest -o addopts=` to avoid repository SDK mocks. Add
`tests/unit/engines/myosuite/test_project_task_forecast.py` to chapter 38's existing
four-module actual SDK campaign. Retain JUnit, runtime and every executed source
and test hash. Portable skips establish no native availability.

Qualified capture marker attachments, lab/ground registration, uncertainty,
independent holdout, contact/grip readiness and physiological model validation
remain open. Do not convert a short probe into a fitted swing or a muscle-only
claim. Maintain all 17 production rows, six ecosystems and the ultimate muscular
OpenSim full-private-capture, full-state, full-horizon excitation/contact replay.

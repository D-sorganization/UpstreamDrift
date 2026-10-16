# Functional Test Matrix and Computer-Control Test Plan

## Purpose and Scope

This is the repeatable, release-facing manual functional test plan for
UpstreamDrift. The **PyQt6 launcher and native PyQt6 tools are the primary
product under test**. The React/Tauri shell is a secondary parity surface: it
must be compared with verified PyQt6 behavior, but it cannot define the
expected product behavior while it remains behind the native launcher. The
plan is a complement to automated tests; it records what a user can actually
open, operate, and observe.

Items marked `experimental`, `beta`, or `deprecated` are tested for honest and
safe behavior; they are not held to the same release bar as `ready` features.
Optional engines and third-party tools may be recorded as **blocked by an
unavailable dependency**, never as passed.

### Inventory Sources — There Are Two, and They Disagree

| Surface                      | Source of truth                                             | Consumers                                       | Matrix IDs        |
| ---------------------------- | ----------------------------------------------------------- | ----------------------------------------------- | ----------------- |
| Native PyQt6 launcher tiles  | `src/config/models.yaml` via `ModelRegistry`                | `src/launchers/launcher_orchestrator.py`        | `NAT-*`           |
| Shared/web launcher tiles    | `src/config/launcher_manifest.json` via manifest loader     | `/api/launcher/*`, `ui/src/pages/Dashboard.tsx` | `NAT-*` + `MAN-*` |
| Provider model-pack children | sibling `*_Models/model_pack.yaml` repos, merged at runtime | both                                            | `PRV-*`           |
| Web routes                   | `ui/src/App.tsx`                                            | React/Tauri shell                               | `WEB-*`           |
| Backend routes               | `src.api.local_server.create_local_app()`                   | both shells                                     | `API-*`           |
| Parity ledger                | `src/config/feature_parity.json`                            | CI gate `tests/config/feature_parity/`          | `PAR-*`           |

Measured on `origin/main` (`c523c07b5`) with all sibling model-pack repos
present: **75 native models, 75 shared-manifest tiles, only 51 ids in common**
(24 native-only, 24 web-only), 17 React routes, 310 mounted backend routes
(154 distinct non-`/api/v1` paths), 105 launcher menu entries, 37 parity
entries. Regenerate all of these before every campaign:

```bash
QT_QPA_PLATFORM=offscreen python3 scripts/testing/enumerate_test_surface.py --json
```

The tile inventory is **environment-dependent**: provider children only exist
when the sibling `*_Models` repos resolve. Record the measured counts in `PF-06`
and treat a change in count as an inventory change requiring new matrix rows,
not as a test failure.

## Test Execution Rules

1. Start with a clean product process set and record the commit SHA, OS,
   product build, backend URL, and installed optional dependencies.
2. Use only disposable fixture data. Do not upload real participant data or
   credentials. Create outputs in a per-run temporary workspace.
3. Record one result per matrix ID: `PASS`, `FAIL`, `BLOCKED`, or `NOT RUN`.
   A pass needs the stated observable result; opening a screen alone is not a
   pass for an action.
4. On failure, capture the exact repro steps, expected/actual result, error
   text, screenshot, product/build information, and diagnostic logs. Check a
   second time from a clean state before filing, unless it crashes or risks
   data loss.
5. File each confirmed defect as a GitHub issue linked to the current
   functional-testing epic. Do not close a defect merely because it cannot be
   reproduced on a different optional-engine configuration.
6. Update the execution record in this document after every test campaign and
   link the epic and all newly filed defects. A release candidate cannot claim
   functional sign-off while any `ready` critical-path item is failed or not
   run without an explicit waiver.
7. Execute `QT-*` and native-tool rows before `WEB-*`. For every relevant
   React/Tauri observation, identify the equivalent PyQt6 workflow and record
   it as `parity`, `gap`, or `not comparable`; a web-only limitation is a
   product-parity gap, not a reason to weaken the PyQt6 expectation.

## Preflight and Fixtures

| ID    | Precondition / fixture                                                                                                                                                                                                                                                                                                                                                                                                                                   | Expected result                                                                                                                                                                      |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PF-01 | Fresh checkout/build; backend and UI start commands from the current developer guide.                                                                                                                                                                                                                                                                                                                                                                    | UI is reachable, no startup crash, version/build is recorded.                                                                                                                        |
| PF-02 | Browser/Tauri shell and PyQt launcher available.                                                                                                                                                                                                                                                                                                                                                                                                         | Each process has a visible window and is responsive.                                                                                                                                 |
| PF-03 | Baseline fixture set: `tests/data/motion_pipeline/golden/sample.c3d` (`SHA-256 1c412b512384a9b43694b41067e67a5a055e352b4d3d77fff6e52aea0f2115df`), `tests/fixtures/models/simple_pendulum.urdf` (`8233d6e138ed371752037aaf89a727983b11dd2dab12f66833a4695303af5442`), and a disposable CSV copied from `scratch/functional-qa-2026-07-24/data-explorer-fixtures/swing_metrics.csv` (`5aec182f7c74384b001993478a8684ae8c229159ab8a1d9a15410697ecfe5651`). | Fixtures are readable and their path/checksum is recorded before use; the CSV is only an invalid-input fixture, never a C3D substitute.                                              |
| PF-04 | Dependency probe for MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite, MATLAB/Simscape, GPU, camera/media tools.                                                                                                                                                                                                                                                                                                                                              | Each is classified `available`, `missing`, or `broken`; `broken` is a defect.                                                                                                        |
| PF-05 | Error-path fixtures: unsupported file extension, malformed small CSV/JSON, and unreachable backend URL.                                                                                                                                                                                                                                                                                                                                                  | Errors are actionable, recovery is possible, and the UI does not crash.                                                                                                              |
| PF-06 | Run `scripts/testing/enumerate_test_surface.py --json > $EVIDENCE/surface.json` before any other row.                                                                                                                                                                                                                                                                                                                                                    | Exit 0. Record native tile count, shared tile count, provider-pack repos resolved, React route count, backend route count. Any id in `surface.json` with no matrix ID fails `SH-12`. |
| PF-07 | Record which sibling model-pack repos resolved (`MuJoCo_Models`, `Drake_Models`, `Pinocchio_Models`, `OpenSim_Models`, `Tools`, `Movement_Optimizer`).                                                                                                                                                                                                                                                                                                   | Each is `resolved` or `absent`. `PRV-*` rows are `NOT RUN` (not `BLOCKED`) for absent packs; a resolved-but-unloadable pack is a defect.                                             |
| PF-08 | Start the backend exactly as documented (`python3 launch_upstream_drift.py --api-only`) and poll `GET /api/health` and `GET /readyz` until 200, max 120 s.                                                                                                                                                                                                                                                                                               | Both return 200 within the budget. A non-200 `/readyz` with a 200 `/api/health` is a defect, not a wait condition.                                                                   |
| PF-09 | Confirm the port contract: the documented API port and the Vite proxy target in `ui/vite.config.ts` are the same value.                                                                                                                                                                                                                                                                                                                                  | Ports match with no manual override. Divergence is the `#8076` class of defect and blocks all `WEB-*` rows.                                                                          |

## Plan Layout

The matrix, agent script, execution record and ledgers live in separate files
under `docs/testing/functional-test-plan/` so each part stays under the documentation size budget:

- [Functional Matrix: Shell, Launcher, Web, and API](functional-test-plan/01_functional_matrix_shell_launcher_web_api.md)
- [Launcher Tools and Desktop Product Inventory](functional-test-plan/02_launcher_tools_and_product_inventory.md)
- [Computer-Control Agent Script and Issue Templates](functional-test-plan/03_computer_control_agent_script.md)
- [Execution Record](functional-test-plan/04_execution_record.md)
- [Defect Verification Ledger](functional-test-plan/05_defect_verification_ledger.md)
- [Coverage Ledger and Improvement Backlog](functional-test-plan/06_coverage_ledger_and_backlog.md)

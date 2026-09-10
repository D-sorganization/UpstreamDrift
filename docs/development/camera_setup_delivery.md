# Native Camera Setup Delivery

PR #9954 addresses issue #9952, child of #9906, implements the missing fresh-install recording setup.
The header and first wizard page open the same native editor. Existing rig plan,
USB topology, mode parsing, connection validation and atomic document APIs remain
canonical. Each save creates a new plan in the configured capture library.
Camera identity and controls survive offline editing and rescans. No JSON editing
is required, and imported-video routes do not require camera setup.

## Validation

- 563 Capture Rig tests pass on Python 3.12 after regenerating shared help.
- 16 atlas/workflow and 15 goal tests pass; six source modules pass configured mypy.
- Ruff passes; documentation and source-module budgets pass.
- Native editor at 640 by 540 pixels was visually inspected with three camera rows.
- Timeout/cancellation tests preserve existing choices and offer scan recovery.

## Remaining Acceptance

The frontend-inclusive wheel from 79b90700a passes installed editor save/select/
reopen without checkout samples and preserves the current capture. Dependency
checks pass. The rebuilt packaged map includes Camera Setup. SHA256:
`b9b4a964bd24a24e39b2bede95d2faaebe8ba0ac9be07e68f46fb3dff2631fb5`.
Complete CI and protected merge. Live camera mode support and calibration accuracy
remain separate hardware acceptance; no synthetic test qualifies physical accuracy.
The prerequisite installed-runtime PR is #9950 at d5f44f211. Its standard CI
passes; scalar optimization bounds fail separately under #9953. A missing-file
research CI job passed on rerun; no workspace-interference cause is established.

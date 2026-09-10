# Native Camera Setup Delivery

Issue #9952, child of #9906, implements the missing fresh-install recording setup.
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

Build the normal frontend-inclusive wheel from committed source and verify the
installed editor saves, reopens and selects plans without checkout sample files.
Complete CI and protected merge. Live camera mode support and calibration accuracy
remain separate hardware acceptance; no synthetic test qualifies physical accuracy.
The prerequisite installed-runtime PR is #9950 at d5f44f211.

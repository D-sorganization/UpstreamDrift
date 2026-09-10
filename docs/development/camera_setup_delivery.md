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

## Guided Workflow Acceptance

Issue #9909 adds three actual Qt integration checks in
`tests/tools/capture_rig/test_guided_capture_acceptance.py`. They cover measured
and unknown club lengths across restart, immutable capture snapshots after bag
changes, fresh calibration confirmation, changed zoom blocking reuse while
editing remains usable, and actual video import/comparison save recognized by
the wizard. No detector or body solver is replaced by these UI tests; numerical
qualification remains in the existing pipeline test suites.

## Map Maintenance and Progress Identity

The source is `src/config/capability_connections.json`. Executable prerequisites
are separate from informational architecture arrows. Run
`python3 -m scripts.generate_capability_atlas` to regenerate the browser reference,
graph data and Mermaid diagrams; CI checks freshness.

Progress fingerprints inspect bounded metadata and media size/modification time
rather than decoding or hashing video during navigation. They invalidate UI
state; they are not media-integrity certificates. External replacement with
unchanged size/modification time requires manual review. Reference alignment
retains its camera, clock and asset-binding validation. Generate the atlas from
canonical LF source bytes so its input hashes agree with CI checkouts.

## Final Local Acceptance Update

At f7e7f1663, all 566 Capture Rig tests pass on Python 3.12, including the
three guided restart/comparison cases and searchable in-app examples. Every
help link targets an existing workflow step. The independent worker suite now
passes 30 checks after adding a persisted calibration-to-overlay test: new
non-planar points pass through `start_cameras_from` and
`project_reference_to_camera` and agree with independent OpenCV projections
within 0.00001 pixels. The accepted calibration bytes remain unchanged.
This checks coordinate and lens integration, not a detector/body-model job or
physical calibration accuracy. The initial added test lacked capture bundle
metadata; adding the existing synthetic bundle fixture corrected that test
setup without changing production code or tolerances.

PR #9955 at 63cb27402 has passed the full unit and quality gates. Its optional
SDK/authority jobs are still queued as of 2026-09-10 10:46 UTC. The prior
statement about scalar bounds describes the prerequisite failure; the fix
is not merged yet. Hold the current acceptance additions for the next
integration of #9955 and #9950, then validate and publish #9954 normally.

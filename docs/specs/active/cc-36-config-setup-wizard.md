# CC-36 Config Validation Setup Wizard

- status: active
- issue: [#6809](https://github.com/D-sorganization/UpstreamDrift/issues/6809)
- branch: `feat/6809-config-setup-wizard`

## Problem

Canonical-core runs need a deterministic preflight that catches common setup
mistakes before an engine starts. The MVP must validate config preconditions and
explain fixes in plain language, without delegating decisions to Sidekick or any
LLM-backed agent.

## Scope

- Validate canonical-core units and frames against `canonical-v2`.
- Validate the canonical model block has a stable id, named joints, and
  floating-base dimensions satisfying `nq == nv + 1`.
- Validate subject calibration is present and marked complete or validated.
- Provide a pure wizard view model and an embeddable launcher tool surface.

## Non-Goals

- No LLM autonomy or automatic config mutation.
- No engine-specific adapter conversion.
- No replacement for existing anthropometrics, motion-pipeline, or model-pack
  contracts.

## Design

`src.shared.python.config.setup_wizard` owns the reusable API:

- `validate_canonical_setup_config(config)` returns a
  `SetupValidationReport`.
- `SetupValidationIssue` carries `code`, `field_path`, plain-language
  `message`, and deterministic `suggested_fix`.
- `SetupWizardViewModel` is a headless state machine with four gates:
  units/frames, model, calibration, and review.

`src.tools.config_setup_wizard` exposes the MVP as an embeddable tool. The
adapter imports without PyQt6; the GUI imports PyQt only when a host creates the
widget.

## Acceptance Criteria

- A valid canonical-v2 SI/Z-up config with a complete calibration has no errors.
- Non-SI units, non-Z-up frames, invalid model dimensions, and missing
  calibration produce blocking validation errors.
- Each error includes a suggested fix suitable for display in a wizard.
- Wizard progression stops at the first step with blocking errors and advances
  after the caller supplies a corrected config.

## Validation

Focused tests live in `tests/unit/config/test_setup_wizard.py` and cover
validation rules, suggested fixes, wizard step progression, and embeddable
adapter protocol conformance.

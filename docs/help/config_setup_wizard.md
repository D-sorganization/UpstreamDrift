---
title: Setup Wizard
tile_id: config_setup_wizard
status: complete
---

# Setup Wizard

## Purpose

Setup Wizard checks a canonical-core run configuration before you start
a simulation and tells you, in plain language, what is wrong and how to
fix it. It walks four steps - units and frames, canonical model,
calibration, review - and refuses to advance past a step that still has
blocking errors. It validates; it does not write your config for you.

## Inputs

A single JSON object pasted into the text box ("Paste canonical-core
setup JSON"). It must decode to an object. The validated fields and
their required units:

| Field | Required value / unit |
| --- | --- |
| `convention` | exactly `"canonical-v2"` |
| `units` | `"SI"`, or an object mapping the six quantities below |
| `units.length` | `"m"` |
| `units.mass` | `"kg"` |
| `units.time` | `"s"` |
| `units.angle` | `"rad"` |
| `units.force` | `"N"` |
| `units.torque` | `"N*m"` (aliases `N-m`, `Nm`, `N.m` accepted) |
| `frame` or `world_frame` | exactly `"world_Zup"` |
| `gravity` | optional; a finite 3-vector in m/s^2, must equal `[0.0, 0.0, -9.80665]` within 1e-6 |
| `model.canonical_id` (or `model.id`) | non-empty string |
| `model.joint_names` | non-empty list of non-empty strings |
| `model.nq`, `model.nv` | positive integers satisfying `nq == nv + 1` |
| `calibration.status` | `"complete"`, or `calibration.validated` is `true` |
| `calibration.anthropometrics_ref` | or `subject_anthropometrics`, or `subject_id` - at least one must be set |

## Outputs

- A status line: the current step name plus `valid` or `needs fixes`.
- A per-step list of `title: status (issue_count)`, where status is one
  of `complete`, `blocked`, `ready`, `waiting`.
- For each issue, two lines: `field_path: message` and `Fix:
  suggested_fix`. Every issue also carries a stable `code`
  (`CC36_CONVENTION`, `CC36_UNITS`, `CC36_UNIT_MISMATCH`,
  `CC36_WORLD_FRAME`, `CC36_GRAVITY_SHAPE`, `CC36_GRAVITY_FRAME`,
  `CC36_MODEL_REQUIRED`, `CC36_MODEL_ID`, `CC36_MODEL_JOINTS`,
  `CC36_MODEL_DIMENSION_REQUIRED`, `CC36_MODEL_DIMENSIONS`,
  `CC36_CALIBRATION_REQUIRED`, `CC36_CALIBRATION_INCOMPLETE`,
  `CC36_CALIBRATION_SUBJECT`) and a severity of `error` or `warning`.
  Only `error` blocks advancing.
- On malformed input, `Invalid input: <exception>` in the status line.

## Method

The tile is a thin PyQt6 view over a pure state machine. The widget is
`ConfigSetupWizardWidget` in
[`gui.py`](../../src/tools/config_setup_wizard/gui.py) - about 100
lines: a `QPlainTextEdit` for input, a read-only `QTextEdit` for output,
a status `QLabel`, and Back / Validate / Next buttons.

All logic is in
[`SetupWizardViewModel`](../../src/shared/python/config/setup_wizard.py),
whose module docstring is explicit: "The setup wizard is intentionally a
pure validation layer. It does not call Sidekick, an LLM, or any
autonomous agent service." `validate_canonical_setup_config` runs three
deterministic checkers - `_validate_units_and_frames`,
`_validate_model`, `_validate_calibration` - and returns a frozen
`SetupValidationReport`. `advance()` validates, then moves forward only
if `can_advance()` finds no errors owned by the current step. Gravity
comparison uses `numpy.allclose`.

`_embed_adapter.py` registers `ConfigSetupWizardAdapter` with the
process-wide embeddable-tool registry at import time, requesting a tab
(not dock) placement with a 720x520 px minimum.

## Limitations

- **It does not read or write files.** There is no file picker, no
  "load current config", and no save. You paste JSON in and read text
  out; applying the fixes is your job.
- **It does not fix anything.** Suggested fixes are static strings, not
  actions.
- It checks declarations, not reality. It confirms that `model.nq` and
  `model.nv` are declared consistently; it never loads the model or asks
  an engine adapter what the true dimensions are.
- The `nq == nv + 1` rule assumes a canonical-v2 floating-base model with
  a quaternion. A configuration that is legitimately not floating-base
  will be reported as an error.
- Gravity must be exactly Earth-standard `-9.80665` m/s^2 in Z-up. Any
  other magnitude or direction is an error, so reduced-gravity or
  alternative-frame setups cannot pass.
- `is_dirty()` always returns `False`; the wizard holds no state worth
  prompting about, and closing the tab discards the pasted text.
- Maturity is **beta**. The widget module is marked
  `pragma: no cover - GUI smoke follows later`, so the Qt surface is not
  covered by tests.

## See Also
- [Engine selection](engine_selection.md)
- [Simulation controls](simulation_controls.md)
- [Model Explorer](model_explorer.md) - where model dimensions come from

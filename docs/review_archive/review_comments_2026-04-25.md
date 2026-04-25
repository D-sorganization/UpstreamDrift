# Review Comments Archive - 2026-04-25

Generated: 2026-04-25T11:50:47.535127

## Reviewer (chatgpt-codex-connector[bot]) (6 comments)

### PR #3282: src/shared/python/calc_backend/tests/test_inline_calcs.py:1

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Keep drift baseline aligned with renamed test file**

Renaming `test_calc_backend.py` to this module without updating `TOOLS_BASELINE_HASHES` in `src/shared/python/calc_backend/tests/test_calc_backend_drift.py` leaves the drift guard pointing at a non-existent path (`.../tests/test_calc_backend.py`, line 45). In `test_calc_backend_modules_match_tools_baseline`, that entry will now hit the `if not path.exists(...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3282#discussion_r3142361415)

---

### PR #3283: src/shared/python/ai/sample_tools.py:672

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Preserve validate_cross_engine result contract**

This return payload removes previously exposed fields like `source`, `max_delta`, and `engines` and flips `payload['success']` to `False`, which breaks existing callers that treat this tool as a structured diff report; the current test suite already dereferences those keys in `tests/unit/ai/test_sample_tools.py:259-272` and `tests/test_sample_tools_real.py:72-...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3283#discussion_r3142371444)

---

### PR #3283: src/shared/python/ai/sample_tools.py:669

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Fail tool execution when validation is unimplemented**

Returning `{"success": False}` inside the payload does not trigger workflow failure handling, because `WorkflowEngine._execute_step_tool` only fails on `ToolResult.success == False` (i.e., execution error), not on fields inside `result`; this means the `run_validation` step in the cross-engine workflow can be marked completed even though validation never...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3283#discussion_r3142371445)

---

### PR #3287: src/shared/python/physics/shaft_model.py:401

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Split shaft_model.py below the repository size cap**

The new `shaft_model.py` exceeds the repository's mandatory file-size limit from `AGENTS.md` (section **5d: No Monolithic Files**, max 400 lines). This file reaches 418 lines, so the refactor still leaves a monolithic module and reintroduces the maintainability debt this change is trying to resolve. Please split this file into smaller focused modules (for ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3287#discussion_r3142402740)

---

### PR #3287: src/shared/python/physics/shaft_integrator.py:401

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Break up shaft_integrator.py to satisfy AGENTS.md limits**

`shaft_integrator.py` also violates `AGENTS.md` section **5d (No Monolithic Files)** by exceeding the 400-line maximum for newly created files (this module is 528 lines). Keeping integration, static analysis, and factory logic in one oversized file undermines the stated architecture rule and increases long-term maintenance risk; it should be split in...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3287#discussion_r3142402741)

---

### PR #3292: src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/tabs/controls_tab.py:38

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Avoid wiring incompatible actuator mixin**

By inheriting `_ActuatorControlsMixin` here, `Edit.../Params...` now flow through `open_actuator_detail_dialog()` to `tabs/actuator_detail_dialog.py`, whose constructor calls `ControlSystem.get_control_type()`, `get_constant_value()`, `get_damping()`, and `get_sine_params()`. Those methods are not defined in `control_system.py`, so opening the actuator detail dialog...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3292#discussion_r3142411978)

---


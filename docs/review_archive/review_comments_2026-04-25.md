# Review Comments Archive - 2026-04-25

Generated: 2026-04-25T11:50:47.535127

## Reviewer (chatgpt-codex-connector[bot]) (3 comments)

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


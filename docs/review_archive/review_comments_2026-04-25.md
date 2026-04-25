# Review Comments Archive - 2026-04-25

Generated: 2026-04-25T11:50:47.535127

## Reviewer (chatgpt-codex-connector[bot]) (11 comments)

### PR #3264: src/api/local_server.py:71
### PR #3265: .github/workflows/Jules-Assessment-Remediator.yml:66
### PR #3281: src/shared/python/engine_core/base_physics_engine.py:487
### PR #3282: src/shared/python/calc_backend/tests/test_inline_calcs.py:1

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Remove incorrect changelog claim about deleted RL files**

This new changelog entry states that `src/reinforcement_learning/trajectory_funnel_benchmark.py` and its tests were deleted, but those files are still present in the repository (`src/reinforcement_learning/trajectory_funnel_benchmark.py`, `tests/unit/reinforcement_learning/test_trajectory_funnel_benchmark.py`, and `tests/reinforcement_learning/test_tr...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3260#discussion_r3142247464)
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Remove incorrect deletion claim from changelog**

This new changelog row says `src/reinforcement_learning/trajectory_funnel_benchmark.py` and its tests were deleted, but those files still exist in the repository (`src/reinforcement_learning/trajectory_funnel_benchmark.py`, `tests/reinforcement_learning/test_trajectory_funnel_benchmark.py`, and `tests/unit/reinforcement_learning/test_trajectory_funnel_benchmar...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3261#discussion_r3142241668)
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Avoid hard dependency on python-multipart at server import**

Importing `data_explorer` unconditionally here makes `src.api.local_server` fail to import in environments without `python-multipart` installed, because `src/api/routes/data_explorer.py` defines an `UploadFile` endpoint and FastAPI raises `RuntimeError` during route setup. This repository already treats multipart as optional (for example, `tests/un...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3264#discussion_r3142264352)
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Include local kill-switch action in sparse checkout**

The new `kill-switch` job checks out only `.github/WORKFLOWS_PAUSED` via `sparse-checkout`, but the next step uses the local action `./.github/actions/check-kill-switch`. Because that directory is not fetched, the runner cannot resolve `action.yml` and the job fails before evaluating the kill switch, which blocks normal workflow execution. Please either i...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3265#discussion_r3142264993)
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Keep GOLF_SUITE_MODE fallback in mode resolution**

This change drops support for the legacy `GOLF_SUITE_MODE` variable, so environments that have not migrated yet will silently fall back to the default mode (`remote`) and start enforcing auth where local-mode bypass previously worked. Because local auth behavior is security-critical and the repo still contains existing usage of the legacy names, this is a br...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3276#discussion_r3142346090)

---

### PR #3276: src/api/auth/security.py:28
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Stop advertising CONTACT_FORCES in MyoSuite capabilities**

This declares `Capability.CONTACT_FORCES`, but `MyoSuitePhysicsEngine` does not implement `compute_contact_forces()`, so calls fall back to `DynamicsInterface.compute_contact_forces()` which returns `np.zeros(3)` by default. With this capability flag set, downstream code that correctly gates on `engine.capabilities()` will treat zero vectors as real ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3279#discussion_r3142352327)
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Avoid making BasePhysicsEngine abstract without migrations**

Adding `@abstractmethod capabilities()` here makes every existing `BasePhysicsEngine` subclass that lacks this method non-instantiable. In this tree, `PendulumPhysicsEngine` and `GolfSwingPendulumEngine` do not implement `capabilities()`, so constructing them now raises `TypeError` (e.g., `Can't instantiate abstract class ... without an implementat...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3281#discussion_r3142352713)
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Keep drift baseline aligned with renamed test file**

Renaming `test_calc_backend.py` to this module without updating `TOOLS_BASELINE_HASHES` in `src/shared/python/calc_backend/tests/test_calc_backend_drift.py` leaves the drift guard pointing at a non-existent path (`.../tests/test_calc_backend.py`, line 45). In `test_calc_backend_modules_match_tools_baseline`, that entry will now hit the `if not path.exists(...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3282#discussion_r3142361415)

---

### PR #3283: src/shared/python/ai/sample_tools.py:672

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Accept legacy GOLF_API_SECRET_KEY during migration**

Reading only `UPSTREAM_API_SECRET_KEY` here breaks deployments that still provide `GOLF_API_SECRET_KEY`: in production, startup now raises a missing-secret runtime error; in non-production, the app falls back to a random per-process key and invalidates JWTs on restart. This should keep the legacy env var as a temporary fallback to avoid an outage during re...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3276#discussion_r3142346092)

---

### PR #3276: src/api/database.py:15
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Preserve validate_cross_engine result contract**

This return payload removes previously exposed fields like `source`, `max_delta`, and `engines` and flips `payload['success']` to `False`, which breaks existing callers that treat this tool as a structured diff report; the current test suite already dereferences those keys in `tests/unit/ai/test_sample_tools.py:259-272` and `tests/test_sample_tools_real.py:72-...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3283#discussion_r3142371444)

---

### PR #3283: src/shared/python/ai/sample_tools.py:669

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Fallback to existing default SQLite file on rename**

Changing the implicit default database file name without a fallback/migration path causes users who rely on default `DATABASE_URL` behavior to boot against a brand-new empty SQLite database after upgrade, effectively hiding existing accounts and state stored in `golf_modeling_suite.db`. Add a compatibility check (or one-time migration) when the old file ex...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3276#discussion_r3142346095)
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


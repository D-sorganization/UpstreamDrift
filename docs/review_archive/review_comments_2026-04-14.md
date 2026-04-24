# Review Comments Archive - 2026-04-14

Generated: 2026-04-14T13:26:06.565746

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #2673: src/engines/physics_engines/drake/python/perturbation/analyzer.py:424

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Map actuator torques with Drake actuation matrix**

`compute_a` now uses `np.resize(tau_act, nv)`, which repeats actuator values when `nu != nv` (for example `[t0, t1] -> [t0, t1, t0]`) and applies torque to the wrong generalized coordinates. Because `DrakePerturbationAnalyzer` accepts arbitrary URDFs via `urdf_path`, underactuated/floating-base models will produce silently corrupted accelerations instead of ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2673#discussion_r3082207938)

---

### PR #2673: src/engines/physics_engines/pinocchio/python/pinocchio_physics_engine.py:312

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Preserve GRF fallback path for unsupported contact forces**

Making `PinocchioPhysicsEngine.compute_contact_forces` always raise `NotImplementedError` breaks existing fallback flows that call this method directly and only fall back after receiving a zero vector (for example `extract_grf_from_contacts` in `src/shared/python/physics/ground_reaction_forces.py`). With Pinocchio selected, GRF extraction now aborts...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2673#discussion_r3082207943)

---

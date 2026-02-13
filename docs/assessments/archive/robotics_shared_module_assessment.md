# Robotics Expansion - Shared Module Assessment

> **Date:** 2026-02-03
> **Purpose:** Assess which robotics expansion modules should be moved to shared/python for cross-project reuse

## Executive Summary

The robotics expansion (Phases 3-5) introduces 12 new modules across learning, deployment, and research domains. Of these, **7 modules are recommended for sharing** due to their generic, domain-agnostic nature.

---

## Assessment Criteria

| Criterion          | Description                                      |
| ------------------ | ------------------------------------------------ |
| **Generic**        | Not tied to golf/swing-specific domain           |
| **Self-contained** | Minimal dependencies on other project modules    |
| **Reusable**       | Applicable to other robotics/simulation projects |
| **Stable API**     | Well-defined interfaces unlikely to change       |

---

## Phase 3: Learning and Adaptation (PR #1077)

| Module                              | Location                | Recommendation | Rationale                                                           |
| ----------------------------------- | ----------------------- | -------------- | ------------------------------------------------------------------- |
| `rl/base_env.py`                    | `learning/rl/`          | ⚠️ Partial     | Base Gymnasium wrapper is generic; task configs are domain-specific |
| `rl/humanoid_envs.py`               | `learning/rl/`          | ❌ Keep        | Specific to humanoid locomotion tasks                               |
| `rl/manipulation_envs.py`           | `learning/rl/`          | ❌ Keep        | Specific to manipulation tasks                                      |
| `imitation/dataset.py`              | `learning/imitation/`   | ✅ **Share**   | Generic demonstration dataset handling                              |
| `imitation/learners.py`             | `learning/imitation/`   | ✅ **Share**   | BC, DAgger, GAIL are domain-agnostic                                |
| `sim2real/domain_randomization.py`  | `learning/sim2real/`    | ✅ **Share**   | Generic parameter randomization                                     |
| `sim2real/system_identification.py` | `learning/sim2real/`    | ⚠️ Partial     | Concept is generic; may need interface refinement                   |
| `retargeting/retargeter.py`         | `learning/retargeting/` | ⚠️ Partial     | Useful but tightly coupled to skeleton configs                      |

**Recommendation:** Share `imitation/` and `domain_randomization.py`

---

## Phase 4: Industrial Deployment (PR #1078)

| Module                       | Location                    | Recommendation | Rationale                                                |
| ---------------------------- | --------------------------- | -------------- | -------------------------------------------------------- |
| `realtime/state.py`          | `deployment/realtime/`      | ✅ **Share**   | Generic robot state dataclasses                          |
| `realtime/controller.py`     | `deployment/realtime/`      | ⚠️ Partial     | Protocol interfaces are generic; implementation may vary |
| `digital_twin/twin.py`       | `deployment/digital_twin/`  | ⚠️ Partial     | Concept is valuable; needs abstraction layer             |
| `digital_twin/estimator.py`  | `deployment/digital_twin/`  | ✅ **Share**   | Kalman filter is generic                                 |
| `safety/monitor.py`          | `deployment/safety/`        | ✅ **Share**   | ISO compliance checking is broadly applicable            |
| `safety/collision.py`        | `deployment/safety/`        | ✅ **Share**   | Potential field methods are domain-agnostic              |
| `teleoperation/devices.py`   | `deployment/teleoperation/` | ✅ **Share**   | Device protocols are standardized                        |
| `teleoperation/interface.py` | `deployment/teleoperation/` | ⚠️ Partial     | Useful but may need customization                        |

**Recommendation:** Share `safety/`, `teleoperation/devices.py`, `estimator.py`, `state.py`

---

## Phase 5: Advanced Research (PR #1079)

| Module                        | Location                   | Recommendation | Rationale                                                  |
| ----------------------------- | -------------------------- | -------------- | ---------------------------------------------------------- |
| `mpc/controller.py`           | `research/mpc/`            | ✅ **Share**   | iLQR solver is generic optimization                        |
| `mpc/specialized.py`          | `research/mpc/`            | ❌ Keep        | CentroidalMPC is locomotion-specific                       |
| `differentiable/engine.py`    | `research/differentiable/` | ✅ **Share**   | Autodiff engine wrapper is generic                         |
| `deformable/objects.py`       | `research/deformable/`     | ⚠️ Partial     | FEM/cloth are generic; may need physics engine abstraction |
| `multi_robot/system.py`       | `research/multi_robot/`    | ⚠️ Partial     | Task coordination is generic; robot interface varies       |
| `multi_robot/coordination.py` | `research/multi_robot/`    | ✅ **Share**   | Formation control math is universal                        |

**Recommendation:** Share `mpc/controller.py`, `differentiable/engine.py`, `coordination.py`

---

## Proposed Shared Structure

```
shared/python/
├── robotics/                    # New robotics utilities
│   ├── __init__.py
│   ├── imitation/              # From Phase 3
│   │   ├── dataset.py          # Demonstration handling
│   │   └── learners.py         # BC, DAgger, GAIL
│   ├── sim2real/               # From Phase 3
│   │   └── domain_randomization.py
│   ├── safety/                 # From Phase 4
│   │   ├── monitor.py          # Safety limits checking
│   │   └── collision.py        # Potential field avoidance
│   ├── control/                # From Phase 4 & 5
│   │   ├── state.py            # Robot state dataclasses
│   │   ├── estimator.py        # Kalman filtering
│   │   └── mpc.py              # iLQR solver
│   ├── teleoperation/          # From Phase 4
│   │   └── devices.py          # Input device protocols
│   ├── differentiable/         # From Phase 5
│   │   └── engine.py           # Autodiff wrapper
│   └── multi_robot/            # From Phase 5
│       └── coordination.py     # Formation control
```

---

## Implementation Notes

1. **Dependency Isolation**: Shared modules should only depend on standard libraries (numpy, scipy) and shared utilities
2. **Protocol-Based Design**: Use Python protocols/ABCs for extensibility
3. **No Physics Engine Lock-in**: Abstract physics engine interactions behind protocols
4. **Documentation**: Each shared module needs standalone docstrings and examples

---

## Priority Order

| Priority | Module                     | Impact                        | Effort |
| -------- | -------------------------- | ----------------------------- | ------ |
| 1        | `safety/`                  | High - Industrial compliance  | Low    |
| 2        | `imitation/`               | High - Research enablement    | Low    |
| 3        | `mpc/controller.py`        | Medium - Optimization utility | Low    |
| 4        | `differentiable/engine.py` | Medium - Research enablement  | Medium |
| 5        | `teleoperation/devices.py` | Medium - Hardware abstraction | Low    |
| 6        | `coordination.py`          | Medium - Multi-robot utility  | Low    |
| 7        | `domain_randomization.py`  | Medium - Sim2real utility     | Low    |

---

## Next Steps

1. [ ] After PRs #1077-1079 are merged, create migration PR
2. [ ] Add shared module imports to `shared/python/__init__.py`
3. [ ] Update import paths in original modules
4. [ ] Add unit tests for shared modules
5. [ ] Document shared robotics utilities in `/docs/api/`

---
issue: 14943
summary: "State-aware counterfactual defaults, simscape lifecycle, putting control (#14943 follow-up)"
---

## Handoff

- Addressing Brick's acceptance review #6772934494 on #14943:
  - Evaluates counterfactual defaults at configuration q with state restoration
  - Enforces SimscapeAdapter lifecycle, 1-D preconditions, and dof-shaped vectors
  - Adds regression tests for counterfactual adapter boundary guards
  - Implements compute_control_acceleration_at_state on SharedPuttingEngine

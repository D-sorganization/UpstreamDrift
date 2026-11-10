---
issue: 14943
summary: "State-aware counterfactual defaults, simscape lifecycle, putting control (#14943 follow-up)"
---

## Handoff

- Addressing Brick's acceptance review #6772934494 on #14943:
  - Evaluates counterfactual defaults at configuration q with state restoration
  - Enforces SimscapeAdapter lifecycle and 1-D array preconditions
  - Implements compute_control_acceleration_at_state on SharedPuttingEngine

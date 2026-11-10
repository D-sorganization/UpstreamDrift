---
issue: 14943
summary: "CounterfactualKind/Result, simscape/putting defaults, conformance (#14943 slices 1–4)"
---

## Handoff

- Slices 1–4 of #14943 (Part of #14934, supersedes #14965):
  - Added typed CounterfactualKind/CounterfactualResult and conformance helpers
  - Added compute_control_acceleration_at_state and compute_gravity_only_acceleration
  - Deprecated legacy compute_zvcf with one-release shims
  - Updated divergence inventory and added unit tests

---
issue: 11898
summary: "The F09c full-body MuJoCo path records a state-dependent generalized effort once per fixed interval and returns the existing T01 frozen-input bundle, with manufactured-plant tests of eight-substep ZOH reproduction; no native full-body execution, bounded-control, support/grip, capture-matching, or physiology qualification is claimed"
---

The F09c full-body MuJoCo path now records a state-dependent generalized
effort once per fixed interval and returns the existing T01 frozen-input
bundle. The recorder validates detached complete-state controller observations,
full finite effort vectors, exact root-zero effort, controller provenance, and
the exact applied history. Manufactured-plant tests cover eight-substep ZOH
reproduction without calling the controller during replay; no native full-body
execution, bounded-control, support/grip, capture-matching, or physiology
qualification is claimed.

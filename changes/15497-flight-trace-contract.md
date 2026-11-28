---
issue: 15497
summary: "Reject malformed bunker flight and rollout traces before visualization"
dl_state: "in_review"
next_step: "Independent review and CI; leave this slice unarmed per owner instruction"
branch: "codex/15497-flight-trace-contract"
---

Add always-on BunkerFlight invariants for finite SI coordinates, complete paths,
matching carry and landing, continuous rollout, and a ground-level rest point.
Retain airborne bounce samples and floating-point landing residue. No physics,
API, renderer, or web implementation changes. Refs #15497.

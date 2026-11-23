---
issue: 15493
summary: "Add bracketed approaching signed-distance contact root and finite missed-contact penalty"
---

Slice 1 contact-convergence primitive; backend-agnostic, no physics model added.
A missed contact yields a smooth finite cost, never NaN. No acceptance claim.

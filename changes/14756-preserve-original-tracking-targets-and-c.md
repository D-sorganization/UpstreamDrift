---
issue: 14756
summary: "Preserve original tracking targets and clocks separately from same-input rollout evidence."
---

Both reference generation paths retain the complete original target clock
(seconds, including its input origin) and coordinates in spec order, separately
from the generated rollout. Lead-in bundles retain the same full target;
their existing lead_in duration records the rollout's offset from target start.
Target arrays must be present together, finite, and shape-compatible with a
strictly increasing clock. A versioned SHA-256 binds the exact spec and target
shape/clock/coordinates (canonical little-endian float64 array bytes).
Loading rejects missing or mismatched target identity; historical bundles
without target evidence remain readable with no inferred target.

This is offline provenance preservation, not a tracking-error metric, a new run
registry, dynamic parity qualification, or physical acceptance. Existing run
and capture provenance is retained. Unit fixtures distinguish a reproducible
zero rollout from a nonzero original target; no real-engine run was performed.
Next: independent review and CI disposition; keep the PR unarmed.

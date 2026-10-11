---
issue: 12275
summary: "Retain hash-bound manufactured native diagnostics"
dl_state: "in_review"
next_step: "Review byte-preserved evidence and documentation; keep all production qualification gates open."
branch: "docs/manufactured-feedback-evidence-12275"
---

Retain compact, source-bound synthetic observations and pure-test snapshots for MuJoCo, OpenSim, Pinocchio and Drake. Preserve successful and failed attempt receipts exactly, with a manifest binding archived bytes to original paths and SHA-256 identities. Document runtime prerequisites, limits and reproduction boundaries in the package README and canonical references. Do not copy private captures or runtime binaries. No production model/variant is qualified by these observations.

The separate Drake zero-effort observation is a near-touch free-fall sample
with descriptive qdd comparisons only. It does not qualify a production
model/variant.

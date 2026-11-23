---
issue: 14756
summary: "Reject a same-engine wrong-run ledger entry in render_best_matches and label unbound entries in SUMMARY."
---

P3 slice: `scripts/render_best_matches.py` joined ledger metrics by engine name
only. A ledger entry may now declare `input_sha256`; a mismatch with the
rendered input hashes stops the run, and an entry without it is labelled
UNBOUND in SUMMARY.md. SUMMARY also states the operation is a visual replay.
Run, capture and receipt identity binding stays with the RUN manifest work.
No ledger data or scientific gate changes.

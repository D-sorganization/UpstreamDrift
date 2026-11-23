---
issue: 14756
summary: "Bind parity-report candidate fingerprints to full candidate provenance and array evidence."
---

P6 identity-only slice: candidate_sha256 is a full SHA-256 over canonical
candidate metadata and freshly computed array checksums, shapes and dtypes.
The human-readable candidate_id remains the model name. This identifies the
candidate package, not a run, report operation, engine execution or physical
acceptance. Missing provenance remains missing; hashing does not verify it.
Old q-only fingerprints intentionally change. No scientific gates change.

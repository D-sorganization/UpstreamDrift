---
issue: 15119
summary: "Record the reviewed GAAI policy-test retirement for merge-group CI"
---

The deleted architecture test only read the retired GAAI project rules. Record
its governing fleet issue and replacement retirement guard in the existing
reviewed-deletion registry; keep the deletion checker unchanged.

Validation: the operational deletion check fails before the record and passes
afterward; the 34 focused retirement/deletion-checker tests pass. Fresh independent
acceptance and protected merge-group CI remain required at the repaired head.

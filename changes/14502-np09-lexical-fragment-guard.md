---
issue: 14502
summary: "Guard NP-09 lexical test against collated changelog fragment removal"
---

- Scan changes/14502-np09-io-sensitivity.md in test_lexical_guardrail_zero_violations only when the fragment is still on disk, so merge-queue unit-test-gate survives collation (#14502).

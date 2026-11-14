---
issue: 15093
summary: "Guard NP-05 lexical test against collated changelog fragment removal (#15093)"
---

- Check `changes/14480-np05-edit-vs-finetune-benchmark.md` in `test_lexical_guardrail_zero_violations` only when the change fragment is present on disk so weekly changelog collation does not break `unit-test-gate` (#15093).

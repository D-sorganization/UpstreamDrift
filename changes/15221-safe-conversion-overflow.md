---
issue: 15221
summary: "Handle overflow through safe numeric conversion fallback and strict contracts"
dl_state: "in_review"
next_step: "CI and automated review"
branch: "codex/burn1116-15221-numeric-overflow"
paths: "src/shared/python/core/type_utils.py,tests/unit/core/test_type_conversion_overflow.py"
---

28 regressions failed before the two-line fix; 114 related tests pass and module coverage rises from 67.8% to 75.5% without changing its floor.

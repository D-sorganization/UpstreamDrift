---
issue: 14227
summary: "Record first green scheduled CI Optional Stack after launcher processes fix (#15089)"
---

- Refresh `docs/ci/optional_stack_14227.md` with scheduled run `43688504401` (2026-11-15) green across all lane jobs.
- Add runbook contract test `test_runbook_records_green_scheduled_optional_stack_after_slice3` in `tests/ci/test_optional_stack_gravity_contract_14227.py`.
- Point `ci-optional-stack.yml` at the #14227 runbook so anti-phantom-merge Rule 3 passes on docs-only acceptance closes.
- Bind the historical acceptance to `d7bdbc569964ca2f85af6f3f1e05bd0362b87e7b`, verified against the run API and each lane's checkout log; the previously named `b8159ea3c` was a later commit.

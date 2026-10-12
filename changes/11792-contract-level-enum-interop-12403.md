---
issue: 11792
summary: "Normalize contract-level enum values at the UD alias boundary"
dl_state: "in_progress"
next_step: "Review the follow-up ModuleType annotation and focused mypy/test results, then decide on one final full gate before publication. The baseline failures remain separately tracked in #12404 and #12405; Tools-owned #5577 remains separate."
owner: "codex"
branch: "fix/contract-enum-interop-12403-20261012"
paths: "src/shared/python/_contracts_level.py,src/shared/python/core/contracts/level.py,src/shared/python/ud_import_alias_policy.py,tests/unit/test_contract_level_interop.py,tests/shared/wave10_contracts/test_level.py,docs/development/CONTRACT_LEVEL_ENUM_INTEROP_20261012.md,docs/development/feedback_controls/CONTRACT_LEVEL_ENUM_INTEROP_TURNOVER_20261012.md"
---

This is a software maintenance child, #12403, under active #11792 work. It normalizes compatible enum values through the UD-installed public wrapper and in UD-owned shared/core setters. The canonical Tools setter source is unchanged; its independently reproduced direct-call defect is tracked separately in #5577. The connection between this bug and merge-group CI failures is unproven. No native or scientific qualification is claimed.

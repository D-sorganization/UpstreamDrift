---
issue: 15106
summary: "Prune dead testpaths/LEGACY_SRC_TEST_DIRS, enable green tests/* suites in unit-test-gate, and enforce unrun suite exclusions (#15106)"
---

- Pruned dead `testpaths`/`LEGACY_SRC_TEST_DIRS`, enabled directory unit auto-marking on 7 verified `tests/*` suites, and enforced `scripts/config/unrun_test_suite_exclusions.json` via `scripts/ci/check_unrun_suite_exclusions.py` (#15106).

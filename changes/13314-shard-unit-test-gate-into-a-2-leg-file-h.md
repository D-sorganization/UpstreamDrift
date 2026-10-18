---
issue: 13314
summary: "Shard unit-test-gate into a 2-leg file-hash matrix so each leg stays well under the 35 min limit without raising it"
branch: "ci/13314-shard-unit-test-gate"
paths: ".github/workflows/ci-standard.yml,scripts/ci/pytest_unit_gate_shard.py,tests/ci/test_unit_gate_shard.py"
---

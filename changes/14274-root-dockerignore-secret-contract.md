---
issue: 14274
summary: "Guard root Docker context secret-exclusion rules"
dl_state: "in_review"
next_step: "Complete independent review and open the scoped CI contract PR."
branch: "codex/14274-dockerignore-secret-contract-20261103"
paths: ".dockerignore, scripts/ci/check_dockerfile_contracts.py, tests/unit/scripts/test_dockerfile_contracts.py"
---

This slice guards only the root build-context .dockerignore directives. It does not implement Docker full pattern matching, Dockerfile-specific or nested context checks, runtime secret protection, or upgrade/rollback guarantees; issue #14274 remains open for its broader acceptance.

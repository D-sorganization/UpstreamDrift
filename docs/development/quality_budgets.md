# Quality budgets index

A single, discoverable index of the repository's enforced quality budgets and
ratchets (issue #7133, resolution-plan step 6: "make all quality budgets
visible in one policy file"). Each row links a budget to its source-of-truth
config and the script that enforces it; the `quality-gate` job in
`.github/workflows/ci-standard.yml` runs these on every PR.

Budgets are **ratchets**: the recorded baseline may shrink but must not grow
without an explicit, reviewed edit to the config file. New code must use the
documented helpers rather than adding grandfathered exceptions.

| Budget / ratchet               | Config (source of truth)                           | Enforcer script                              |
| ------------------------------ | -------------------------------------------------- | -------------------------------------------- |
| Per-file size budget           | `scripts/config/file_size_budget.json`             | `scripts/ci/check_file_size_budget.py`       |
| Module size budget             | `scripts/config/module_size_budget_baseline.json`  | `scripts/check_module_size_budget.py`        |
| Doc size budget                | `scripts/config/doc_size_budget.json`              | `scripts/check_module_size_budget.py` (docs) |
| Error-handling ratchet         | `scripts/config/error_handling_baseline.json`      | `scripts/ci/check_error_handling_ratchet.py` |
| Suppression discipline ratchet | `scripts/config/suppression_ratchet_baseline.json` | `scripts/ci/check_suppression_ratchet.py`    |
| MyPy exclusion budget          | `scripts/config/mypy_exclusion_budget.json`        | `scripts/check_mypy_exclusion_budget.py`     |
| Full-src mypy baseline         | `scripts/config/full_src_mypy_baseline.json`       | `scripts/check_mypy_exclusion_budget.py`     |
| Dependency direction rules     | `scripts/config/dependency_direction_rules.json`   | `scripts/check_dependency_direction.py`      |
| pip-audit waivers              | `scripts/config/pip_audit_waivers.json`            | `scripts/ci/check_pip_audit_waivers.py`      |
| SBOM baseline                  | `scripts/config/sbom_baseline.json`                | `make sbom`                                  |
| UX field coverage baseline     | `scripts/config/ux_field_coverage_baseline.json`   | UX coverage checks                           |
| Monolith refactor register     | `docs/development/monolith_refactor_register.md`   | `scripts/gen_monolith_register.py` (#7131)   |

## Non-budget gates (also blocking in `quality-gate`)

These have no per-file baseline; they fail outright on any violation:

- **Lint / format** — `ruff check` and `ruff format --check` (separate steps).
- **No new `print()` in `src/`** — `scripts/check_no_print_calls.py` (CLI
  stdout exceptions are listed in `pyproject.toml [tool.ruff.lint.per-file-ignores]`).
- **No placeholders** — TODO/FIXME must reference a tracked issue.
- **Local-only runner routing** — `scripts/check_local_only_workflows.py` (#7127).
- **Workflow inventory / action SHA pinning** — `scripts/check_workflow_inventory.py`,
  `scripts/check_github_actions_pinned.py`.
- **Security** — Bandit, Semgrep, detect-secrets, Trivy, pip-audit.

## Adding or changing a budget

1. Add the config file under `scripts/config/` (or the documented location).
2. Wire its enforcer into the `quality-gate` job.
3. Add a row above so the budget is discoverable.

`tests/scripts/test_quality_budgets_index.py` verifies every config path
referenced here exists, so this index cannot silently drift from reality.

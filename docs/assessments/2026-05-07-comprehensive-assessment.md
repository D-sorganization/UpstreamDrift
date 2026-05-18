# UpstreamDrift — Comprehensive A-O Health Assessment

**Date:** 2026-05-07
**Branch:** main
**HEAD:** `9fc03ba39df06acef46221a08322e305c3c734f8`
**Owner/Repo:** D-sorganization/UpstreamDrift
**Source LOC:** 415385
**Test LOC:** 231638
**Code Files:** 4156
**Branch Protection:** No

## Scores

| Criterion | Name                     | Score | Weight | Weighted  |
| --------- | ------------------------ | ----- | ------ | --------- |
| A         | Project Organization     | 52    | 5%     | 2.60      |
| B         | Documentation            | 100   | 8%     | 8.00      |
| C         | Testing                  | 75    | 12%    | 9.00      |
| D         | Error Handling           | 0     | 10%    | 0.00      |
| E         | Performance              | 50    | 7%     | 3.50      |
| F         | Code Quality             | 46    | 10%    | 4.60      |
| G         | Dependency Hygiene       | 90    | 8%     | 7.20      |
| H         | Security                 | 45    | 10%    | 4.50      |
| I         | Configuration Management | 100   | 6%     | 6.00      |
| J         | Observability            | 100   | 7%     | 7.00      |
| K         | Maintenance Debt         | 0     | 7%     | 0.00      |
| L         | CI/CD                    | 100   | 8%     | 8.00      |
| M         | Deployment               | 90    | 5%     | 4.50      |
| N         | Legal & Compliance       | 100   | 4%     | 4.00      |
| O         | Agentic Usability        | 90    | 3%     | 2.70      |
| **Total** |                          |       |        | **71.60** |

## Findings Summary

- **P0 (Critical):** 1
- **P1 (High):** 6
- **P2 (Medium):** 1

### P0 Findings

- **[H]** [UpstreamDrift] 10 potential hardcoded secrets detected

### P1 Findings

- **[A]** [UpstreamDrift] Top-level repository clutter (29 files)
- **[D]** [UpstreamDrift] 1 bare `except:` statements
- **[D]** [UpstreamDrift] 104 broad `except Exception` blocks
- **[D]** [UpstreamDrift] 883 lint/type suppressions
- **[K]** [UpstreamDrift] 883 lint/type suppressions
- **[L]** [UpstreamDrift] No branch protection on main

### P2 Findings

- **[F]** [UpstreamDrift] 22 TODO/FIXME/XXX items without tracked issues

## Full Evidence

```json
{
  "repo": "UpstreamDrift",
  "branch": "main",
  "head_sha": "9fc03ba39df06acef46221a08322e305c3c734f8",
  "head_date": "2026-05-06",
  "owner_repo": "D-sorganization/UpstreamDrift",
  "A": {
    "src_files": 2231,
    "test_files": 1281,
    "manifests": 2,
    "gitignore_lines": 215,
    "has_readme": 1,
    "clutter_files": 29
  },
  "B": {
    "readme_lines": 297,
    "readme_headers": 30,
    "docs_files": 357,
    "md_files": 9
  },
  "C": {
    "test_py": 1270,
    "test_rs": 0,
    "src_py": 1656,
    "src_rs": 0,
    "test_total": 1270,
    "src_total": 1656,
    "has_coverage": 0,
    "has_pytest_config": 1
  },
  "D": {
    "bare_except": 1,
    "except_exception": 104,
    "noqa_suppressions": 883
  },
  "E": {
    "benchmark_files": 0,
    "cache_decorators": 0
  },
  "F": {
    "todo_fixme": 22,
    "duplicate_risk": 0
  },
  "G": {
    "req_lockfiles": 1,
    "req_files": 2
  },
  "H": {
    "secrets_raw": 10,
    "bandit_cfg": 0,
    "security_md": 1
  },
  "I": {
    "env_example": 1,
    "config_files": 66
  },
  "J": {
    "logging_refs": 741,
    "metrics_refs": 458
  },
  "K": {
    "suppressions": 883,
    "todo_total": 22
  },
  "L": {
    "workflow_files": 66,
    "precommit_config": 1
  },
  "M": {
    "dockerfile": 1,
    "compose_files": 1
  },
  "N": {
    "license": 1,
    "copyright_headers": 94,
    "contributing": 1
  },
  "O": {
    "claude_md": 1,
    "agents_md": 1,
    "claude_lines": 91,
    "agents_lines": 8
  },
  "code_files": 4156,
  "src_loc": 415385,
  "test_loc": 231638,
  "branch_protection": false
}
```

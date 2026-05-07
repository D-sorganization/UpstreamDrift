# Review Comments Archive - 2026-05-06

Generated: 2026-05-06T19:08:52.014866

## Reviewer (chatgpt-codex-connector[bot]) (3 comments)

### PR #4226: tests/unit/motion_matching/test_cross_option_leaderboard.py:95

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Remove machine-specific cwd from subprocess tests**

The test invokes the CLI with a hard-coded working directory (`/home/dieterolson/Repositories-WSL/UpstreamDrift`), which does not exist in CI or other developer environments, causing `FileNotFoundError` before the script even runs. This makes the new unit tests fail outside the original author’s machine and blocks the test suite.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4226#discussion_r3198554317)

---

### PR #4226: scripts/run_cross_option_leaderboard.py:473

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Load existing results when --skip-fits is requested**

The `--skip-fits` path returns immediately with an empty `LeaderboardSummary`, so the generated report and metrics claim zero attempted/successful fits even when result JSON files already exist. This contradicts the CLI contract (“regenerate reports from existing JSONs”) and produces misleading output for report-only runs.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4226#discussion_r3198554319)

---

### PR #4226: scripts/run_cross_option_leaderboard.py:817

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Avoid serializing Infinity in metrics JSON**

When no fit succeeds, `best_grip_rmse_mm` remains `float('inf')` and is written directly into `metrics_dict`; `json.dumps` emits this as `Infinity`, which is non-standard JSON and breaks strict JSON parsers used by many downstream tools. The metrics output should normalize this sentinel to a finite value or `null` before serialization.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4226#discussion_r3198554322)

---


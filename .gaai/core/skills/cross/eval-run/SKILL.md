---
name: eval-run
description: Evaluate any output file against a structured evals.yaml assertions file and produce a score report with per-assertion pass/fail results. Activate when the Discovery Agent runs the Skill Optimize protocol to measure output quality or detect regressions after skill instruction changes.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-CRS-025
  updated_at: 2026-03-15
  status: experimental
inputs:
  - output_file: path to the output file being evaluated (produced by any skill)
  - evals_file: path to the evals.yaml file containing assertions to run
outputs:
  - score report (YAML or structured Markdown) with per-assertion pass/fail, total score, and failed assertion details
---

# Eval Run

## Purpose / When to Activate

Activate when:
output_file: { path }
evals_file: { path }
run_date: { ISO 8601 }
score:
passed: 4
total: 5
ratio: "4/5"
results: - id: A01
type: code
description: "Word count within ±15% of target"
result: PASS
details: "1247 words (range: 1020–1380)" - id: A02
type: code
description: "Kill list word 'leverage' absent"
result: FAIL
details: "2 matches found"
failed_assertions: - id: A02
description: "Kill list word 'leverage' absent"
type: code
check: regex_not_match
pattern: "\\bleverag(e|ing|ed)\\b"
details: "2 matches found at positions [line 4, line 11]"

```

---

## Non-Goals

This skill must NOT:
```

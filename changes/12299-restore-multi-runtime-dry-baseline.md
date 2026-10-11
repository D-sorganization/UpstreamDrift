---
issue: 12299
summary: "Restore the approved multi-runtime DRY baseline lost during shot-pattern integration"
dl_state: "in_progress"
next_step: "Complete restored-baseline and canonical pre-PR checks before protected integration"
branch: "codex/shot-pattern-refinement-integration"
---

Restores exact approved main bytes instead of generating new allowances. All
588 logged failure fingerprint prefixes existed in the approved baseline.
Evidence and unresolved validation are recorded in
`docs/research/shot_pattern_analysis/phase2/validation/refinement_v3/dry_baseline_restoration.json`.

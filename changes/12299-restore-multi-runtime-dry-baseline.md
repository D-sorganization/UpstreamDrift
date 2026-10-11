---
issue: 12299
summary: "Restore the approved multi-runtime DRY baseline lost during shot-pattern integration"
dl_state: "in_progress"
next_step: "Complete restored-baseline and canonical pre-PR checks before protected integration"
branch: "codex/shot-pattern-safe-followup"
---

Restores exact approved main bytes instead of generating new allowances. All
588 logged failure fingerprint prefixes existed in the approved baseline.
Evidence and unresolved validation are recorded in
`docs/research/shot_pattern_analysis/phase2/validation/refinement_v3/dry_baseline_restoration.json`.

The successor reconciles remote `9c9000`, preserves the same approved baseline,
and hashes all copied fixture source files for frozen-verifier origin checks.
Successor validation and unresolved gates are recorded in
`docs/research/shot_pattern_analysis/phase2/validation/safe_followup_v1/progress.json`.

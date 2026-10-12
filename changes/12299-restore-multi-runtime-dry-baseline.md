---
issue: 12299
summary: "Restore the approved multi-runtime DRY baseline lost during shot-pattern integration"
dl_state: "in_progress"
next_step: "Wait for separately owned development-log issue #11935, rerun complete canonical pre-PR unchanged, then follow protected integration"
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

The local successor at repaired source `5fab697b63` passed 67 focused tests on the 126/126 dev-lock runtime, plus the restored-baseline DRY, architecture, file-size, Rust import, and contact checks. The final attempted full pre-PR passed mypy on 148 files but stopped before pytest because the separate required development-log validator fails repository-wide (#11935, held by another active session). Exact failed and incomplete receipts are indexed in `safe_followup_v1/progress.json`; this local checkpoint is not accepted or pushed.

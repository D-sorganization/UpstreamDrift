---
issue: 12242
summary: "Keep Research PDF Headings With Their Following Tables and Paragraphs"
dl_state: "in_review"
next_step: "Complete the final canonical gate and live publication verification, then deliver through protected merge; no merge claimed yet."
branch: "codex/shot-pattern-main-integration"
---

## Current Review Checkpoint

2026-10-12: Integrated on `codex/shot-pattern-main-integration` for review; not yet merged. The original 30-page v9 pagination check below is historical. The [39-page delivery receipt](../docs/research/shot_pattern_analysis/phase2/validation/refinement_v3/delivery_receipt.json) records the final 39-page illustrated PDF, SHA256 `fd5506da87f5e619a550fb8cc871a0e8195b6fb98965a99809e3238a0f01508b`, all-page root review and the scoped independent review. Its local compiler succeeded; the built-in compiler bundle limitation remains separately tracked as #12120. No new scientific qualification is implied by pagination acceptance.

Complete the final canonical gate and live publication verification before protected merge. This checkpoint does not close deferred physical validation, renew historical scientific approvals, or claim completed integration acceptance.

## Historical Checkpoint

The prior checkpoint is preserved below as history, not current acceptance.

Prior state: `in_progress`. Prior branch: `codex/shot-pattern-main-integration`.

Prior next step: Integrate accepted PDF repair into the final shot-pattern publication PR

Two pagination regressions failed before repair; all nine PDF tests and Ruff lint/format now pass. Root visually reviewed all 30 v9 report pages and verified identical extracted body token sequence against v8. Receipt: phase2/validation/final_report_pdf_v9/visual_review.json (study checkout), SHA256 3672fd8e3d4fd5fb468df07e0e21548287d46df9eed34f2f3abd6e6f0e9cb853. Scientific narrative acceptance is separate and pending. Issue remains open until a merged PR.

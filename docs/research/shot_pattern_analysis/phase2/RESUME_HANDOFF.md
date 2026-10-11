# Shot Pattern Study: Agent Resume Handoff

## Current State

The goal is active after the user resumed. All 10.44 million attempted simulation
rows are complete. Do not rerun simulations. The scientific report, all assigned
baseline PNG reviews, and the 49 selective density-layout replacement reviews
are complete within their recorded scopes. Exact aggregate coverage, final
integration gates, and remote delivery are being finalized.

Astra accepted the narrative after 406 numeric/count checks and 52 bound-source
hash checks: `validation/final_narrative_independent_review_v8/review.json`.
The unchanged `REPORT.md` SHA256 is
`ff281f76ee649c44c88afda58701329e7898e8964f9efb6d8aeec73dcd77d79b`.
All 30 pages of report PDF v9 were visually reviewed; nine pagination tests pass
and body tokens match v8. See `validation/final_report_pdf_v9/visual_review.json`.
This report PDF is separate from the built-in LaTeX editor preview.

## Checkouts and Ownership

All checkouts are under `/home/dieterolson/.codex/worktrees/`:

| Role | Checkout | Branch / Checkpoint |
| --- | --- | --- |
| Study, Data, and Review Receipts | `shot-pattern-analysis/UpstreamDrift` | `codex/shot-pattern-matched-delivery`; after `242c9c7074`, Sol owns final evidence consolidation |
| Delivery Integration | `shot-pattern-main-integration/UpstreamDrift` | `codex/shot-pattern-main-integration`; `4efe168c3a` plus current delivery documentation |
| Frozen Producer | `shot-pattern-presentation/UpstreamDrift` | `codex/shot-pattern-density-gutter`; `4023111d2aaf8b20449a36f8691a295f4a94ceff` |
| Selective Layout Repair | `shot-pattern-layout-gutter/UpstreamDrift` | `codex/shot-pattern-layout-gutter`; `f7b3ade771`, source repair `5a1fdbfe51` |

Luna owns native-format receipts, archive restoration, and delivery preparation.
Root owns current handoff documentation and final coordination. Astra's scientific
and assigned graphic reviews are complete. Do not mutate the armed Phase 1 PR
#11941. Preserve the frozen producer and its historical Tools pin `3678409`.
Delivery uses Tools `86d0f28b1cc5acf61185e07e320c816c2d005512` and already includes
precondition guard `a221007665`, PDF repair `2110d4a59b`, and density repair
`44c774fff6`. No pending guard cherry-pick is needed.

## Graphics Acceptance and Provenance

Review evidence lives in `validation/supplement_v6_final_graphics_review_v2/`.
The original v6 publication has 384 ordered pairs, 26 unique cells, and 6,456
assets. Baseline PNG holds concern 49 academic density triples; these are
replaced in `supplement_report_v2_layout_repair`, leaving all numerical outputs
unchanged. Issue #12243 records the layout repair.

- `layout_child_reconciliation_receipt.json`: exactly 147 graphic files changed;
  6,309 other assets and seven numerical report files are unchanged.
- `layout_child_strict_loader_receipt.json`: all 6,456 assets and 384 pairs pass.
- `layout_child_clearance_receipt.json`: all 384 density SVGs pass the measured
  legend/title and title/colorbar clearance checks.
- Root's 19 curve PNGs, Sol's 15 shaft PNGs, and Astra's 15 shaft PNGs have
  separate actual visual reviews. Native-format review covers all 49 changed
  PDFs and 49 changed SVGs against their replacement PNGs.

Original native checks live in `validation/supplement_v6_vector_review_v2/`.
All 2,024 PDFs and 2,024 SVGs rendered. The 1,122 PDF bounding-box warnings were
checked using 10,842 painted-ink edge probes: minimum visible clearance 6.4952 pt,
no ink in the outer 2 pt bands, and no render failures. The renderer completed
all rows but exited 1 while serializing its summary; a separate finalizer exited
0. `bbox_painted_ink_boundary_review.json` preserves this distinction and hashes.
Do not infer acceptance from a successful render alone.

## Artifact Delivery

Use [Artifact Delivery Plan](review_followups/ARTIFACT_DELIVERY_PLAN.md).
Accepted primary/control bundles are under `/tmp/shot-pattern-publication-bundles-v1/`.
The child supplemental bundle is `/tmp/shot-pattern-delivery-resume/supplement-v2`:
six ZIPs, each at most 500,000,000 bytes; manifest SHA256
`5d776ea73d9564812431416a01c2661a3f9335de1554ef111d24673b23125f74`.
Fresh local restorations verified all 690 primary, 548 control, and 6,465
supplement files by inventory, size, and hash. This is offline evidence, not
remote delivery. Its receipt is
`/tmp/shot-pattern-delivery-resume/local_fresh_restoration_receipt_v2.json`.

The reader publication tree must map the repaired child directory to
`supplement_report_v1/`, matching the unchanged report's existing links. All 98
unique supplemental report links resolve in the child. Keep the original study
baseline separate and intact. Remote upload must be followed by a fresh download,
trusted manifest comparison, and full archive/member verification.

## Remaining Gates and Blockers

1. Integrate Sol's final evidence/coverage commit into delivery, preserving its
   current root handoff and peer index. Never reset delivery to the older study tree.
2. Commit current documentation and final archive receipts. Run canonical RM
   `python -m scripts.pre_pr` with Repository_Management first on `PYTHONPATH`.
   Resolve owned regressions; record unrelated repository failures truthfully.
3. Push the delivery branch, publish and verify the immutable artifact release,
   and open/attach a ready PR with a Handoff section and change fragments. Arm
   only through `scripts/automerge_guard.py`; never bypass hooks or protection.
4. Built-in LaTeX compilation remains blocked before source parsing by Tectonic's
   unavailable bundle fetch (#12120). Latest evidence:
   `validation/latex_compile_resume_v9/receipt.json`. Keep the existing article
   open; no alternate LaTeX PDF proves that the right-panel compiler works.

## Scientific Limits and Reproduction

Read [Report](REPORT.md), [Design](DESIGN.md), [Reproduction](REPRODUCING.md), and
[Turnover](TURNOVER.md). Results are geometric/model-conditional, not measured
player data. Primitive shaft-twist SD differs from delivered-face SD. Matching
nominal landing points changes speed/face/path; it does not isolate a causal
curvature effect. Individual, common-three, and common-six support populations
are different. Scoring uses assumed course geometry and a historical baseline;
no per-round or universal best-pattern claim is justified. Empirical validation
#11984 remains deferred. Unsupported cases remain visible.

Historical stop and intermediate execution details are preserved in git and
Turnover; this document gives current state only. No new remote push, release,
or Phase Two PR is claimed until its own delivery receipt exists.

# Shot Pattern Study: Agent Resume Handoff

## Current State

The goal is active after the user resumed. All 10.44 million attempted simulation
rows are complete. Do not rerun simulations. The scientific report, all assigned
baseline PNG reviews, and the 49 selective density-layout replacement reviews
are complete within their recorded scopes. Exact aggregate coverage accepts 2,304 pair PNGs and 104 cell PNGs.
Repository gate failures and verified research prerelease delivery remain
explicit below.

Astra accepted the narrative after 406 numeric/count checks and 52 bound-source
hash checks: `validation/final_narrative_independent_review_v8/review.json`.
The unchanged `REPORT.md` SHA256 is
`ff281f76ee649c44c88afda58701329e7898e8964f9efb6d8aeec73dcd77d79b`.
All 30 pages of report PDF v9 were visually reviewed; nine pagination tests pass
and body tokens match v8. See `validation/final_report_pdf_v9/visual_review.json`.
This report PDF is separate from the built-in LaTeX editor preview.

The phase-two-first illustrated LaTeX article is accepted for document delivery:
33 pages, 13 source-bound figures, zero Tectonic warnings. Final PDF SHA256
`ca030b3c97fb759b384aab58b3da6be68687a194f353976754d43e7ffa69beda`; review
receipt: `validation/illustrated_article_v1/review.json`. Local compiled output is
`validation/illustrated_article_v1/final_build/shot_pattern_analysis.pdf` in the
STUDY checkout. The user-facing PDF is titled “Shot Pattern Analysis - Illustrated
Research Report.pdf”. Its source archive is
`/tmp/shot-pattern-delivery-resume/illustrated-report-source.zip` (15 members;
SHA256 `6567a9dcee60d6e18e4b1fe39ca34b3631eac09a578ce3eacd64914c93c7c1d2`).
The built-in Codex LaTeX editor/compiler still fails before parsing on its
Tectonic bundle fetch (#12120); standalone CLI success does not clear that issue.
The local compile command, run from the article directory, was:

```bash
tectonic --keep-logs --keep-intermediates \
  --outdir phase2/validation/illustrated_article_v1/final_build \
  shot_pattern_analysis.tex
```

## Checkouts and Ownership

All checkouts are under `/home/dieterolson/.codex/worktrees/`:

| Role                             | Checkout                                      | Branch / Checkpoint                                                                              |
| -------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Study, Data, and Review Receipts | `shot-pattern-analysis/UpstreamDrift`         | `codex/shot-pattern-matched-delivery`; after `242c9c7074`, Sol owns final evidence consolidation |
| Delivery Integration             | `shot-pattern-main-integration/UpstreamDrift` | `codex/shot-pattern-main-integration`; `4efe168c3a` plus current delivery documentation          |
| Frozen Producer                  | `shot-pattern-presentation/UpstreamDrift`     | `codex/shot-pattern-density-gutter`; `4023111d2aaf8b20449a36f8691a295f4a94ceff`                  |
| Selective Layout Repair          | `shot-pattern-layout-gutter/UpstreamDrift`    | `codex/shot-pattern-layout-gutter`; `f7b3ade771`, source repair `5a1fdbfe51`                     |

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
all rows but exited 1 while serializing its summary; a separate finalizer exited 0. `bbox_painted_ink_boundary_review.json` preserves this distinction and hashes.
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

The reader publication tree maps the repaired child directory to
`supplement_report_v1/`, matching the unchanged report's existing links. All
98 supplemental report link occurrences and 98 unique targets resolve in the
child. Keep the original study baseline separate and intact. The research
prerelease is published at
[shot-pattern-phase2-research-2026-10-10](https://github.com/D-sorganization/UpstreamDrift/releases/tag/shot-pattern-phase2-research-2026-10-10),
bound to DELIVERY checkpoint `cb5eff6dcd9df7d878f10578078b6fea69c8b681`.
Fresh download verification passed for all 21 assets, the primary/control/
supplement inventories (690/548/6,465 members), the 15-member source archive
with 13 selected figures, and all 98 report links. The durable receipt is
`validation/artifact_delivery_v3/remote-restoration-v3.json.gz`.

## Remaining Gates and Blockers

1. Final review evidence is integrated. Never reset delivery to the older study
   tree. Preserve its current root handoff and peer index.
2. Canonical RM pre-PR ran at `dac6618c58`: Ruff, mypy, consistency, fragments,
   and workflow checks pass. Affected tests: 5,861 passes, 34 failures, seven
   errors in unchanged outside-study test files. That is path-scope evidence, not
   an independent runtime baseline reproduction. Fleet development-log policy
   fails. The raw-XML import-policy finding was repaired in `c8dff74a0a`; 41 focused
   tests, Ruff, mypy, and central import-policy checks pass, with unavailable
   Semgrep advisory-skipped. Full log and receipts live
   in `validation/artifact_delivery_v2/`. Do not waive or describe these gates as green.
   Focused base-commit checks reproduce 38 nonpassing statuses among the 41
   original nodes; the other three pass in isolation at both base and delivery.
   Their full-suite failures remain unresolved; see `validation/baseline_gate_audit_v1/`.
   A corrected-PYTHONPATH full pre-PR attempt later terminated with exit 143
   during tests, before a final summary; it is incomplete and does not replace
   the failed complete gate.
3. The 21-asset v3 inventory is published at the research prerelease linked
   above, and fresh remote download/member verification passed. Open/attach a
   reviewable PR with explicit blocked gates, a Handoff section, and change
   fragments. Never bypass hooks or protection.
   The first restoration attempt found a manifest-to-release-name mapping
   mismatch after all 21 asset hashes passed. Sol reviewed the corrected
   bijective size/SHA mapping; a fresh isolated rerun exited 0 with no asset
   changes. `validation/artifact_delivery_v3/remote_verification_summary.json`
   records both attempts; the gzip remote-restoration receipt records the
   successful full byte verification.
4. The illustrated article has a successful local Tectonic CLI compile and accepted
   33-page visual review. The separate built-in LaTeX editor remains blocked before
   parsing by its bundle fetch failure (#12120); local CLI success does not resolve
   that editor issue. The accepted PDF and source archive are included in the verified v3
   inventory; remote restoration is verified by the receipt above.

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
Turnover; this document gives current state only. The prerelease is a research
artifact delivery, not production or merge acceptance. The Phase Two PR [#12278](https://github.com/D-sorganization/UpstreamDrift/pull/12278)
is open, ready for review, and labelled blocked. Its PR body is the latest
published handoff. A separate user-authorized validation follow-up is active on
`codex/shot-pattern-validation-followup`; the remote PR head observed at its
start was `8c5ddfabc831c071450e1124caf222ac364326d9`. The current CI unit gate
reported 13 failures and 25,132 passes. These newer results do not retroactively
clear the original failed or incomplete local gates. Preserve every frozen
output and publication receipt; do not remove the blocked label without
complete qualifying validation.

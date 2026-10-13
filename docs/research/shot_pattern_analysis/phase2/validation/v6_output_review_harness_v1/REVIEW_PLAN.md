# V6 Output Review Plan

This is a read-only component review. Neither source execution nor a complete-file contact sheet establishes visual acceptance.

1. After the producer has completed successfully with a hash-bound terminal receipt, `review_batch.py` checks all seven against the committed expected hash map before reading figures. It imports the exact public pair identity/slug helper from the frozen producer checkout. No substring matching or old cell-only filenames are used.
2. Review cells (26, 312 assets) and all three pair groups (56 across-lean, 164 curve-strength, 164 shaft-SD; 6144 assets) in bounded index batches. Each complete pair requires all 16 formats; each complete cell requires all 12. While the producer is live or failed, the entire requested batch remains pending without reading numerical or figure outputs. After a completed boundary, missing or changed component files are integrity failures. No polling loop or quiet-time completion assumption.
3. Decode every PNG, preserve every PNG/PDF/SVG hash, and extract SVG captions. Social-density protocol, matched target, and common-six count are checked against the numeric record. Contacts contain only two charts each; inspect all contacts, then full-size examples for every club, contrast, source-protocol combination, covariance profile, narrow range, and low-support case. Examine academic density row gutter, long caption containment, main/marginal ticks, legends, and honest raw-flight versus endpoint populations. Review PDF text/page scope and SVG text alongside PNG visuals; hashes alone do not qualify these formats.
4. Store each batch in a new evidence directory. The initial batch inventory explicitly says visual review and final manifest reconciliation are pending. Add separate visual decisions with exact inspected file hashes; do not rewrite the initial snapshot.
5. Reconcile the union of all reviewed identities with all 384 numerical records and 26 cells. Do not reuse the old 344-name inventory or infer scope from a total count. Prior visual evidence may be reused only for exact matching asset bytes; new paths and metadata must still reconcile to exact protocol identities.
6. After producer exit 0 and its final receipt appear, invoke the unchanged strict public loader from the exact frozen producer source, verify its 15-source map and bound inputs/reference, verify the full artifact ledger and final file hashes, and reconcile every reviewed snapshot. Require exactly the independently preflighted 6456 intended outputs unless the actual ledger explicitly documents a supported-status exception; any unexpected missing output is a finding, not a changed acceptance criterion.
7. Only then write whole-publication acceptance. The report and editable reference remain separate deliverables requiring their own final rendering/claims review.

## Example Bounded Snapshot

Run from any directory using the qualified Python environment:

```bash
/tmp/shot-pattern-validation-venv-20261009/bin/python /tmp/shot-pattern-v6-output-review/review_batch.py \
  --repo /home/dieterolson/.codex/worktrees/shot-pattern-presentation/UpstreamDrift \
  --report /home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/phase2/supplement_report_v1 \
  --completion-receipt /home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/phase2/validation/supplement_rendering_revision_v6/exit.json \
  --start-receipt /home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/phase2/validation/supplement_rendering_revision_v6/start.json \
  --start-sha256 ab2d3b97b8be26a0fbe4bfa7f2fabaccefb8de602493f76a6d07d2682595bfc0 \
  --expected-numeric /home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/phase2/validation/protocol_name_recovery_v6/expected_numeric_hashes.json \
  --group pairs --kind across_lean --offset 0 --limit 12 \
  --output /tmp/shot-pattern-v6-output-review/across_lean_000_011
```

The harness has 22 passing fixture/negative tests plus Ruff/format checks. It has not been executed against production v6. The completion gate uses the actual wrapper's `--output-dir` command shape, compares every start-record field except status with a SHA-pinned start receipt, then verifies pipeline receipt, producer source map and figure ledger. The complete command comparison rejects changed inputs or additional flags even when a forged terminal record claims success. All runtime validation uses explicit exceptions, not Python `assert`.

The six source-bound SVG fixtures cover three saved pair records and all three cell protocols. Pair `source_protocols` is present in the saved schema and must agree with ordered `sources[*].protocol_id`; common support is `common_supported_n / total_n_per_population`. Cell per-pattern support is nested under `summary.summary[pattern].supported_n`, and the total is `config.n_shots`. Numeric target captions use two displayed decimals while filenames retain full exact target identity.

Actual images still require human/agent visual inspection. These operational tests qualify the review harness, not v6 graphics or the scientific model.

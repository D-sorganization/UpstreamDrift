# Reference Alignment Controls Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-reference-controls
- Branch: feat/9883-reference-alignment-controls
- Baseline commit: 66a7f017d (rendering #9888; parent #9885)
- Implementation commit: SELF
- Pull request: not created
- Governing issue/epic: #9883; advanced reference epic #9863

## Objective and Status

Professional comparison editing: model placement, expert image adjustment, paired swing events, independent expert scrubbing, lesson notes and recoverable settings. Implementation is locally qualified; parent merges and protected review remain pending.

## Files and Decisions

reference_controls.py uses the shared pose6dof rotation math and validates registrations through the existing models. Image adjustments preserve perspective. reference_timeline.py owns an optional expert decoder, labels missing time coverage and provides independent scrubbing for event pairing. reference_comparison.py separates Placement, Timing and Notes, preserves pending edits on switch/close, supports undo/reset, shortcuts and explicit manual review of changed evidence. Saving reviewed settings backs up the previous sidecar. Failed saves do not mutate the saved baseline. Corrupt settings are reported without replacement. styling.py sizes inspector tabs without overriding theme colors. The generated capability atlas now includes comparison inputs and deliverables.

## Validation

Forty alignment, renderer and state regressions passed. Seven alignment scenarios passed again after layout adjustments. Three-module mypy passed before the final compact-tab style addition; final checks remain below. Full-PR architecture budget passed. LoD no-growth passed with 490 baseline occurrences and 60 reductions; DRY no-growth passed with 666 historical fingerprints. No baselines changed. Synthetic native screenshots at 1280x800 and 900x740 exposed hidden tabs and a buried pairing action; both were adjusted. Screenshots are layout evidence, not real calibration validation.

## Blockers and Risks

Parent timing #9885 and rendering #9888 remain protected-review dependencies. Another agent has pushed to #9885; those changes are preserved by normal merges and central scope notices. Keep one SPEC row per PR; the fleet merge driver can restore an old duplicate #9879 row, so run the duplicate gate after integration. No other worktree was modified.

## Next Steps

Commit this isolated work, merge current rendering, run focused validation and all required gates, create the protected PR and finish the advanced epic only after all dependencies merge.

## Change Log

- SELF: Implement instructor controls, guarded workflow and generated comparison map; update DL-#9883 in place.

# Reference Alignment Controls Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-reference-controls
- Branch: feat/9883-reference-alignment-controls
- Baseline commit: 66a7f017d (rendering #9888; parent #9885)
- Implementation commit: SELF
- Pull request: #9890 (draft; #9885 merged; rendering #9889 dependency)
- Governing issue/epic: #9883; advanced reference epic #9863

## Objective and Status

Professional comparison editing: model placement, expert image adjustment, paired swing events, independent expert scrubbing, lesson notes and recoverable settings. Implementation is locally qualified; parent merges and protected review remain pending.

## Files and Decisions

reference_controls.py uses the shared pose6dof rotation math and validates registrations through the existing models. Image adjustments preserve perspective. reference_timeline.py owns an optional expert decoder, labels missing time coverage and provides independent scrubbing for event pairing. reference_comparison.py separates Placement, Timing and Notes, preserves pending edits on switch/close, supports undo/reset, shortcuts and explicit manual review of changed evidence. Saving reviewed settings backs up the previous sidecar. Failed saves do not mutate the saved baseline. Corrupt settings are reported without replacement. styling.py sizes inspector tabs without overriding theme colors. The generated capability atlas now includes comparison inputs and deliverables.

## Validation

Forty alignment, renderer and state regressions passed. Seven alignment scenarios passed again after layout adjustments. Three-module mypy passed before the final compact-tab style addition; final checks remain below. Full-PR architecture budget passed. LoD no-growth passed with 490 baseline occurrences and 60 reductions; DRY no-growth passed with 666 historical fingerprints. No baselines changed. Synthetic native screenshots at 1280x800 and 900x740 exposed hidden tabs and a buried pairing action; both were adjusted. Screenshots are layout evidence, not real calibration validation.

## Blockers and Risks

Timing #9885 is merged. Rendering #9888 was superseded by compatible #9889; its pending merge remains a dependency. Another agent has pushed to #9885; those changes are preserved by normal merges and central scope notices. Keep one SPEC row per PR; the fleet merge driver can restore an old duplicate #9879 row, so run the duplicate gate after integration. No other worktree was modified.

## Next Steps

Commit this isolated work, merge current rendering, run focused validation and all required gates, create the protected PR and finish the advanced epic only after all dependencies merge.

## Change Log

- SELF: Implement instructor controls, guarded workflow and generated comparison map; update DL-#9883 in place.

## Rendering Integration

Merged rendering 7e06f8b68 and timing de430c500, including concurrent peer #9886 compatibility and the shared encoder path. Current settings UI and its canonical handoff remain authoritative for #9883. Rechecked SPEC duplicates after the merge driver ran.

Alignment Undo preserves subsequently written lesson notes; an explicit regression covers it. Removed the old #9882 SPEC row restored by the merge driver, retaining #9888. Re-run duplicate validation after every stack merge.

Final screenshots refreshed and visually inspected: all three tabs fit the desktop inspector; Pair Current Frames is visible on the laptop; lesson notes remain accessible in the scrolling Notes tab. Four-module mypy and 67 integrated regressions pass.

Restored standing design-manual governance and earlier task context in AGENT_HANDOFF.md after the doc-governance test exposed its removal. Canonical current state remains here. Run tests/scripts/test_design_manual_governance_contract.py before readiness.

Merged final sampler bounds 7bd6b4d0c, including its five adverse regressions and refreshed source-hashed benchmark. No alignment UI behavior changed.

Integrated peer rendering #9889 (a7c224e2e): preserved the newer instructor UI, strict export implementation, and five legacy gap-bound regressions. Retained the independent scene-evidence fixture. Existing registration already validates camera/clock view identity; the duplicate peer validator is unnecessary.

Final integration: 88 focused alignment/render/state/registration/timing/evidence/governance tests passed. Full architecture, module and file budgets passed. Regenerated atlas after source-hash drift in the merge. Broad mypy reports five pre-existing dependency errors in keypoint_offsets.py, \_unit_contracts.py, trc_adapter.py and rtmpose_onnx_estimator.py; explicit edited-module checking is recorded separately.

Explicit mypy --follow-imports=silent passes all four edited UI modules. Atlas freshness passes after regeneration.

Rendering #9889 merged as a506a2958. Integrated that exact main tree; it equals the reviewed peer head. Retained the previously validated instructor UI and a single camera/clock validator. Both dependencies are now merged.

Preserved remote rewrite e8a070d11 by a normal merge. Its product modules match our validated interface, but omitted evidence was retained: five gap-bound tests, benchmark, screenshots and canonical governance/handoff records. No force push.

CI exposed the expanded fleet-managed AGENTS.md exceeding the unchanged 50 KiB documentation budget. Moved the detailed local infrastructure directory into docs/agents/shared-infrastructure.md, adjusted relative links and title case, and retained the required entry-point discovery workflow plus every managed block. Registered the directory in the catalog.

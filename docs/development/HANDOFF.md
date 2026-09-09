# Fleet Guide Compatibility Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-reference-controls
- Branch: fix/9892-fleet-guide-validation
- Baseline commit: 271662f68 (instructor PR #9890)
- Implementation commit: SELF
- Governing issue: #9892; fleet adoption Repository_Management#1579
- Pull request: pending

## Objective and Status

Keep centrally generated guidance while allowing the guide checker to distinguish required local files from optional or explicitly hub-owned references. Separated from #9890 after a concurrent push removed its policy to clear CI. No further interface changes are needed in this follow-up.

## Files and Decisions

Restore the generated CLAUDE sections and canonical spec-heading/script-owner corrections from Repository_Management#1627. The checker ignores only exact managed notices and separator paragraphs; substantive duplicate instructions still fail. Optional references are recognized within their sentence, and explicitly Repository_Management-owned paths are external. Required local paths still fail, including mandatory references on the same line before an optional sentence.

## Validation

Two reproductions failed before implementation. All 27 guide-consistency tests and the live checker pass. The inherited instructor feature has 88 focused feature/evidence/governance passes, four-module mypy, source-hashed sampling evidence and desktop/laptop synthetic screenshots. Document catalog, size, title and SPEC duplicate checks pass. No guard or baseline is disabled.

## Blockers and Risks

The interface PR #9890 remains separately owned through completion. This branch merges its concurrent 271662f68 revision while retaining policy with the qualified checker. Central policy changes are in Repository_Management#1627. User clarification is pending on repeated policy-PR closures and half-ton runner access; this follow-up does not resolve those external blockers.

## Next Steps

Commit and publish a focused PR against main after #9890 merges; complete protected checks and restore default-branch policy adoption. Keep this issue separate from the advanced motion epic. Preserve root standing design-manual context and update DL-#9892.

## Change Log

- SELF: Separate tested fleet-guide compatibility from instructor delivery without force pushes.

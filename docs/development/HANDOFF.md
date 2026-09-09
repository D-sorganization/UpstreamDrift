# Capability Atlas Handoff

## Identity

- Repository: `D-sorganization/UpstreamDrift`
- Working directory: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-capability-atlas`
- Branch: `feat/9850-capability-atlas`
- Baseline commit: `384cba80e6249afb0247c97d18536c0335c054d6`
- Implementation commit: `SELF`
- Pull request: #9856
- Governing issue/epic: #9850, children #9852/#9853; parent product #9849.
- Development log: `DL-#9850`. Previous #9784 handoff remains in git.

## Objective and Status

- Generate maintained architecture/capability/workflow references from existing registries.
- Status: in progress; graph, offline browser atlas, Mermaid exports, navigation and freshness tests implemented.
- Remaining: final browser/format/type checks, protected PR and merge.

## Files and Decisions

- `scripts/capability_atlas/`, `scripts/generate_capability_atlas.py`: validated source model and deterministic outputs.
- `src/config/capability_connections.json`: semantic edges with artifacts, source evidence and limitations.
- `ui/public/capability-atlas/`: offline searchable HTML/SVG reference, JSON and Mermaid diagrams.
- Existing Project Map and system overview link to generated `CAPABILITY_ATLAS.md`; web launcher has Capability Map link.
- `.prettierignore` excludes byte-exact generated outputs; generators and freshness tests own them.
- Reuse 57 canonical tiles/42 feature contracts; preserve single-view limitations and web gaps. No live-health or scientific validity claims.
- User-owned changes: shared checkout untouched. Avoid #9843 camera GUI scope and the active optimization task's source/CI.

## Validation

- TDD red recorded for missing model/generator and launcher navigation.
- `python3 -m pytest tests/scripts/test_capability_atlas.py -q -n 0 --timeout=60 --no-cov` — first 7 tests passed; navigation/source-link tests added afterward.
- `python3 -m mypy scripts/capability_atlas scripts/generate_capability_atlas.py --follow-imports=silent` — passed.
- `python3 -m ruff check scripts/capability_atlas scripts/generate_capability_atlas.py tests/scripts/test_capability_atlas.py` — passed before final test additions.
- Browser rendered via loopback; source-link normalization and gap wording improved after visual inspection.

## Blockers and Risks

- No source blocker; CI and protected merge remain required.
- Existing source metadata can be stale; atlas preserves declared status and gives source evidence, not runtime certification.
- Source changes require one-command regeneration; pinned Tools submodule must be initialized.

## Next Steps

1. Run formatter on sources, regenerate outputs, verify freshness and browser filters.
2. Commit/push through hooks; create protected PR; preserve parallel SPEC/handoff entries during merge.
3. Continue product/performance #9851 on its separate worktree and check GUI #9843 progress.

## Change Log

- `SELF` — Build maintained architecture/capability atlas and record source/validation boundaries.

- SELF — Verified web launcher type-check, search/empty/gap filters, and Tools source URLs in both HTML and Markdown.

- SELF — Integrated origin/main 19a390b78, preserving concurrent optimization/UI work and regenerating maps from the updated parity registry.

- SELF — Preserved remote agent merge 3aecbb6a4 with a normal merge; generated outputs remain based on the integrated canonical registry.

- SELF — Added atlas freshness to the required code-quality job after its existing project dependency installation. Parser-only probe exposed transitive dependencies in the canonical registry loader; the structural parser environment stays minimal. Ten atlas tests pass.

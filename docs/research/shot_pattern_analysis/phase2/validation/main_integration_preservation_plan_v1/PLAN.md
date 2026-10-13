# Main Integration Preservation and Regeneration Plan

Read-only review of `2953c52b37` versus fetched `fa7ee92f5c`; no integration worktree accessed, network requests made, or repository files changed. The merge-tree inventory lists 12 conflicted paths. Auto-merged canonical registries still require semantic review; a clean textual merge is not validation.

## Canonical Inputs Before Generated Views

| Conflicts | Authority and Resolution |
|---|---|
| `docs/architecture/CAPABILITY_ATLAS.md`, `ui/public/capability-atlas/{graph.json,index.html}` | Regenerate using merged `scripts.generate_capability_atlas`. Its `outputs(root)` also emits system/capture-workflow/capture-goals Mermaid files; include those if changed. `scripts/capability_atlas/model.py` reads models.yaml, launcher_manifest.json, feature_parity.json and capability_connections.json. Preserve registry identities from both branches, including shot-pattern registration and upstream native-task features. Do not splice generated HTML/graphs or select an entire side. |
| `docs/development/ORG01_CAPABILITY_BASELINE.md` | Regenerate via `python -m scripts.generate_capability_baseline` from semantically merged `src/config/capability_migration.json`; preserve aliases, exclusions and feature identities. |
| `docs/development/feature_parity_matrix.md` | Regenerate via `python -m scripts.generate_feature_parity_matrix` from merged `src/config/feature_parity.json`. Preserve status/rationale/tracking entries, not merely counts. |
| `docs/agent_context/{README.md,index.html}` | Render the merged catalog/contracts and original review evidence with the correctly pinned agent-context runtime. Do not hand-edit generated freshness badges. |
| `docs/shared_tools/divergence_inventory.{md,v1.json}` | Regenerate using `python -m scripts.shared_tools.divergence_inventory --write` against the resolved working tree AND resolved gitlink. Classification is computed; never union old generated rows/counts. Preserve substantive implementations/readmes first. |

Suggested order: resolve canonical registries and source/submodule choices; review stable IDs; regenerate baseline/parity; regenerate atlas; merge review records without renewal; render context; generate divergence; run each `--check` and focused contracts. Use the scripts at the merged revision, not an assumed older CLI implementation. Source generator checks are freshness checks, not runtime/scientific acceptance.

## Agent Context Review Records

Actual `reviews.json` has five study-side keys and six upstream keys. Only `launcher-atlas` differs among shared keys; upstream adds `project-task-native-replay`. Retain all identical records and the new upstream record byte-for-byte. Preserve both original launcher-atlas records as historical evidence with their original fingerprints/input hashes/rationales; choose the canonical current record only if its entire input fingerprint matches the merged boundary. If neither matches, retain an original record as stale and expose that state; do not manufacture a combined fingerprint or rerun `review` merely to turn the generated check green.

`docs/agent_context/USAGE.md` explicitly distinguishes `render`, `check`, `evaluate` from `review RELATION --rationale ...`. A new review requires actual inspection of both public endpoints, contract and relevant tests. Any stale boundary remains a named integration gate until that work is done. The upstream catalog adds native project task/replay entries; these must survive even if generated files conflict.

## Tools Pin Is a Material Boundary

Study gitlink: `3678409fc51024150ab28970b72e3b468935f345`.
Upstream gitlink: `86d0f28b1cc5acf61185e07e320c816c2d005512`.

Resolve this deliberately with upstream consumer requirements and study replay requirements recorded. Do not copy vendor content into shared Python or silently reset the pin to make old study receipts pass. Use agent-context from the resolved Tools pin, as its usage guide requires, through an isolated runtime or explicit isolated `PYTHONPATH` to that exact provider package. Do not install into or mutate the live shared validation venv, PRESENTATION environment, or frozen producer. No network/install action is authorized here; report an isolated dependency-setup requirement if the merged provider cannot run with existing dependencies. Regenerate divergence from the actual resolved trees. Historical simulation/report receipts remain bound to their original pin/runtime/source; integration success does not authorize replacing those hashes. Preserve the frozen producer checkout and explicit reproduction route. Any new-pin consumer claim needs real compatibility evidence, separately scoped from model acceptance.

## Calculation Registry

This is an editable governance inventory, **not a generated scientific approval artifact**. Both branches have `release_status: blocked-inventory-required`, `inventory_commit: null`, and empty `calculations`. Preserve these states.

The exact blocker-ID union is **55**: upstream54 plus study-only `UP-D1-shot-pattern-variance-inventory`. There are two differing shared IDs:

- `UP-D1-feedback-comparison-admission-inventory`: upstream refines state identity into `state-schema/initial-state-payload`; retain that more explicit requirement and all qualification limits.
- `UP-D1-grip-wrench-reduction-inventory`: upstream retains the earlier record and appends F10i/#12060 bushing-law inventory, actual77 checks/12 absent-SDK skips, and explicit no-full-body/measurement acceptance limits. Retain the added content without promoting its status.

No blocker deletion, duplicate ID, invented calculation row, or release promotion. Preserve source-specific receipt claims as historical, not newly rerun evidence. Read `scripts/config/design_manual_governance.json` and run `python -m scripts.check_design_manual_governance` on the resolution. QMD under `manuals/upstreamdrift` remains manual authority; do not edit generated LaTeX/PDF/DOCX/HTML to resolve this JSON conflict or regenerate unrelated manual outputs. The standalone shot-pattern article remains a separate research reference.

## Agent Handoff

The actual files are already oversized: study611 lines, upstream1010. Blind concatenation loses the current-state-only/under150-line contract. Preserve substantive active work, blocked gates, source pins, branch/PR separation, MATLAB R2025b requirement and study publication status; link to existing detailed turnover documents instead of copying all chronological sections. Do not silently drop upstream active MSK/Drake/turn/grip/native-task records or replace their source-specific evidence with this study's status.

Keep original full files in Git/history and, if summarization needs traceability, save an explicit old-heading→retained pointer mapping in integration evidence. Do not invent completion for entries whose current status is unknown. If a correct concise consolidation needs owner clarification, report the specific ambiguity rather than deleting it. Do not edit shared HANDOFF/DEVELOPMENT_LOG/SPEC change-log rows; use the required fragment mechanism for this integration.

## Completion Evidence

Record resolved canonical-source diffs and generator commands, exact merged Tools pin, output freshness checks, retained registry IDs (55 blocker union), context review statuses and any stale boundaries. Inspect auto-merged shared-infrastructure and registry additions. Require no conflict markers, valid JSON, title-policy checks, scoped contract tests and manual-governance check. Full branch pre-PR checks follow separately.

No generated map, hash refresh, successful merge, or gate exit can substitute for empirical/model qualification or the frozen study's final publication acceptance. No network or merge was performed for this review.

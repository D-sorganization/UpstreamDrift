# Child Publication Contract Checkpoint

Core immutable-parent/child provenance checks and report-consumer adaptation are implemented. This is not execution-ready: no real parent has been verified by this new command, no CLI is yet exposed, and no graphics have been rerendered.

The parent public loader is unchanged. The new frozen verifier executes it under pinned source/import origin, interpreter hash and exact matched source/runtime/native snapshot, with source checks before and after. The child contract preserves numerical bytes and checks every origin against the verified parent ledger, permits only the explicit density revision set, rejects extra files, and rechecks after parent verification. Coherently rehashed protected-source changes are rejected. Report consumers share one loader dispatch and preserve parent and child receipt identities.

Tests began with missing-module failures, then exposed omitted-source pin acceptance and missing final-receipt identity; each has a passing regression. The integration run initially referenced a nonexistent test filename (recorded), then exposed an exception-type compatibility regression (fixed) and an absent sparse Markdown fixture (restored exactly from HEAD). No fixture skip was introduced.

Next API: `render_selected_pair_figures(pair, output_dir, *, figure_kinds: frozenset[str]) -> list[dict]`. Its returned ledger rows must match the parent's ordered identities, protocols, counts, targets, conditioning and relative paths exactly; only asset hashes change. The adapter passes `child / "figures"` as output_dir. It must resolve parent-verified source refs/aliases and reconstruct the existing sorted six-population input mask without calling any statistical analyzer. The expanded caption scope may permit academic overhead and social overhead PNG only after that repair is independently qualified; this checkpoint deliberately retains density-only scope.

No MAIN or FINAL_LAYOUT files, producer processes, numerical artifacts or live graphics were modified. No commit or push was made.

## Independent Qualification Gate Added

Root correctly identified that rendering alone could publish the initial receipt. A new failing regression reproduced this; creation now writes only REPORT_SUPPLEMENT_PUBLICATION_CANDIDATE.json. The separate finalizer requires a hash-pinned independent accepted review, candidate identity, exact old/new producer delta, every revised asset hash and supporting evidence hashes. It copies those exact review bytes, revalidates inputs/outputs, and only then publishes the final revision receipt. Tests explicitly use synthetic review fixtures; no real visual acceptance was generated. Final checkpoint: 54 integration tests pass, Ruff and four-source Mypy pass, diff-check clean. The new contract module is 455 lines. Source remains uncommitted and no CLI or real rendering has run.

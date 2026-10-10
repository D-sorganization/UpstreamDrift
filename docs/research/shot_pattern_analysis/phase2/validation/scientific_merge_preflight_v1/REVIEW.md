# Scientific Source Merge Preflight

## Exact Projection

Read-only comparison: study revision `2953c52b37`; synthetic merge tree `ac0b22ea4d746025523fdfc6a6e53312654d738e`. Inventory is exactly the **54 paths** in `source_transition_supplement.json.new_execution.source_sha256`, read from the fixed study revision. `evidence.json` contains all accepted/study/synthetic SHA256 values. Metadata conflicts in this synthetic tree remain unresolved; this is not final merge acceptance.

**One new upstream difference:** `src/shared/python/physics/grip_contact_model.py`. All other 53 inventory paths have identical study-versus-synthetic bytes, including native Rust flight sources, ball simulation/launch/properties, rigid/incremental contact kernels, constants, geometry, matching, noise, scoring provider, and orchestration paths included in the inventory.

**Two already-present differences from the older accepted snapshot:** `academic_plots.py` and `matched_report_plots.py`. These are previously reviewed study reporting changes, not introduced by the upstream merge. Preserve their separate rendering qualification and the old immutable receipt; do not misclassify them as new physics changes.

## Actual Upstream Change and Reachability

The grip facade adds `compute_grip_pressure_centre` to its `_grip_forces` imports and `__all__`. A directly related dependency, `_grip_forces.py`, is outside the 54-file inventory: upstream names the normal-force-weighted grip centroid explicitly, returns `None` for unloaded contacts instead of the old origin, and retains `compute_center_of_pressure` as a warning-emitting deprecated alias. This is a meaningful grip-provider behavior change, not merely formatting.

The study's `matched_export.py` includes `grip_contact_model.py` because it conservatively snapshots `physics/*contact*.py`, not because the matched ball-impact solver calls the grip facade. Focused source searches in both study tool/script trees and ball_simulator/rigid_contact/incremental_contact/physics initializer show no imports or calls to the changed grip facade/centroid functions. The physics initializer itself is unchanged. Thus there is no identified direct active shot-study path to the modified grip centroid; this is not a universal call-graph proof or an assertion that all physics dependencies outside the inventory are unchanged.

The 54-file inventory is an exact historical source projection, **not a complete transitive dependency audit**. Tools namespace/provider preflight is separate. Native installed binary, dependency/runtime versions and backend selection were not executed/requalified here; a Rust source match does not prove binary identity.

## Focused Resolved-Tree Gates

After the integration worktree is stable, preserve the added upstream API and its unavailable-state semantics. Run upstream `tests/unit/physics/test_cop_consolidation.py` and `tests/unit/test_grip_contact_model.py` to check the new centroid and deprecated alias. Do not change expectations back to fabricated zero data merely to accommodate older callers.

For the actual study boundary, run existing `tests/tools/shot_pattern_analysis/test_matched_physics.py`, `tests/unit/physics/test_incremental_contact.py`, `tests/unit/physics/test_rigid_contact.py`, and source/provenance/report-consumer contract tests. Explicitly verify the active matched adapter resolves the same intended contact and flight providers. These are bounded existing regressions, not a request to rerun Monte Carlo/bootstrap or claim empirical validation.

Recompute this projection from the **final resolved Git tree**, then inspect any additional changes before accepting integration. Keep native/runtime verification separate. A changed grip facade makes a strict old snapshot unequal even if unreachable from the study: do not overwrite acceptance hashes or weaken loaders. Reproduce the historical publication through the frozen source/runtime contract; any separately accepted merged-source use needs its own scope and evidence.

No mutable integration tree, live producer, figures or simulation output was read; no tests, network, installation or source changes were performed.

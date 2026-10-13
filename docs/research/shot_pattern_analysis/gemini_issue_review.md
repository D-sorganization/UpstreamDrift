# UpstreamDrift Shot Pattern Analysis: Issue Record and Publication Review

## Executive Summary

This independent review evaluates the fourteen prepared shot-pattern analysis issue records ([SPA-001](../../../docs/research/shot_pattern_analysis/issues/SPA-001.json) through [SPA-014](../../../docs/research/shot_pattern_analysis/issues/SPA-014.json)), the analysis register ([README.md](../../../docs/research/shot_pattern_analysis/issues/README.md)), research methods documentation ([README.md](../../../docs/research/shot_pattern_analysis/README.md)), implemented physics/tool fixes, and test suites.

The review was performed by Gemini 3.8 Flash High within the isolated worktree `codex/shot-pattern-analysis` under active simulation conditions: four background matrix worker shards (PIDs 322627, 322629, 330846, 331185) were calculating at review time the 24-cell corrected experiment matrix (720,000 shots across driver, 7-iron, and pitching wedge).

All scientific sources are frozen. Every running worker verifies source file hashes before and after each cell; modifying any source file would immediately abort active runs with a `RuntimeError`. Furthermore, bot credentials are not yet configured on this host (the CLI authenticates as personal account `dieterolson`), which blocked publication at review time. The subsequent explicit user authorization below supersedes that task-specific restriction.

The fourteen issue records are thoroughly grounded in real defects and necessary qualifications. However, several critical omissions, formatting defects, duplicate scopes, and typing issues require correction once workers finish and bot authentication is established.

---

## Categorization of Issue Records (SPA-001 Through SPA-014)

The issue records span four distinct categories:

| Category | Issue Records | Description and Governance Boundary |
| --- | --- | --- |
| **Corrected Defects** | [SPA-001](issues/SPA-001.json), [SPA-002](issues/SPA-002.json), [SPA-009](issues/SPA-009.json), [SPA-010](issues/SPA-010.json), [SPA-011](issues/SPA-011.json) | Algorithmic and software defects in impact mechanics, baseline comparison contracts, and CLI/GUI input handling that have been corrected in code and verified by unit tests. |
| **Pending Publication & Replacement** | [SPA-003](issues/SPA-003.json), [SPA-006](issues/SPA-006.json), [SPA-012](issues/SPA-012.json), [SPA-013](issues/SPA-013.json), [SPA-014](issues/SPA-014.json) | Work requiring active simulation completion, matrix scoring post-processing, full-package typing fixes, or baseline table metadata updates before pull request publication. |
| **Modeling Assumptions & Sensitivities** | [SPA-004](issues/SPA-004.json), [SPA-005](issues/SPA-005.json), [SPA-007](issues/SPA-007.json) | Mathematical approximations and sensitivity dimensions (rigid shaft rotation, descriptive endpoint covariance, hybrid club presets) that explore mechanisms without asserting measured golfer distributions. |
| **Deferred Empirical Validation** | [SPA-008](issues/SPA-008.json) | Physical effects outside the modeled physics (off-center impact/gear effect, dynamic shaft bending, turf interaction, golfer execution covariance) whose validation against real-world populations is explicitly deferred. |

---

## Cross-Verification Against Implemented Fixes and Test Evidence

### SPA-001: Tangential Impact Translation Impulse
- **Classification:** Corrected Defect.
- **Problem Statement:** Friction previously applied angular spin impulses to the ball without an equal-and-opposite tangential linear impulse to ball and clubhead, violating linear momentum conservation and forcing launch trajectories to follow face normal unconditionally.
- **Implementation Status:** Corrected in [`models.py`](../../../src/shared/python/physics/impact_model/models.py#L108-L142) (commit `19257dabb2`). Tangential linear impulse is calculated and applied to ball and club velocities.
- **Test Evidence:** Verified in [`test_impact_tangential_impulse.py`](../../../tests/unit/physics/test_impact_tangential_impulse.py#L38-L59) and [`test_impact_central_contact_review.py`](../../../tests/unit/physics/test_impact_central_contact_review.py#L33-L82). 117 impact unit tests pass.

### SPA-002: Finite Club Mass in Sticking Contact
- **Classification:** Corrected Defect.
- **Problem Statement:** The no-slip friction impulse cap used the infinite-club-mass formula ($\frac{2}{7} m_{\text{ball}}$) instead of the effective contact mass $\frac{m_{\text{ball}} m_{\text{club}}}{m_{\text{ball}} + m_{\text{club}}}$, and omitted preexisting ball spin from contact surface slip velocity.
- **Implementation Status:** Corrected in [`models.py`](../../../src/shared/python/physics/impact_model/models.py#L90-L106) (commit `19257dabb2`).
- **Test Evidence:** Verified in [`test_impact_tangential_impulse.py`](../../../tests/unit/physics/test_impact_tangential_impulse.py#L38-L92) and [`test_impact_central_contact_review.py`](../../../tests/unit/physics/test_impact_central_contact_review.py#L104-L120).

### SPA-003: Replacement of Superseded Impact Results
- **Classification:** Pending Publication & Replacement.
- **Problem Statement:** The initial four 30,000-shot driver simulation bundles were generated using the uncorrected impact solver.
- **Implementation Status:** Historical bundles marked superseded in [`historical_control_report.md`](../../../docs/research/shot_pattern_analysis/historical_control_report.md). The replacement 24-cell matrix (720,000 shots across driver, 7-iron, and pitching wedge) is actively running under [`matrix.py`](../../../src/tools/shot_pattern_analysis/matrix.py).
- **Test Evidence:** Shard execution, run-start provenance, and manifest contracts verified in [`test_matrix.py`](../../../tests/tools/shot_pattern_analysis/test_matrix.py).

### SPA-004: Geometric Face–Loft and Shaft-Lean Coupling
- **Classification:** Modeling Assumption & Sensitivity.
- **Problem Statement:** Assuming constant delivered loft regardless of face closure masked the long-left / short-right dispersion pattern observed in mid-to-high lofted clubs.
- **Implementation Status:** Rigid shaft-axis rotation geometry implemented in [`delivery_geometry.py`](../../../src/tools/shot_pattern_analysis/delivery_geometry.py) and integrated into [`core.py`](../../../src/tools/shot_pattern_analysis/core.py).
- **Test Evidence:** 17 geometry tests in [`test_delivery_geometry.py`](../../../tests/tools/shot_pattern_analysis/test_delivery_geometry.py), plus [`test_physics.py`](../../../tests/tools/shot_pattern_analysis/test_physics.py) and [`test_sensitivity.py`](../../../tests/tools/shot_pattern_analysis/test_sensitivity.py). Deterministic sensitivity verified in [`corrected_sensitivity.json`](../../../docs/research/shot_pattern_analysis/corrected_sensitivity.json).

### SPA-005: Long/Short Covariance and Dispersion Statistics
- **Classification:** Modeling Assumption & Sensitivity.
- **Problem Statement:** Scalar lateral standard deviation omitted carry covariance, signed correlation, and directional quadrant fractions.
- **Implementation Status:** Descriptive endpoint statistics implemented in [`dispersion_stats.py`](../../../src/tools/shot_pattern_analysis/dispersion_stats.py) and integrated into [`reporting.py`](../../../src/tools/shot_pattern_analysis/reporting.py).
- **Test Evidence:** 8 tests in [`test_dispersion_stats.py`](../../../tests/tools/shot_pattern_analysis/test_dispersion_stats.py).

### SPA-006: Club-Appropriate Tee and Approach Scoring Contexts
- **Classification:** Pending Publication & Replacement.
- **Problem Statement:** Scoring driver landing positions as fairway approaches to a 15 m green circle misrepresented tee shot utility. Driver shots require a fairway corridor benchmark, while irons/PW require green-approach benchmarks.
- **Implementation Status:** Implemented in [`scenario_scoring.py`](../../../src/tools/shot_pattern_analysis/scenario_scoring.py) and [`scoring_cache.py`](../../../src/tools/shot_pattern_analysis/scoring_cache.py) (commit `60816b4470`).
- **Test Evidence:** Verified in [`test_scenario_scoring.py`](../../../tests/tools/shot_pattern_analysis/test_scenario_scoring.py) and [`test_scoring_cache_independent.py`](../../../tests/tools/shot_pattern_analysis/test_scoring_cache_independent.py). Post-processing of active 24-cell matrix bundles remains pending simulation completion.

### SPA-007: Provenance of Illustrative Preset Inputs
- **Classification:** Modeling Assumption & Sensitivity.
- **Problem Statement:** Default delivery parameters risked being mistaken for empirical player distributions or complete club specifications.
- **Implementation Status:** Explicitly labeled as illustrative assumptions across [`README.md`](../../../docs/research/shot_pattern_analysis/README.md#L73-L85), [`presets.py`](../../../src/tools/shot_pattern_analysis/presets.py), and [`gui.py`](../../../src/tools/shot_pattern_analysis/gui.py#L119).
- **Test Evidence:** Verified in [`test_presets.py`](../../../tests/tools/shot_pattern_analysis/test_presets.py).

### SPA-008: Physical Scope and Model Qualification Boundaries
- **Classification:** Deferred Empirical Validation.
- **Problem Statement:** Numerical convergence and vector conservation do not validate off-center strikes, gear effect, dynamic shaft droop/lead, or real player variability.
- **Implementation Status:** Disclosed in [`README.md`](../../../docs/research/shot_pattern_analysis/README.md#L140-L146) and [`shot_pattern_analysis.tex`](../../../docs/research/shot_pattern_analysis/shot_pattern_analysis.tex).
- **Test Evidence:** Governed by explicit absence of empirical validation claims.

### SPA-009: Rejection of Separating Normal Contact
- **Classification:** Corrected Defect.
- **Problem Statement:** Negative normal approach velocity produced an unphysical attractive impulse pulling separated bodies together.
- **Implementation Status:** Input contract added to [`models.py`](../../../src/shared/python/physics/impact_model/models.py) raising `ValueError` on non-positive normal approach.
- **Test Evidence:** Verified in [`test_impact_tangential_impulse.py`](../../../tests/unit/physics/test_impact_tangential_impulse.py#L94-L98).

### SPA-010: Validation of Experiment Delivery Baselines
- **Classification:** Corrected Defect.
- **Problem Statement:** Experiment comparison contracts verified sample size and seed but failed to reject disparate club baselines (e.g., comparing driver vs. 7-iron at an identical target).
- **Implementation Status:** Delivery contract added to [`comparison.py`](../../../src/tools/shot_pattern_analysis/comparison.py#L27-L52).
- **Test Evidence:** Verified in [`test_comparison.py`](../../../tests/tools/shot_pattern_analysis/test_comparison.py#L47-L58).

### SPA-011: Custom Delivery Inputs in GUI Subprocess
- **Classification:** Corrected Defect.
- **Problem Statement:** Modifying delivery spin boxes in the GUI relabeled the preset selection to "custom", but spawned subprocesses rejected `--club-preset custom`.
- **Implementation Status:** Custom preset support added to [`__main__.py`](../../../src/tools/shot_pattern_analysis/__main__.py#L45-L80) and wired into [`gui.py`](../../../src/tools/shot_pattern_analysis/gui.py).
- **Test Evidence:** Verified in [`test_entrypoint.py`](../../../tests/tools/shot_pattern_analysis/test_entrypoint.py#L68-L89).

### SPA-012: Static Typing at Package Boundaries
- **Classification:** Pending Publication & Replacement.
- **Problem Statement:** Mypy detected typing issues across optional Qt headers, resize signatures, and physics calls.
- **Implementation Status:** GUI and physics call boundaries were typed in staged changes. However, running `mypy --python-version 3.12` on `src/tools/shot_pattern_analysis` reveals an unaddressed error in [`numerics.py:74`](../../../src/tools/shot_pattern_analysis/numerics.py#L74) (`float | str` comparison).
- **Test Evidence:** GUI typing passes; package-wide mypy fails until `numerics.py` is corrected.

### SPA-013: Run-Start Source Provenance and Drift Detection
- **Classification:** Pending Publication & Replacement.
- **Problem Statement:** Hashing sources only at export time risked recording post-run modifications rather than code loaded during execution.
- **Implementation Status:** [`matrix.py`](../../../src/tools/shot_pattern_analysis/matrix.py#L130-L155) captures `run_start.json` before execution and asserts source identity upon completion. However, standalone exports in [`reporting.py:496-498`](../../../src/tools/shot_pattern_analysis/reporting.py#L496-L498) still compute hashes only at export time.
- **Test Evidence:** Verified in [`test_matrix.py`](../../../tests/tools/shot_pattern_analysis/test_matrix.py).

### SPA-014: Explicit Versioning of Extended Scoring Table
- **Classification:** Pending Publication & Replacement.
- **Problem Statement:** Extending tee/fairway/rough tables to 600 yards retained the abbreviated baseline version string `table9-plus-anchor-reconciled-putting/1` in [`scoring.py:227`](../../../src/tools/shot_pattern_analysis/scoring.py#L227).
- **Implementation Status:** Content SHA-256 uniquely identifies the extended table; bumping the version string is deliberately scheduled for after active simulations finish.
- **Test Evidence:** Verified in [`test_scoring.py`](../../../tests/tools/shot_pattern_analysis/test_scoring.py#L30-L41) and [`test_scoring_cache_independent.py`](../../../tests/tools/shot_pattern_analysis/test_scoring_cache_independent.py).

---

## Identified Duplicates and Scope Overlaps

1. **SPA-001 vs. SPA-002 (Impact Mechanics):**
   Both records address defects in `RigidBodyImpactModel.solve` in [`models.py`](../../../src/shared/python/physics/impact_model/models.py). SPA-001 addresses missing tangential linear impulse translation; SPA-002 addresses the sticking Coulomb cap and finite clubhead mass. While physically distinct, they share the exact same validation test files ([`test_impact_tangential_impulse.py`](../../../tests/unit/physics/test_impact_tangential_impulse.py) and [`test_impact_central_contact_review.py`](../../../tests/unit/physics/test_impact_central_contact_review.py)) and landed in the same commit (`19257dabb2`). They should be linked to the same pull request rather than published as disjoint work streams.
2. **SPA-006 vs. SPA-014 (Scoring Benchmark Table):**
   SPA-006 introduces driver tee scoring and extended 600 yd baseline tables. SPA-014 tracks the fact that the version metadata string in [`scoring.py:227`](../../../src/tools/shot_pattern_analysis/scoring.py#L227) was not bumped when the tables were extended. SPA-014 is an omitted metadata cleanup resulting directly from SPA-006.
3. **SPA-007 vs. SPA-008 (Scope and Governance Boundaries):**
   Both records address documentation boundaries. SPA-007 bounds input parameters (hybrid presets vs. measured golfer data); SPA-008 bounds physical model applicability (central contact vs. off-center strikes/player dynamics). Both are resolved through documentation caveats without requiring algorithm changes.
4. **Issue Register Count Discrepancy:**
   [`AGENT_HANDOFF.md:7`](../../../AGENT_HANDOFF.md#L7) states: *"thirteen concrete local publication records in docs/research/shot_pattern_analysis/issues/"*. However, there are fourteen records ([`SPA-001.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-001.json) through [`SPA-014.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-014.json)). The handoff file was not updated when SPA-014 was added.

---

## Identified Omissions and Structural Inconsistencies

1. **Markdown Table Formatting Break in Register:**
   In [`issues/README.md:23-24`](../../../docs/research/shot_pattern_analysis/issues/README.md#L23-L24), an extraneous blank line separates SPA-013 and SPA-014. In GitHub Flavored Markdown, a blank line terminates the table, rendering SPA-014 as broken text or a detached single-row table.
2. **Omission of SPA-009 Through SPA-014 in Narrative Resolution Requirements:**
   In [`issues/README.md:26-57`](../../../docs/research/shot_pattern_analysis/issues/README.md#L26-L57), the section `## Regression and Resolution Requirements` details requirements for SPA-001 through SPA-008, but completely omits SPA-009, SPA-010, SPA-011, SPA-012, SPA-013, and SPA-014.
3. **Missing `## Validation` Sections in JSON Records:**
   While [`SPA-001.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-001.json) through [`SPA-008.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-008.json) and [`SPA-010.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-010.json) include a `## Validation` section in `body`, five issue records ([SPA-009](../../../docs/research/shot_pattern_analysis/issues/SPA-009.json), [SPA-011](../../../docs/research/shot_pattern_analysis/issues/SPA-011.json), [SPA-012](../../../docs/research/shot_pattern_analysis/issues/SPA-012.json), [SPA-013](../../../docs/research/shot_pattern_analysis/issues/SPA-013.json), and [SPA-014](../../../docs/research/shot_pattern_analysis/issues/SPA-014.json)) omit `## Validation` entirely.
4. **Handoff Section Inconsistencies across JSON Records:**
   - In SPA-001 to SPA-008 and SPA-010, the handoff reads: `Branch: \`codex/shot-pattern-analysis\`. Implementation and evidence are tracked in \`docs/research/shot_pattern_analysis/issues/README.md\`. Keep open until implementing PR is merged or an explicitly permitted disposition applies.`
   - In SPA-009, the tracking pointer and permitted disposition clause are dropped.
   - In SPA-011, SPA-012, SPA-013, and SPA-014, the branch name lacks backticks (`Branch: codex/shot-pattern-analysis`), and both the tracking pointer and permitted disposition clause are missing.
5. **Mypy Type Error Omission in SPA-012 Scope:**
   [`SPA-012.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-012.json) scopes typing issues to GUI headers, resize overrides, and physics keyword arguments. It omits the type error in [`numerics.py:74`](../../../src/tools/shot_pattern_analysis/numerics.py#L74), where `row["dt_020_vs_010_m"]` is typed as `float | str`, causing:
   `error: Argument 1 to "isfinite" has incompatible type "float | str"; expected "SupportsFloat | SupportsIndex"` and `error: Unsupported operand types for <= ("str" and "float")`.
   Because SPA-012's acceptance criteria requires passing package mypy, SPA-012 cannot be satisfied without correcting `numerics.py:74`.
6. **GUI Custom Mode Test Coverage Gap in SPA-011:**
   [`SPA-011.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-011.json) requires verifying a "headless GUI custom shaft-rotation run". [`test_entrypoint.py:68-89`](../../../tests/tools/shot_pattern_analysis/test_entrypoint.py#L68-L89) tests CLI `--club-preset custom`, but [`test_embed_adapter.py:204-249`](../../../tests/ui/tools/shot_pattern_analysis/test_embed_adapter.py#L204-L249) only tests `seven_iron` modification (which preserves `club_id="seven_iron"`). A headless test verifying widget execution with `club_id="custom"` is missing.
7. **Standalone Run Provenance Snapshot Gap in SPA-013:**
   [`matrix.py`](../../../src/tools/shot_pattern_analysis/matrix.py) implements run-start snapshotting (`run_start.json`) and drift detection. However, standalone execution via [`reporting.py:496-498`](../../../src/tools/shot_pattern_analysis/reporting.py#L496-L498) still hashes files only during export. The problem stated in SPA-013 remains unaddressed for non-matrix runs.

---

## Audit of Issue Closure Claims and Policy Compliance

1. **Non-Negotiable Closure Rule Compliance:**
   Repo policy mandates: *"NEVER close a feature or bug issue without one of: 1. A merged PR that demonstrably implements the acceptance criteria (use Closes #N in the PR description), OR 2. An explicit wontfix, roadmap, duplicate, or invalid label."*
   All fourteen JSON records maintain `"publication_status": "pending_agent_bot_authentication"`, and [`issues/README.md:6`](../../../docs/research/shot_pattern_analysis/issues/README.md#L6) explicitly states: *"No issue is claimed to have been filed or closed."* This strictly adheres to policy.
2. **Ambiguous Implementation State Descriptions:**
   - In [`issues/README.md:15`](../../../docs/research/shot_pattern_analysis/issues/README.md#L15), SPA-006 is labeled *"Replacement Tee Scoring Pending"*. The scoring logic ([`scenario_scoring.py`](../../../src/tools/shot_pattern_analysis/scenario_scoring.py)) is already implemented and covered by unit tests; what is pending is scoring the active 24-cell matrix runs. This should be phrased *"Scoring Implemented; Matrix Bundle Scoring Pending"*, mirroring SPA-004.
   - In [`issues/README.md:22`](../../../docs/research/shot_pattern_analysis/issues/README.md#L22), SPA-013 is labeled *"Run-Start Snapshot Correction Pending"*, yet `matrix.py` has already implemented run-start snapshotting and drift checks. It should be clarified as *"Matrix Snapshot Implemented; Standalone Export Integration Pending"*.

---

## Publication Safety and Pull Request Sequencing Audit

1. **Authentication and Identity Gate (GOV-1):**
   The host `gh` CLI authenticates as `dieterolson` (the repository owner). Creating issues or pull requests under this credential would violate GOV-1 (agent actions must be performed under dedicated bot credentials). Publication must remain blocked until bot authentication is established.
2. **Scientific Code Freeze Gate:**
   Four background worker processes are executing matrix shards. Each shard executes `source_snapshot()` and asserts hash equality post-run. Modifying any scientific Python or Rust file before all workers complete will crash active processes and invalidate runs.
3. **Missing Pre-PR Tool:**
   [`README.md:206-207`](../../../docs/research/shot_pattern_analysis/README.md#L206-L207) records that `python3 -m scripts.pre_pr` does not exist in this checkout (`No module named scripts.pre_pr`). Workflows relying on this module will fail.
4. **Pull Request Quality and Policy Compliance:**
   - Repository rules prohibit draft pull requests (*"Full PRs, never drafts"*).
   - The current branch contains uncommitted staged changes (`gui.py`, `models.yaml`, etc.), active run outputs in `corrected_results/`, and pending documentation updates. Opening a PR in this intermediate state would violate policy.

---

## Actionable Findings and Proposed Corrections

The following actionable corrections are ordered by dependency and priority. **No source files should be edited until running simulation workers terminate.**

```
       Simulation Phase                    Post-Simulation Phase                       Publication Phase
[Four Matrix Shards Running] ---> [Correct numerics.py & scoring.py] ---> [Commit Clean Artifacts]
   (Scientific Code FROZEN)       [Update issue JSONs & README.md   ]      [Configure Bot Identity ]
                                  [Run Matrix Scoring Postprocess   ]      [Publish PR (Non-Draft) ]
```

### Finding 1: Table Formatting Defect in Issues README
- **Location:** [`docs/research/shot_pattern_analysis/issues/README.md#L23`](../../../docs/research/shot_pattern_analysis/issues/README.md#L23)
- **Problem:** Blank line before SPA-014 breaks Markdown table parsing.
- **Correction:** Remove blank line 23 so line 24 joins the table directly after line 22.

### Finding 2: Incomplete Narrative in Issues README
- **Location:** [`docs/research/shot_pattern_analysis/issues/README.md#L26-L57`](../../../docs/research/shot_pattern_analysis/issues/README.md#L26-L57)
- **Problem:** `## Regression and Resolution Requirements` omits SPA-009 through SPA-014.
- **Correction:** Append explicit requirement paragraphs for SPA-009 (separating contact), SPA-010 (baseline comparison contracts), SPA-011 (GUI custom inputs), SPA-012 (package static typing), SPA-013 (run-start provenance freezing), and SPA-014 (extended table versioning).

### Finding 3: Structural Inconsistencies Across Issue JSON Records
- **Location:** [`SPA-009.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-009.json), [`SPA-011.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-011.json), [`SPA-012.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-012.json), [`SPA-013.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-013.json), [`SPA-014.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-014.json)
- **Problem:** Missing `## Validation` section in `body`; inconsistent backticks on branch name; missing tracking pointer to `issues/README.md`; missing permitted disposition clause.
- **Correction:**
  - Add `## Validation` section to each JSON body referencing its test file.
  - Standardize `## Handoff` across all records to:
    `Branch: \`codex/shot-pattern-analysis\`. Implementation and evidence are tracked in \`docs/research/shot_pattern_analysis/issues/README.md\`. Keep open until implementing PR is merged or an explicitly permitted disposition applies.`

### Finding 4: Incomplete Scope in SPA-012 (Mypy Typings)
- **Location:** [`docs/research/shot_pattern_analysis/issues/SPA-012.json`](../../../docs/research/shot_pattern_analysis/issues/SPA-012.json) and [`src/tools/shot_pattern_analysis/numerics.py#L74`](../../../src/tools/shot_pattern_analysis/numerics.py#L74)
- **Problem:** SPA-012 omits `numerics.py:74` where `coarse` and `fine` are inferred as `float | str`.
- **Correction:**
  - Expand SPA-012 problem statement to include `numerics.py` refinement metric typing.
  - After simulations complete, cast differences in `numerics.py:69-70`:
    ```python
    coarse = max(float(row["dt_020_vs_010_m"]) for row in cases)
    fine = max(float(row["dt_010_vs_005_m"]) for row in cases)
    ```

### Finding 5: GUI Custom Mode Test Gap in SPA-011
- **Location:** [`tests/ui/tools/shot_pattern_analysis/test_embed_adapter.py`](../../../tests/ui/tools/shot_pattern_analysis/test_embed_adapter.py)
- **Problem:** Acceptance criteria requires verifying a headless GUI custom shaft-rotation run, but tests only cover modified `seven_iron`.
- **Correction:** Add a dedicated headless test in `test_embed_adapter.py` that selects the `custom` preset, enters custom parameters, and verifies export execution.

### Finding 6: Standalone Run Provenance Gap in SPA-013
- **Location:** [`src/tools/shot_pattern_analysis/reporting.py#L496-L498`](../../../src/tools/shot_pattern_analysis/reporting.py#L496-L498)
- **Problem:** Standalone runs compute source hashes only at export time.
- **Correction:** After simulations complete, update `run_analysis` in `core.py` to capture run-start hashes, pass them into `AnalysisResult`, and record them in `_export_receipt`.

### Finding 7: Record Count in Agent Handoff
- **Location:** [`AGENT_HANDOFF.md#L7`](../../../AGENT_HANDOFF.md#L7)
- **Problem:** Reports 13 records instead of 14.
- **Correction:** Update handoff text to `"fourteen concrete local publication records"`.

### Finding 8: Post-Simulation Execution Order
1. Allow the four background worker shards to finish all 24 cells.
2. Edit [`scoring.py:227`](../../../src/tools/shot_pattern_analysis/scoring.py#L227) to bump the baseline version to `table9-600yd-plus-anchor-reconciled-putting/1` (SPA-014).
3. Fix type casting in [`numerics.py:69-70`](../../../src/tools/shot_pattern_analysis/numerics.py#L69-L70) (SPA-012).
4. Run scoring post-processing on all 24 completed matrix cells using [`score_saved_bundle`](../../../src/tools/shot_pattern_analysis/scoring.py#L358) (SPA-006).
5. Update [`issues/README.md`](../../../docs/research/shot_pattern_analysis/issues/README.md) and issue JSONs (SPA-009, SPA-011 to SPA-014).
6. Run full test suite and mypy verification.
7. Configure bot identity and open the complete, non-draft pull request.

## Parent Response and Subsequent State

This is a review-time record, not the final execution handoff. The user then
expressly authorized the personal GitHub identity for this task: “use my personal
identity - I authorize this.” Gemini through `agy` is filing the prepared issues;
no credentials or repository settings are changed.

The canonical Repository_Management pre-PR runner has been located. Its official
wrapper runs gates against the current working directory and will be used before
publication; direct tests alone are not being substituted for the required gate.

Two stable workers continue; interrupted incomplete shards will resume safely.
The issue register now also includes SPA-015 for overgeneralized mirror symmetry.
The parent accepted the standalone provenance and numerical typing findings for
correction after the frozen runs. Record-format corrections will be synchronized
with the published issue bodies. No issue is closed before merged PR evidence.

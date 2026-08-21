# Adversarial Integration & Organization Review — 2026-08-21

Scope requested by the repository owner: assess whether UpstreamDrift forms a
coherent, fully integrated whole — launcher/tool integration completeness,
data provenance in the GUIs (which engine/swing/trial produced what), linkage
from GUI tabs to supporting calculation documentation, and overall repository
organization. Every confirmed finding is filed as a GitHub issue; this
document is the running index of that review.

Review branch: `claude/upstreamdrift-adversarial-review-x29y1c`.

## Method

Parallel deep-dive audits over five axes, each verified against the code
before filing:

1. Launcher tile ↔ tool integration completeness (registry vs. reality).
2. GUI data provenance — engine / swing / trial labeling on displayed and
   exported data.
3. Calculation documentation ("calc sheets") and their linkage from GUIs.
4. Repository organization coherence (duplicated/orphaned trees, entry
   points, docs sprawl, config sprawl).
5. Motion pipeline and simulation workflow end-to-end integration.

Pre-existing open issues checked for overlap before filing: #8763, #8776,
#8689, #8695 (duplication/shadow-package debt), #8766, #8735 (test debt).

## Findings and Filed Issues

_(Populated as the review proceeds; each row is filed as a GitHub issue.)_

| # | Issue | Area | Severity | Title |
|---|-------|------|----------|-------|
| 1 | #8816 | Provenance | Critical | Cross-engine dashboard charts one aggregate robustness score replicated per engine |
| 2 | #8817 | Provenance | Critical | Cross-engine desktop GUI silently substitutes a 2-DOF stub, charts stay labeled as real engines |
| 3 | #8818 | Provenance | Critical | Ball Flight GUI wind/altitude/aero controls never read by the simulation |
| 4 | #8819 | Provenance | Critical | Swing→Flight Pipeline engine selector is cosmetic; engine_name never used |
| 5 | #8820 | Provenance | High | Unified dashboard exports carry no engine/model/run/timestamp identity |
| 6 | #8821 | Provenance | High | JSON/CSV exports get no provenance sidecar (binary formats do) |
| 7 | #8822 | Provenance | High | ProvenanceInfo has no engine field; callers omit model_path |
| 8 | #8823 | Provenance | High | ProvenanceValue UI provenance system built+tested but unused |
| 9 | #8824 | Provenance | High | Workspace project/session lineage subsystem has zero consumers |
| 10 | #8825 | Provenance | High | Launch Monitor plots stay stale across project changes |
| 11 | #8826 | Provenance | Medium | Launch Monitor data export not linked to its reproducibility manifest |
| 12 | #8827 | Provenance | Medium | Pose Studio engine status pill stale after silent mock downgrade |
| 13 | #8828 | Provenance | Medium | Shared plot titles/exports carry no identity; include_metadata ignored |
| 14 | #8829 | Provenance | Medium | Engine dashboards swallow model-load failures; engine name invisible embedded |
| 15 | #8830 | Organization | High | Third-party model repos vendored twice (submodules + ~467 MB committed copies) |
| 16 | #8831 | Organization | High | Six launcher entry points; docs/README.md points to the deprecated one as primary |
| 17 | #8832 | Organization | Medium | feature_parity.json attributes tile grid to deprecated unified_launcher.py |
| 18 | #8833 | Organization | High | Orphaned docs/ui/FEATURE_PARITY_MATRIX.md contradicts generated matrix and README |
| 19 | #8834 | Organization | Medium | scripts/validate_suite.py validates a pre-migration layout, can never pass |
| 20 | #8835 | Organization | Medium | Environment drift: four Python versions, two base-image digests |
| 21 | #8836 | Organization | Medium | Generated artifacts committed in output/ and reports/ with no ignore rules |
| 22 | #8837 | Organization | Medium | .scratch/ holds 98 committed agent issue drafts with no policy |
| 23 | #8839 | Organization | Medium | docs/README.md describes a nonexistent tree; exempt from catalog checker |
| 24 | #8840 | Organization | Low | Four self-admitted pending docs consolidations left unexecuted |
| 25 | #8841 | Organization | Medium | Top-level motion_matching trees are wrong-cwd artifacts; workflows disagree on leaderboard |
| 26 | #8842 | Organization | Low | notebooks/bunkershot3d/phase1_mvp.py cannot run (imports/paths wrong) |
| 27 | #8843 | Calc docs | High | In-app Help broken at root: USER_MANUAL.md path missing; 10/35 components resolve no help |
| 28 | #8844 | Calc docs | High | docs/help/analysis_tools.md documents a fabricated API |
| 29 | #8845 | Calc docs | High | Ball-flight calc sheet drifted: wrong lift law, dead fields, 5x cd1, stale assumptions |
| 30 | #8846 | Calc docs | Medium | 21 of 25 GUI tools have no help affordance; build_help_menu is dead code |
| 31 | #8847 | Calc docs | Medium | MethodCitation DOI metadata produced but surfaced to no user |
| 32 | #8848 | Calc docs | Medium | LM user guide documents API-only Strokes Gained as workbench feature; ADR 0036 missing |
| 33 | #8849 | Calc docs | Medium | sg_optimizer coefficients uncited despite data_sources.md traceability claim |
| 34 | #8850 | Calc docs | Medium | Key calc docs orphaned; docs/index.md has no hyperlinks |
| 35 | #8851 | Calc docs | Low | 21 broken doc links; check_markdown_links.py always exits 0 |
| 36 | #8852 | Launcher | High | Stale TOOLS_GITLINK_SHA fail-closes all six provider:tools tiles |
| 37 | #8853 | Launcher | High | PyQt and web launchers read two different tile registries (47 divergent IDs) |
| 38 | #8854 | Launcher | High | 12+ registered tiles point at nonexistent directories; no path-resolution test |
| 39 | #8855 | Launcher | High | Tile status chip derived purely from YAML — broken tiles always render Ready |
| 40 | #8856 | Launcher | Medium | Simscape embed adapter path unimportable (3D_Golf_Model identifier); silent failure |
| 41 | #8857 | Launcher | Medium | Two incompatible embedding contracts; five engine EmbeddableTool adapters unreachable |
| 42 | #8858 | Launcher | Medium | external_tools_adapter ignores vendor/ud-tools, breaks on correct clones |
| 43 | #8859 | Launcher | Medium | src/launchers/model_registry.py dead code shadowing real registry, can't parse models.yaml |
| 44 | #8860 | Launcher | Medium | Dotted module strings in path field; simulation_backends tile unlaunchable |
| 45 | #8861 | Launcher | Medium | feature_parity.json marks six unreachable shell tiles as parity |
| 46 | #8863 | Launcher | Low | Registry hygiene: unregistered tools, __all__ overwrite, hidden-tile leak |

(#8817 was extended with the finding that the REST API also silently
substitutes stub engines for known engine names.)

| # | Issue | Area | Severity | Title |
|---|-------|------|----------|-------|
| 47 | #8864 | Pipeline | High | Motion pipeline is a second, unmounted FastAPI app — curl-only |
| 48 | #8865 | Pipeline | High | Two independent C3D ingestion stacks; GUI upload bypasses the pipeline |
| 49 | #8866 | Pipeline | High | starting_pose_matcher ships ~1,600 lines of unwired per-engine extractors |
| 50 | #8867 | Pipeline | High | Two parallel canonical-pose representations maintained independently |
| 51 | #8868 | Pipeline | High | Realtime IPC: one gated publisher, one unlaunchable subscriber, dead training pair |
| 52 | #8869 | Pipeline | Medium | Three pub/sub implementations; ws transport a silent no-op |
| 53 | #8870 | Pipeline | Medium | JaxSim half-integrated: no tier, no adapter, dashboard runs nothing |
| 54 | #8871 | Pipeline | Medium | Simulation results in-memory only; export_paths always empty |
| 55 | #8873 | Pipeline | Medium | output/README.md documents a results API/CLI/config that don't exist |
| 56 | #8874 | Pipeline | Medium | 7 of 11 Rust crates uninstallable via extras; one crate has no caller |
| 57 | #8875 | Pipeline | Low | Pipeline API advertises source formats it rejects |
| 58 | #8876 | Pipeline | Low | Simscape service returns {} in live mode, blanking Pose Studio view |

## Synthesis — Does It Form a Coherent Whole?

Not yet. The review's cross-cutting conclusion is that UpstreamDrift's
pieces are individually substantial but the connective tissue is missing
or dishonest in five recurring ways:

1. **Built-but-unwired infrastructure.** The exact subsystems that would
   answer the owner's questions exist, tested, with zero production
   consumers: UI provenance labels (#8823), the project/session lineage
   spine (#8824), method citations with DOIs (#8847), the shared help-menu
   builder (#8846), per-engine skeleton extractors (#8866), the ws
   transport (#8869), and several Rust accelerators (#8874).
2. **Dishonest surfaces.** GUIs display claims the computation does not
   back: identical robustness bars labeled per-engine (#8816), stub
   engines labeled mujoco/drake/pinocchio (#8817), decorative wind and
   engine-source controls (#8818, #8819), always-"Ready" tiles (#8855),
   stale "live" status pills (#8827, #8876).
3. **Registry/document drift.** Two tile registries with 47 divergent IDs
   (#8853), two feature-parity matrices in contradiction (#8833), calc
   sheets contradicting shipped constants (#8845), help docs describing a
   fabricated API (#8844), a stale vendored-tools SHA fail-closing six
   tiles (#8852).
4. **No end-to-end data spine.** The flagship C3D→tracked-motion pipeline
   is unmounted and GUI-invisible (#8864, #8865); simulation results never
   reach disk through a contract downstream tools can consume (#8871,
   #8873); exports lose identity the moment they leave the app
   (#8820–#8822, #8826, #8828).
5. **Organizational accretion.** Duplicate vendored trees (#8830),
   deprecated entry points still documented as primary (#8831), committed
   scratch/output debris (#8836, #8837), and unexecuted doc consolidations
   (#8839, #8840).

The encouraging inverse: nearly every fix is wiring, not invention. The
provenance dataclasses, the lineage store, the availability computation,
and the citation records all exist — the work is connecting them to the
surfaces users actually see, then adding CI gates so the wiring cannot
silently rot again (path-resolution tests #8854, doc/code constant
parity #8845, link checking #8851, gitlink parity #8852).

## Status

Review complete: 46 issues filed (#8816–#8876) across five axes on
2026-08-21. Suggested sequencing: the four critical provenance issues
(#8816–#8819) and the two registry-honesty issues (#8852, #8855) first —
they are where users are actively being misled today.

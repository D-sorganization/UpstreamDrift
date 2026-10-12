# Agent Handoff: Current-State Index

## Active Study and Current Status

- **User-Requested Resume**: Goal pursuit is active after the usage reset. Start with [Resume Handoff](docs/research/shot_pattern_analysis/phase2/RESUME_HANDOFF.md).
- **Matched Delivery Study — Phase Two**: All 10.44 million attempted simulation rows are complete. Do not rerun simulations. Read [Scientific Report](docs/research/shot_pattern_analysis/phase2/REPORT.md) and [Design](docs/research/shot_pattern_analysis/phase2/DESIGN.md).
- **Scientific Acceptance**: Astra accepted the report after 406 numeric/count checks and 52 source-hash checks. These are geometric, model-conditional results; no measured player benefit or universal best shot shape is established.
- **Revised Report PDF**: Local Desktop delivery is39pages with conventional lateral-x/downrange-y axes,100 verified replay paths per pattern, expanded impact/flight/scoring methods and graphical summary. SHA256 `fd5506da87f5e619a550fb8cc871a0e8195b6fb98965a99809e3238a0f01508b`; review/delivery receipt `phase2/validation/refinement_v3/delivery_receipt.json`. Frozen33page public edition remains unchanged; revised AffineDrift article is locally rendered but not published (#5004).
- **Graphics**: All assigned baseline PNG reviews are complete. The 49 density-layout replacements (#12243) have root/Sol/Astra PNG acceptance and native PDF/SVG review. Coverage reconciliation accepts 2,304 pair PNGs plus 104 unchanged cell PNGs. All three archive bundles passed fresh local restoration.
- **Native Formats**: All 2,024 PDF and 2,024 SVG baseline renders completed. The 1,122 PDF bounding-box warnings were checked against painted pixels: 10,842 probes, minimum 6.4952 pt visible-ink clearance, no clipped edge ink. The renderer's final summary failed serialization; a separate successful finalizer preserved that distinction.
- **Integration and Publication**: The research prerelease is [published](https://github.com/D-sorganization/UpstreamDrift/releases/tag/shot-pattern-phase2-research-2026-10-10), bound to delivery checkpoint `cb5eff6dcd9df7d878f10578078b6fea69c8b681`. Its original payload is 21 inventory assets; two separate verification receipts were subsequently uploaded (23 release files total). Fresh download verification passed for the 21 payload assets, all three bundle member inventories, the 15-member illustrated source archive with 13 figures, and 98/98 report links. Receipt SHA256 `94f1fdea97207cb0eb8bf59db981d09cffd60d74d02f3ddd8957a3990e728843`: `docs/research/shot_pattern_analysis/phase2/validation/artifact_delivery_v3/remote-restoration-v3.json.gz`. The first restore attempt failed on archive-name mapping after asset hashes passed; Sol reviewed the corrected size/SHA bijection and the clean rerun exited 0 without asset changes. Attempt details: `docs/research/shot_pattern_analysis/phase2/validation/artifact_delivery_v3/remote_verification_summary.json`.
- **LaTeX Editor Blocker**: The local Tectonic 0.15.0 CLI compiled and visually reviewed the illustrated report successfully; the separate built-in Codex editor/compiler still fails before parsing because it cannot fetch its Tectonic bundle (#12120). Local CLI success does not clear the editor issue.

## In-Flight Branch Stacking and Dependencies

- **Phase 1 PR #11941**: Previously armed on `codex/shot-pattern-analysis`; do not touch its state.
- **Tools Pin**: Current integration gitlink is `6b6e4be55e7086a33db43b36d2632765be619bde`; exact submodule initialized for local tests. Historical evidence keeps its original pins.
- **Frozen Producer**: Presentation commit `4023111d2aaf8b20449a36f8691a295f4a94ceff` retains historical Tools pin `3678409`. Do not install, refresh receipts, or rerun it.
- **Artifact Routing**: Package `supplement_report_v2_layout_repair` but restore it as `supplement_report_v1` in the reader publication tree, preserving accepted report links and keeping the original study baseline intact.

## Must-Read Architecture Pointers

- Upstream renamed the native OpenSim mixed-marker observation reference to chapter 62: `manuals/upstreamdrift/chapters/62-native-opensim-mixed-marker-observation.qmd`. Archived peer records retain their historical chapter numbers.

- [Architecture Map](docs/architecture/C4.md) and [Shared Infrastructure](docs/agents/shared-infrastructure.md).
- [Resume Handoff](docs/research/shot_pattern_analysis/phase2/RESUME_HANDOFF.md), [Turnover](docs/research/shot_pattern_analysis/phase2/TURNOVER.md), and [Artifact Delivery Plan](docs/research/shot_pattern_analysis/phase2/review_followups/ARTIFACT_DELIVERY_PLAN.md).
- [CLAUDE.md](CLAUDE.md) and [AGENTS.md](AGENTS.md) govern integration, issue closure, and PR delivery.

## Verification Gates and Commands

- `python scripts/check_document_title_case.py AGENT_HANDOFF.md`
- `python -m src.tools.check_markdown_links AGENT_HANDOFF.md`
- Root's 282 provider/scoring/GUI/report/contact tests and Astra's 71 launcher-boundary tests passed; exact receipts live under `phase2/validation/`.
- Canonical pre-PR ran at `dac6618c58`: Ruff, mypy, consistency, fragments, and workflow checks pass. Affected tests report 5,861 passes, 34 failures, and seven errors in unchanged outside-study test files; this is scope evidence, not runtime-baseline proof. Fleet development-log policy also fails. The subsequent XML parser fix passed 41 focused vector tests, Ruff, mypy, and the central import-policy check; Semgrep was unavailable and advisory-skipped. A later corrected-PYTHONPATH full pre-PR attempt at `cb5eff6dcd` terminated with exit 143 during tests before a final summary; it is incomplete and does not replace the failed complete gate.
- Focused baseline audit: all 41 original failed/error nodes span 18 unchanged paths. At the base commit, 31 failed and seven errored; three passed and also pass in isolated current-branch tests. The full-suite cause of those three remains unresolved. See `phase2/validation/baseline_gate_audit_v1/`.
- Verify archive manifests and every restored member before publication. Upload success alone is insufficient.

## Do-Not List and Next Steps

- **Successor Integration**: Own branch `codex/shot-pattern-safe-followup` merges `2921f4b756` and repairs the frozen verifier fixture at `ef0f14a36b`; both verifier modules pass with the exact Tools pin. Incoming remote `9c9000` is reconciled at `cb4948eb1b`; frozen writer keyword compatibility is restored at `17ad207777` with four red-to-green tests and Astra review. A clean CI runtime matches all 126 evaluated `requirements-dev.lock` pins; the Rust wheel build and import probe, contact node, full DRY, architecture, and file-size checks pass locally. Repaired-source focused checks pass (67 tests) and final attempted pre-PR passed mypy on 148 source files; full canonical pre-PR remains incomplete because the required repository-wide development-log validator fails under separately owned #11935. No merge acceptance is claimed. Failed and incomplete attempts are preserved. See [Successor Progress](docs/research/shot_pattern_analysis/phase2/validation/safe_followup_v1/progress.json). AffineDrift #5005 metadata and scoped audit repairs are committed locally, full gates running; public routes remain 404.

- Preserve frozen science, baseline graphics, peer records, and armed Phase 1 PR state.
- Do not claim player calibration, isolated causal curvature benefit, per-round strokes gained, or successful compilation in the built-in LaTeX editor.
- PR #12278 remains unmerged. Integration branch `codex/shot-pattern-refinement-integration` includes remote `aecd90783b`; successor `codex/shot-pattern-safe-followup` reconciles `2921f4b756` and `9c9000`. Full repository DRY passes with the original approved baseline (#12299); receipt `phase2/validation/refinement_v3/full_repository_dry.txt`. All67 new architecture exceptions and both document-size waivers are removed (#12302). Current39-page report compiles from14 tracked hash-verified figures. Figure API regression #12309 has native writer tests. Full pre-PR remains outstanding; the exact local development-log validator exits 1 with duplicate entries, missing fields, 144 active entries against the 75 ceiling, and a size ceiling breach (#11935; separate active owner). AffineDrift historical pre-PR passed at `0e33178d`; successor publication repairs remain local, with full gates and deployment pending. Follow `phase2/validation/refinement_v3/progress.json`; do not rerun frozen simulations.
- Keep empirical validation #11984 deferred and compiler blocker #12120 open; close implementation issues only with qualifying merged evidence.


## Peer Record Index

The linked records preserve peer-reported status and provenance; this index does not independently validate those reports.

- [Handoff peer records, part 1](docs/agents/handoffs/archive/shot-pattern-integration-20261011/chunks/AGENT_HANDOFF_PEER_RECORDS-01.md)
- [Handoff peer records, part 2](docs/agents/handoffs/archive/shot-pattern-integration-20261011/chunks/AGENT_HANDOFF_PEER_RECORDS-02.md)
- [Handoff peer records, part 3](docs/agents/handoffs/archive/shot-pattern-integration-20261011/chunks/AGENT_HANDOFF_PEER_RECORDS-03.md)
- [Handoff peer records, part 4](docs/agents/handoffs/archive/shot-pattern-integration-20261011/chunks/AGENT_HANDOFF_PEER_RECORDS-04.md)
- [Imported integration peer-record index](docs/agents/handoffs/shot-pattern-integration-peer-records.md)
- [Snapshot integrity manifest](docs/agents/handoffs/shot-pattern-handoff-archive-manifest.json)

- [Exact Incoming Main Handoff Snapshot](docs/agents/handoffs/archive/shot-pattern-integration-20261011/AGENT_HANDOFF.incoming-9c9000.md.archive)
- [Incoming Main Peer Records, Part 1](docs/agents/handoffs/archive/shot-pattern-integration-20261011/chunks/AGENT_HANDOFF_INCOMING_RECORDS-01.md)
- [Incoming Main Peer Records, Part 2](docs/agents/handoffs/archive/shot-pattern-integration-20261011/chunks/AGENT_HANDOFF_INCOMING_RECORDS-02.md)
- [Incoming Main Peer Records, Part 3](docs/agents/handoffs/archive/shot-pattern-integration-20261011/chunks/AGENT_HANDOFF_INCOMING_RECORDS-03.md)
- [Incoming Main Peer Records, Part 4](docs/agents/handoffs/archive/shot-pattern-integration-20261011/chunks/AGENT_HANDOFF_INCOMING_RECORDS-04.md)

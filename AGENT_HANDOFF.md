# Agent Handoff: Current-State Index

## Active Study and Current Status

- **User-Requested Resume**: Goal pursuit is active after the usage reset. Start with [Resume Handoff](docs/research/shot_pattern_analysis/phase2/RESUME_HANDOFF.md).
- **Matched Delivery Study — Phase Two**: All 10.44 million attempted simulation rows are complete. Do not rerun simulations. Read [Scientific Report](docs/research/shot_pattern_analysis/phase2/REPORT.md) and [Design](docs/research/shot_pattern_analysis/phase2/DESIGN.md).
- **Scientific Acceptance**: Astra accepted the report after 406 numeric/count checks and 52 source-hash checks. These are geometric, model-conditional results; no measured player benefit or universal best shot shape is established.
- **Revised Report PDF**: Local Desktop delivery is39pages with conventional lateral-x/downrange-y axes,100 verified replay paths per pattern, expanded impact/flight/scoring methods and graphical summary. SHA256 `fd5506da87f5e619a550fb8cc871a0e8195b6fb98965a99809e3238a0f01508b`; review/delivery receipt `phase2/validation/refinement_v3/delivery_receipt.json`. The historical33page edition remains frozen; the revised39page article/PDF is live after AffineDrift #5005, with53/53 specific publication checks passing; mobile heading repair #5016 merged at `8162b803e96a`; exact deployed visual verification is pending.
- **Graphics**: All assigned baseline PNG reviews are complete. The 49 density-layout replacements (#12243) have root/Sol/Astra PNG acceptance and native PDF/SVG review. Coverage reconciliation accepts 2,304 pair PNGs plus 104 unchanged cell PNGs. All three archive bundles passed fresh local restoration.
- **Native Formats**: All 2,024 PDF and 2,024 SVG baseline renders completed. The 1,122 PDF bounding-box warnings were checked against painted pixels: 10,842 probes, minimum 6.4952 pt visible-ink clearance, no clipped edge ink. The renderer's final summary failed serialization; a separate successful finalizer preserved that distinction.
- **Integration and Publication**: The research prerelease is [published](https://github.com/D-sorganization/UpstreamDrift/releases/tag/shot-pattern-phase2-research-2026-10-10), bound to delivery checkpoint `cb5eff6dcd9df7d878f10578078b6fea69c8b681`. Its original payload is 21 inventory assets; two separate verification receipts were subsequently uploaded (23 release files total). Fresh download verification passed for the 21 payload assets, all three bundle member inventories, the 15-member illustrated source archive with 13 figures, and 98/98 report links. Receipt SHA256 `94f1fdea97207cb0eb8bf59db981d09cffd60d74d02f3ddd8957a3990e728843`: `docs/research/shot_pattern_analysis/phase2/validation/artifact_delivery_v3/remote-restoration-v3.json.gz`. The first restore attempt failed on archive-name mapping after asset hashes passed; Sol reviewed the corrected size/SHA bijection and the clean rerun exited 0 without asset changes. Attempt details: `docs/research/shot_pattern_analysis/phase2/validation/artifact_delivery_v3/remote_verification_summary.json`.
- **LaTeX Editor Blocker**: The local Tectonic 0.15.0 CLI compiled and visually reviewed the illustrated report successfully; the separate built-in Codex editor/compiler still fails before parsing because it cannot fetch its Tectonic bundle (#12120). Local CLI success does not clear the editor issue.

## In-Flight Branch Stacking and Dependencies

- **Phase 1 PR #11941**: Previously armed on `codex/shot-pattern-analysis`; do not touch its state.
- **Tools Pin**: Protected merged provider is `8a11691b4e9aee7d04dfc39628a8269ca561b07d` (Tools #5573). Gitlink, Cargo and Python pins agree; the exact Git runtime and all 19 shipped Signal files are verified. Historical evidence keeps its original pins.
- **Frozen Producer**: Presentation commit `4023111d2aaf8b20449a36f8691a295f4a94ceff` retains historical Tools pin `3678409`. Do not install, refresh receipts, or rerun it.
- **Artifact Routing**: Package `supplement_report_v2_layout_repair` but restore it as `supplement_report_v1` in the reader publication tree, preserving accepted report links and keeping the original study baseline intact.

## Must-Read Architecture Pointers

- Upstream renamed the native OpenSim mixed-marker observation reference to chapter 62: `manuals/upstreamdrift/chapters/62-native-opensim-mixed-marker-observation.qmd`. Archived peer records retain their historical chapter numbers.

- Manual governance: UP-D0 (#9066) and UP-D1 (#9067) remain release blockers. Edit only canonical `manuals/upstreamdrift` QMD; run `python3 -m scripts.check_design_manual_governance` for governed changes.

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

- **Successor Integration**: Own branch `codex/shot-pattern-safe-followup` reconciles the actual remote advance through `47df000da3` by ordinary rebase and actual main conflicts `8e9b4b974b` / `51f563776a` / `8a22fd99a8` / `21b24c2f57`. The `51f563776a` merge preserves all incoming swing-report work and regenerates its two conflicting inventories. #11935 is resolved by merged #12337. The three user-authorized inherited repairs, provider boundary repairs and strict ownership classifications are reviewed.
- **Final Local Gate**: Stable source `3d6a415bf9` passes all eight canonical RM-6 gates with available Semgrep: 7,594 affected tests pass,377 skip,43 deselect,1 known expected pass and21 passing subtests. Exact CI scan:209 rules/278 files/zero findings. Earlier failed evidence remains retained; unrelated scientific replay/provenance failures stay open in #12398.
- **Dependencies and Publication**: Tools #5573 is merged at `8a11691b4e9a`; all three pins, exact Git installation, all19 shipped Signal files and80 provider/impact tests pass. RM #2082 is merged. All29 Impact Explorer checks, Rust check and served bundle proof pass. AffineDrift #5005 is merged and53/53 live asset/PDF checks pass; #5016 merged through its protected queue at `8162b803e96a` with all required checks passing. Final deployed mobile review remains pending.
- **Protected CI Follow-Up**: All protected predecessor failures are repaired: five content-identity leaves, workflow-key literal formatting, strict historical-path provenance with16 TDD cases, and history-independent unit transport with12 verifier cases. The production verifier separately reproduces84 exact comparisons. The actual main capacity conflict preserves both completed records through the official API. Protected updated CI/merge remains pending; no scientific approval is renewed.
- **Latest Main Reconciliation**: Actual `21b24c2f57` conflicts in two generated agent-context maps are resolved by the exact pinned renderer. Eight incoming files and the single registry resolution remain exact;27 source-contract tests and structural manual governance pass, release stays blocked. Astra preservation review and the complete combined canonical gate pass; protected CI/merge pending.
- **Suite Classification Follow-Up**: Correct unit markers on ten new functions preserve all assertions and numerical criteria. Red/green suite-drift evidence and22 selected unit cases pass; all eight canonical gates pass on stable source3d6a415bf9. Updated protected CI/merge remains pending.
- **Unit Checkout Portability**: The full protected unit lane at `0e52533320` found one already-repaired manifest identity and two tests requiring unavailable historical objects in shallow CI. Six exact archived Git blobs now supply only bounded unit requests;17 focused cases and separate production full-history84-comparison verification pass. Production retrieval remains unchanged; the complete canonical gate now passes; protected updated CI is pending.
- **Latest Main Capacity Repair**: Actual main conflict `8a22fd99a8` is resolved with the official fragment API, preserving both completed #10783 and #9507 metadata. Both archive trees are byte-identical; all other entry titles/fields are unchanged. Validator passes with the existing22-entry WIP warning.
- **Release Boundary**: No frozen study rows or126 development-lock pins changed. No numerical, coverage, size, WIP or scientific acceptance limit is loosened. Manual inventory/release, physical acceptance and built-in compiler gates remain open. See [Successor Progress](docs/research/shot_pattern_analysis/phase2/validation/safe_followup_v1/progress.json).

- Preserve frozen science, baseline graphics, peer records, and armed Phase 1 PR state.
- Do not claim player calibration, isolated causal curvature benefit, per-round strokes gained, or successful compilation in the built-in LaTeX editor.
- PR #12278 remains unmerged pending protected CI/merge. Final source passes the complete canonical gate; publish by ordinary push. Existing auto-merge is armed; do not push or re-arm after queue entry. Final mobile verification waits for the actual #5016 deployment.
- Keep empirical validation #11984 deferred and compiler blocker #12120 open; close implementation issues only with qualifying merged evidence. Newly observed unrelated research replay/provenance failures remain open in #12398; all76actual historical fixture consumers pass, and no full research-suite pass is claimed.

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

- **Necromatcher Continuation**: Merged #11359 restricts historical fits to reviewed swing intervals and retains its research/native integration. See [Next-Agent Turnover](docs/development/necromatcher-next-agent-turnover.md) for the current checkpoints and open scientific gates. This reconciliation adds no new fit or physical qualification.

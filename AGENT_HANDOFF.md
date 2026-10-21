# Agent Handoff

Current state only. History is not deleted: every per-slice section that used to live here is in
[`docs/development/AGENT_HANDOFF_ARCHIVE.md`](docs/development/AGENT_HANDOFF_ARCHIVE.md) (moved verbatim by #12416).
Keep this file at or under 150 lines (AGENTS.md policy). Add one line per active item, with its issue number, and move the
detail of a finished slice to the archive instead of growing this file.

## Must-Read Pointers

- `AGENTS.md`, `CLAUDE.md`: binding policy, discovery workflow and the shared-infrastructure directory.
- `docs/development/DEVELOPMENT_LOG.md`: one `DL-#<issue>` state row per feature. `docs/development/HANDOFF.md`: canonical fleet handoff.
- `SPEC.md`: architecture contract; the change log is written by the fragment collate step, one row per PR.
- `scripts/config/design_manual_governance.json` and `manuals/upstreamdrift/calculation-registry.json`: design-manual authority.
- `docs/development/AGENT_HANDOFF_ARCHIVE.md`: full detail, branches and verification commands for every slice listed below.

## Gate Commands

- `python3 -m ruff check .` and `python3 -m ruff format --check .` (separate steps).
- `python3 -m pytest -m unit -n auto --timeout=60`; focused: `python3 -m pytest <paths> -q -n 0 --no-cov`.
- `python3 -m scripts.check_design_manual_governance` for any governed change.
- `python3 scripts/check_doc_size_budget.py`, `python3 scripts/check_root_clutter.py`, `python3 scripts/check_document_title_case.py <docs>`.
- `python3 shared_scripts/changes_fragment.py new --issue N --summary "..."` instead of editing the SPEC change log or development log.
- Headless only: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy`.

## Do Not

- Do not use `--no-verify`, `FLEET_GUARD=off` or `gh pr merge --admin`; do not loosen a tolerance, budget or coverage floor.
- Do not edit `vendor/ud-tools/`; change Tools upstream and bump the pin.
- Do not edit generated LaTeX, PDF, DOCX or HTML manuals; edit the `manuals/upstreamdrift` QMD source only.
- Do not claim physical qualification from software evidence, an animation or a process exit code.
- Do not touch another session's PR or branch; do not launch GUI processes.

## Markerless Mocap Program (#9063)

- Tools #4706 owns capture and contract schemas; UpstreamDrift #9069 (folded into #9422) owns app orchestration and makes no physical-lab qualification claim.
- ADR-0041 Amendment 1 (#9630, #9619): consumer-side self-calibration fitters live in `src/motion_capture/reconstruct/`; Tools keeps vendor-neutral reference geometry.
- Real-data acceptance is deferred to `docs/development/planning/DV-9619.md`; the rig soak to `DV-9613.md`.

## CI Throughput (RM Epic #2215)

- UD-CI-032 #13573: PR #13584 (`cursor/cancel-merge-queue-audit-1f01`) — merge-queue-safe `cancel-in-progress` audit; RM #2216 landed — arm auto-merge once `quality-gate` is green.
- UD-CI-033 #13595: `codex/13595-ci-lint` fixes Actionlint/ShellCheck blockers and adds the governed RM #2253 policy consumer with trusted event forwarding, a Repository_Management contents-read App token, and `d-sorg-fleet-docker` routing. App checkout succeeded in run 39655811487. Fork PRs skip only this App-token policy job because fork secrets are unavailable; merge_group still runs the policy, with no fork-head code execution or credential fallback. Fleet policy action pins and inventory metadata are covered by actual-guard tests; pinning, inventory, and local-runner guards pass. Commits 4df00351ff and e8b8c1f3cf route the fleet gate and architecture-map contract to `d-sorg-fleet-docker`, with the architecture route covered by a CI infrastructure test. The publication detector’s old literal-echo test failed in unit shard 1; it now exercises the actual Bash step with fake git/output files for PR publication and unrelated paths plus merge-group/push full-suite behavior. Run 39661545707 exposed a merge-base race: event anchor `23a8b8b` and checked-out merge `0942fac` selected strict mode for issues 13579 and 13595. PR baseline resolution now invokes only the protected RM helper with the agreed event/head/repository/default-branch inputs and preserves its full JSON proof in the third artifact receipt. If the helper is not yet published, the consumer writes an explicitly unverified `helper_unpublished` receipt with a null baseline and defaults to strict; present helper proofs must be verified, have proof kind `pull_request_protected_merge_parent`, and contain a full baseline SHA, otherwise the step fails closed. Merge-group continues to use its event base. Fixtures cover exact CLI flags/schema, protected-path selection, strict bootstrap fallback and bad-proof failures. The helper exists in `Repository_Management-pr-baseline-2253`, but its current CLI omits the agreed `--default-branch` option; reconcile that provider mismatch and copy the canonical workflow before qualification. The central gate still rejects existing fork-hosted alternatives and classify conjunction placement (23 affected job instances across `Jules-Issue-Mention-Handler.yml`, `PR-Comment-Responder.yml`, `ci-requires-opensim.yml`, and `ci-standard.yml`); Hume's central guard follow-up handles these patterns, so leave those workflows unchanged. App checkout and pre-push failure guidance are verified; capacity debt remains visible and unchanged. Pending: trusted-base helper API alignment/publication, central guard/fork-policy and protected-registry publication, preflight fallback-worker fix, and CI cohort drain. Then run full `pre_pr` with the central gate enabled. Do not push until the cohort drains and protected registry extensions are published; no UD PR is open.

## Required Before Continuing

- Read `AGENTS.md`, `CLAUDE.md`, and `docs/development/DEVELOPMENT_LOG.md`.
- Epic #10584 and all child packages TB-00 through TB-12 are closed. Full-body models remain under governing programs (#10363, #10378, #10430, #10440).
- Manual governance: UP-D0 (#9066) and UP-D1 (#9067) remain release blockers. Edit only the `manuals/upstreamdrift` QMD source and run `python3 -m scripts.check_design_manual_governance` for governed changes.
- Update this handoff, the development-log entry, and exactly one `SPEC.md` change-log row for every substantive PR.

## Active Epics and Slices (One Line Each; Detail in the Archive)

OpenSim musculoskeletal barbell lifts (OSL, Epic #13038; research record `docs/research/lift_models/lift_models.tex`; state as of 2026-10-20):

- OSL-3 #13041 and OSL-4 #13042: all five lifts complete and unqualified; the bench press landed in #13431, #13532 and #13536 (Hicks force from feet plus bench). Open: the 511.6 N bench pelvis residual and a 4437 N support peak (2.5 times the lifter-plus-bar weight) not yet explained.
- OSL-5 #13043: PR #13521 (armed) adds the deadlift, snatch and clean and jerk; per-lift floors restored; acceleration bound 5e-4 with a `Tolerance-Change-Evidence` trailer. The bench press waits on this PR, then needs its own solve.
- OSL-7 #13044: PR #13512 (armed, Refs) adds solved poses, the planted-feet goal and `docs/development/lifting/evidence/opensim_predictive/RUN_HISTORY.md` (ControlTower runs C to S). Stage 1 does not converge; the owner chooses welded feet, a 30 to 50 interval mesh, or 180 kg contact retuning.
- OSL-8 #13045: PR #13526 (open, not armed; three Gemini rounds rejected) renders a different motion than the OSL-3 IK; see the review comment there before continuing. Verify every render frame by eye.
- Also open: OSL-14 #13057, OSL-15 #13093, OSL-16 #13104, OSL-17 #13131 (floating-base regulation is an owner decision), LIFT-5 #11745, LIFT-6 #11746 and #13072.
- Real-OpenSim tests need `UPSTREAMDRIFT_MSK_BASE_MODEL` pointing at `RajagopalLaiUhlrich2023.osim` and `--timeout=1400`; route imports through `tests.helpers.opensim_support.require_opensim()`.

open-fp vertical force plates as measured constraints (Epic #13061; plan and DV-13061 in #13062):

- Contract OF-01 #13065, reader OF-02 #13066, readings at model time OF-05 #13069, vertical-cell observation rows with sensing-plane lever OF-11 #13067, family-wise elimination screen OF-12 #13070. AGPL boundary: files and network only; all data unqualified until DV-13061.

Segmental spine (Epic #12735, ADR-0055):

- Software pathway landed and consolidated by SPN-16 #12773: manual chapter 66, `docs/research/segmental_spine/README.md#how-to-run-a-study`; all results unqualified, owner-blocked on #12749, #12737, #12755, #12758, #12763, #12770.

Shot-pattern study (straight, draw and fade; #12409, PR #12418):

- Matched phase-two study and article with 2,000-resample intervals and the clubhead-assumption band; consolidates #12278, #12410 and #12411. Start at `docs/research/shot_pattern_analysis/phase2/RESUME_HANDOFF.md`; the pre-consolidation detail is in the archive section of that name.

Grip and club calibration (#12221, #12042, #12117):

- Per-capture grip rotation adopted and refit on the fixed clusters, measured driver cluster, fixed cluster in the static trial: #12221.
- 7-iron MSK grip calibration, neutral hip zero-twist and turn render, retarget turn targets, Drake IK coordinate order, shoulder/trunk/pelvis turn metrics: #12042.
- Club-face fixtures on the current spec, reference ZMP split at ball impact: #12117; hip rewrite uses the spec pelvis alignment: #12109.
- Drake and MuJoCo IK parity weld rotation row: #12039; nominal 7-iron IK and club-face triad gate: #12030.
- Source-preserving BUET-Hamner candidate #12157; zero-subtalar reduction #12167; zero-MTP CustomJoint reduction #12176.

Golf Chain Visualisation, Epic #11706 (GCV):

- Shared ground reaction core GCV-1; GRF engine wiring and CoP consolidation GCV-2/GCV-6: #11708, #11712; force arrow scaling GCV-4.
- Grip wrench core and per-engine extraction GCV-8: #11713, #11714; club force and torque overlays GCV-10: #11716.
- Web club head and golfer head GCV-11/GCV-12: #11717, #11718; golf ball in engine scenes GCV-13: #11719.
- Native export backends, quarter-speed variants, impact time, high-FPS schedule GCV-14: #11720; MyoSuite v3 export worker #11997.
- ClubheadSeries adapters GCV-16 #11722; impact parameters panel GCV-17 #11723; ground reaction manual slice GCV-18 #11724.
- Clubhead speed timing GCV-20: #11767; MyoSuite hip retarget #12052; right knee flexes negative like the left #12057.

Open Swing Visualisation grip and gaze (OSV, Epic #11726):

- Two-hand grip pose OSV-2 #11728; address foot progression OSV-4 #11730; lead-foot address and mirrored left hip axis OSV-6 #11737.
- Contact grip, bushing and impact-phase grip OSV-7: #11739; clubface roll at address OSV-8 #11755; face orientation residual OSV-10 #11759.
- Shared club in MSK OpenSim models and two-hand club in tour-matching variants OSV-9: #11756.
- Gaze neck (closed-loop, forward-dynamics, weight selection, head stabilisation): #11729.

Native F-series (feedback, muscle, Moco, replay):

- F10i native platform integration #12060; F09p prediction integrity #12055; F09o native tools #12044; F09n command solve #12036; F09m tracking objective #12023.
- F09l persistent MyoSuite search #12014; F09j feedback recording #12003; F09i task and independent replay #11983; F09b observation scoring #11837.
- F09e Pinocchio marker FK #11914; F09f MyoSuite excitation replay #11918; native Pinocchio replay #11900; F05d manifold control #11932; F02 manifold feedback #11946.
- F07 muscle qualification guard #11791, asset admission #11819, constrained Moco and replay #12143/#12147/#12173, mixed replay scoring #12195.
- F07 557-muscle excitation diagnostic #12182, source-body inertia gate #12191, shoulder lamina candidate #12211, unqualified Moco feasibility #12225.
- Feedback controls planning #11784; native wrapped-path closure #12234; native contact source closure #12244; physical inertia helper #12183; manufactured native diagnostics #12275.

Necromatcher and matching:

- Reviewed fitting scope #11414; native fit delivery #11240; source-bound shaft continuation #11318; isolated native hypothesis admission #11376; workspace handoff #11239 (merged).
- Raw shaft fragment core, arm morphology, scoped camera, replay impact checkpoints: #11357, #11394, #11464, #11487; local research golf #11493.
- MOSAIC model-aware matching #11532; OpenCap to OpenSim #11400; MyoFullBody muscle-driven swing Epic #11642.
- Centroidal feasibility filter v2 #11669; finish feasibility balance #11667; lift pack parity baseline LIFT-1 #11741 (Epic #11740).
- Simscape: URDF exchange and run-102 playback #11569; same-input parity chapter SIP P-9 #11614 (Epic #11605); OpenSim swing #11617; native OpenSim geometry #11903; MeshCat framing NV-9 #11697.
- Unified spec-exported URDF contract across engines and ledgered MyoSuite topology divergence #11570.
- Character Builder docs and tutorial CMB-12 #11663 (open: CMB-6 #11657, CMB-7 #11658).

Motion matching and tour baselines:

- Exception-identity split at shared contracts #11175; reduced-model and club-only claims MMR-11 #11097; best-candidate viewer fixes MMR-16 #11102.
- `vendor/ud-tools` bump with K0 and K3a #10944; neural-model program NM-09 to NM-12: #10624, #10625, #10626, #10627.
- Tour baselines (exact-capture matching, fail-closed qualification): #10829, #10830, #10831, #10799, #10800, #10844, #10849; TB-12 #10597 merged.
- Hip-calibrated receipt provenance #10271; GolfSwingVisualizer MATLAB consolidation #9225; docs-consistency cross-repo paths #10743.
- Tiger 2000 and Ben Hogan baselines: see the archive section of that name.

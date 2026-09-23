# Matched Swing Program — Single Source of Truth

## Continuation Review Warning

The 2026-09-18 source review found that this document's `acceptance_contract.py`
reference does not exist on reviewed main. The implemented evaluator is
`src/shared/python/motion_matching/acceptance.py`: G1 is the 0–0.85 s dynamic
horizon (whole 25 mm, early 12 mm, terminal 35 mm), not a 30 mm full-capture IK
milestone. Historical tables below conflict with that evaluator and epic
#10363 and MUST NOT be used to accept a run. MS-100 (#10374) must reconcile
and version the complete contract; do not change thresholds during fitting.
Read the [continuation prompt](AGENT_CONTINUATION_PROMPT.md) and exact receipts.
The review does not qualify the evaluator or replace missing physical evidence.

**Program Lead:** Dieter Olson (`agent:local`)  
**Governing Epic:** [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363)  
**Tour Baselines Epic:** [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584) ([Tour Baselines Inventory](../../plans/tour_baselines/README.md))  
**Club-Only Matching Epic:** [#10602](https://github.com/D-sorganization/UpstreamDrift/issues/10602) — shared observation contracts land in CO-01 [#10605](https://github.com/D-sorganization/UpstreamDrift/issues/10605) (`ClubObservation` + `club_calibration`; evidence under `docs/plans/club_only_matching/evidence/`). Final operator/reproduction turnover: [REPRODUCTION_GUIDE.md](../../plans/club_only_matching/REPRODUCTION_GUIDE.md) (CO-10 [#10614](https://github.com/D-sorganization/UpstreamDrift/issues/10614)). Club-only profiles do not satisfy G3; epic #10602 stays open until mandatory native fits land.  
**Specification:** `SPEC.md` § Motion Matching Program  
**Acceptance Contract:** [`GATES.md`](GATES.md) (`src/shared/python/motion_matching/acceptance.py`)  
**Wave Structure:** [`WAVES.md`](WAVES.md)  
**Run Ledger:** `reports/matched_swing_ledger.json` (`src/shared/python/motion_matching/ledger.py`)

---

## 1. Program Goal & Architecture

The **Matched Swing Program** delivers a physically realistic, cross-engine matched golf swing biomechanics pipeline across all six fleet physics engines:

1. **MuJoCo** (Primary kinematic and forward-dynamics reference)
2. **Pinocchio / Crocoddyl** (Analytical rigid-body dynamics & optimal control)
3. **Drake** (High-fidelity contact & trajectory optimization)
4. **OpenSim** (Musculoskeletal and biomechanical validation)
5. **Simscape** (Historical tour capture authority)
6. **MyoSuite** (Neural control and muscle actuation)

### Core Mandates:

- **No Threshold Relaxation:** Numerical tolerances are fixed. G1 kinematic fit is an engineering milestone ($\le 30$ mm IK), not dynamic match or product release.
- **Fail-Closed Gate Evaluation:** Missing files, missing sensor channels, unverified SHA-256 hashes, or non-finite outputs fail closed.
- **Dual-Club Full-Swing Coverage:** G3 professional release requires verified driver and 7-iron captures across all engines. Partial or strength-limited states do not qualify.
- **Path-Anchored Numbers:** Every metric cited in documentation must anchor directly to a verified receipt JSON field.

---

## 2. Where the Numbers Come From

To prevent drift between handoff notes and committed evidence, all headline numbers are categorized into canonical calibrated runs vs baseline runs, verified by `CANONICAL_RUN.md` and continuous freshness tests.

| Metric                  | Quoted Value | Canonical Receipt Path                                                                            | Field Path                        | Notes                                                                                  |
| ----------------------- | ------------ | ------------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------- |
| **Driver Address RMS**  | **5.1 mm**   | `docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json` | `address.calibrated.marker_rms_m` | Full neutral static-trial marker calibration (`--static-seeds`)                        |
| **Driver IK RMS**       | **27.3 mm**  | `docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json` | `ik.marker_rms_m`                 | Calibrated upper/lower attachments, bounded wrists                                     |
| **Driver Dynamics RMS** | **74.6 mm**  | `docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json` | `dynamics.marker_rms_m`           | Unactuated floating root, contact-aware shooting fit                                   |
| **7-Iron Address RMS**  | **4.4 mm**   | `docs/development/full_body_models/evidence/ground_support/anthro_iron_zmp/receipt.json`          | `address.calibrated.marker_rms_m` | Neutral static trial marker calibration                                                |
| **7-Iron IK RMS**       | **28.6 mm**  | `docs/development/full_body_models/evidence/ground_support/anthro_iron_zmp/receipt.json`          | `ik.marker_rms_m`                 | Full 654-frame IK on 7-iron tour capture                                               |
| **7-Iron Dynamics RMS** | **144.0 mm** | `docs/development/full_body_models/evidence/ground_support/anthro_iron_zmp/receipt.json`          | `dynamics.marker_rms_m`           | Dynamics ZMP filter inside foot support polygon                                        |
| **Driver Baseline IK**  | **52.3 mm**  | `docs/development/full_body_models/evidence/ground_support/anthro_driver/receipt.json`            | `ik.marker_rms_m`                 | Nominal attachments (without static-seeds), unwidened leg bounds (1.0), hip zero-twist |
| **7-Iron Baseline IK**  | **72.0 mm**  | `docs/development/full_body_models/evidence/ground_support/anthro_iron/receipt.json`              | `ik.marker_rms_m`                 | Nominal attachments (without static-seeds), unwidened leg bounds (1.0), hip zero-twist |

For detailed factor attribution on the 27.3 mm vs 52.3 mm IK baseline shift, see [`CANONICAL_RUN.md`](../full_body_models/evidence/ground_support/CANONICAL_RUN.md) and [`bisect_receipt.json`](../full_body_models/evidence/ground_support/bisect_receipt.json).

---

## 3. Automated Program Status & Ledger Matrix

### MS-43 Native Engine Nightly Lane (#10342)

| Engine   | Receipt                                                         | Lane status                         | Freshness gate                          |
| -------- | --------------------------------------------------------------- | ----------------------------------- | --------------------------------------- |
| OpenSim  | [opensim_receipt.json](evidence/nightly/opensim_receipt.json)   | bootstrap — refresh on ControlTower | warn &gt; 7 d, fail &gt; 30 d on `main` |
| MyoSuite | [myosuite_receipt.json](evidence/nightly/myosuite_receipt.json) | bootstrap — refresh on ControlTower | warn &gt; 7 d, fail &gt; 30 d on `main` |

Refresh via `bash scripts/ci/run_native_engine_lane.sh --engine opensim --venv /home/dieterolson/opensim-10003`. See [evidence/nightly/README.md](evidence/nightly/README.md).

### MS-62 Simscape 44-to-27 Coordinate Slice (#10349)

| Scope                  | State                                                                 | Evidence                                                                                                                                                       |
| ---------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Kinematic slice + map  | JSON map and projection implemented; virtual-work decomposition green | [receipt.json](../../../evidence/matched/driver_g1_simscape_slice/receipt.json), [parity.json](../../../evidence/matched/driver_g1_simscape_slice/parity.json) |
| Simscape native replay | unqualified — R2025b boundary-load validation pending                 | [run_manifest.json](../../../evidence/matched/driver_g1_simscape_slice/run_manifest.json)                                                                      |

### MS-60 Simscape R2025b Run Management (#10347)

| Scope                               | State                                                              | Evidence                                                                                                                                                                                                                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Runner recipe + run manifest schema | Implemented (R2025b fail-closed)                                   | scripts/matlab/run_simscape_candidate.ps1, simscape_run_manifest.py                                                                                                                                                                                                                                        |
| Run-102 candidate + GIF in tree     | Committed                                                          | [candidate.npz](../simscape_tour_matching/native_evidence/two_window_fit_9967_102/candidate.npz), [playback.gif](../simscape_tour_matching/native_evidence/two_window_fit_9967_102/playback.gif), [run_manifest.json](../simscape_tour_matching/native_evidence/two_window_fit_9967_102/run_manifest.json) |
| Native licensed replay              | DeskComputer R2025b Update 5 recorded; local unlicensed hosts skip | [qualified_candidate_replay.json](../simscape_tour_matching/native_evidence/two_window_fit_9967_102/qualified_candidate_replay.json)                                                                                                                                                                       |

### MS-61 Simscape Topology + Full-Marker Terminal (#10348)

| Scope                                     | State                                                                                        | Evidence                                                                                                                                                                                                                                     |
| ----------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Topology classification (27-DOF, no neck) | Implemented fail-closed                                                                      | `simscape_topology.py`; [topology_report.json](../simscape_tour_matching/native_evidence/two_window_fit_9967_103/topology_report.json)                                                                                                       |
| Dual terminal disclosure                  | Full + head cluster required; body-excluding-head diagnostic only                            | `full_marker_terminal.py`; [terminal_breakdown.json](../simscape_tour_matching/native_evidence/two_window_fit_9967_103/terminal_breakdown.json)                                                                                              |
| R2025b runtime/license + Pinocchio parity | Receipts reuse run-102 identity (no license-cap inference)                                   | [runtime_license_receipt.json](../simscape_tour_matching/native_evidence/two_window_fit_9967_103/runtime_license_receipt.json), [parity_receipt.json](../simscape_tour_matching/native_evidence/two_window_fit_9967_103/parity_receipt.json) |
| Native run-103 Fit / G1 full-marker pass  | **Blocked** — no invented success; DeskComputer R2025b Fit + MS-104 neck/full-body if needed | [native_gate.json](../simscape_tour_matching/native_evidence/two_window_fit_9967_103/native_gate.json) (status=`blocked`; full-marker terminal still > 35 mm from run-102 source)                                                            |

### MS-21 Native Replay Continuation (#10336)

| Scope                             | State                                                                              | Evidence                                                                                                                                                   |
| --------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Merged Driver Candidate in MuJoCo | Replay path implemented; configuration and source dynamics unverified; G1 rejected | [Receipt](../../../evidence/matched/driver_full_mujoco_replay/receipt.json), [Reproduction](../../../evidence/matched/driver_full_mujoco_replay/README.md) |

The source analytic-fit receipt omits armature, interpolation, candidate hash
and root-assistance history. Same-state marker agreement is a kinematic check;
it does not establish native dynamics parity or a qualified full swing.

### MS-111 Pinocchio G2/G3 Continuation (#10385)

| Scope                                       | State                                                                                           | Evidence                                                                                                              |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| G2/G3 schedules + integrator parity         | Software contracts landed; `--ms111-schedule` on Pinocchio fitter                               | `src/shared/python/motion_matching/pinocchio_g2_g3.py`                                                                |
| Independent replay package (save/reopen)    | Armature-propagated portable package; fail-closed schema                                        | [continuation_contract_status.json](evidence/ms111/continuation_contract_status.json)                                 |
| Native driver/iron G2/G3 desk qualification | **Blocked** — awaits MS-107 accepted same-integrator G1 + MS-100 receipts; no native claim here | [continuation_contract_status.json](evidence/ms111/continuation_contract_status.json) (`claims_native_success=false`) |

The section below is generated directly from `reports/matched_swing_ledger.json` by running:

```bash
python scripts/generate_matched_swing_status.py --write
```

## Program State 2026-09-18 (Hand-Written; Truth Reset #10381)

| Engine    | Verified dynamic match                                                                                                           | Best kinematic fit                         | Honest status                                                                                                                                                                                                               |
| --------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pinocchio | FDDP 0.30 s window 19.7 mm whole, replay-identical (`w030r`, ControlTower)                                                       | GN IK 29.6 mm (address), 32 mm over 0.30 s | best G1-horizon candidate 46.8 mm whole, replay-identical, converged (`evidence/matched/driver_g1_crocoddyl_rk45_b100/`, REJECTED: early 25.7, terminal 64.3, yaw 9.3 deg, penetration 17 mm); MS-107 owns the next attempt |
| MuJoCo    | none accepted (tracking 74.6 mm variant, 89 mm primary)                                                                          | 27 mm canonical IK                         | replay path for Pinocchio candidates exists (PR #10448) and rejects the decoupled controls at 0.94 m                                                                                                                        |
| Drake     | none; setup parity 6e-6 m                                                                                                        | 176 mm (FB-4)                              | needs the shared fitter on its plant (MS-13/MS-30) and native IK (MS-17)                                                                                                                                                    |
| OpenSim   | Moco rungs 0.10/0.30 s at the 41 to 42 mm calibration floor; 0.60 s converged but open-loop replay 81 mm whole / 204 mm terminal | 67.8 mm (OS-3b)                            | OG-01..09 model work merged (#10414); no accepted G1; MS-40 shared-document model is the lever                                                                                                                              |
| MyoSuite  | none                                                                                                                             | kinematic replay receipt (MS-52)           | fail-closed provider (MS-50); MS-51 pinned myo_sim + dual-grip golfer scenes ready for smoke (`parity_budget_qualified=false`); 15 mm parity and G1 dynamics still open                                                     |
| Simscape  | run-102, 0 to 0.85 s, 20.3 mm whole but terminal 40.3 > 35 mm; Pinocchio parity 60 um                                            | n/a                                        | cross-validation lane; not a showpiece model                                                                                                                                                                                |

Rules restated: a ledger row is accepted only by `acceptance.py` (non-empty `gates`); self-declared `accepted` flags are UNVERIFIED. Every `evidence/matched/*` receipt on main is REJECTED (see each `reevaluation.json`). The fast decoupled pipeline (`scripts/match_pinocchio_c3d.py`) is an analysis product until a replay passes.

<!-- generated:matched-swing-status -->

### 1. Cross-Engine Engineering Progress Matrix

Auto-generated from committed run ledger (`reports/matched_swing_ledger.json`, 112 committed receipts scanned).

| Engine        | Candidate Lanes                                                                                                                | Evaluated Captures | Best IK RMS | Best Dyn RMS | Receipts | Engine Status                                                            |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------ | ----------- | ------------ | -------- | ------------------------------------------------------------------------ |
| **Mujoco**    | anthropometry, fb4_calibration, fb5_matching, fb6_parity, ground_support, matched, replays, setup_parity, viewer, visual_layer | driver, iron       | —           | —            | 34       | ⚙️ Engineering Milestone (G1 IK pass; unqualified until Simscape parity) |
| **Pinocchio** | fb3_kinematics, fb4_calibration, fb6_parity, ground_support, matched, replays                                                  | driver, iron       | —           | —            | 11       | ⚙️ Kinematic Milestone (Pink QP active; Crocoddyl lift in progress)      |
| **Drake**     | fb3_kinematics, fb4_calibration, fb6_parity, ground_support, matched, replays                                                  | driver             | —           | —            | 6        | ⚙️ IK 47 mm / tracking 382 mm REJECTED                                   |
| **Opensim**   | ground_support, matched, tour_matching                                                                                         | driver             | —           | —            | 11       | ⚠️ Staged (Moco track problem under MS-102)                              |
| **Simscape**  | native                                                                                                                         | driver             | —           | —            | 40       | 🏛️ Historical Tour Authority (Simscape lane baseline)                    |
| **Myosuite**  | matched                                                                                                                        | driver             | —           | —            | 1        | 🔬 Experimental (Fail-closed; MS-50 corrective landed)                   |

### 2. Full-Swing Qualification Ladder (Fail-Closed Gates)

Per Owner-Authorized Contract Revision (MS-100 #10374 / MS-104 #10378 / MS-106 #10380):
Partial, reduced-model, and strength-limited outcomes do not satisfy G3 release. All six engines remain required.

| Gate                           | Criterion                                                     | MuJoCo              | Pinocchio      | Drake          | OpenSim    | Simscape    | MyoSuite   | Gate Status               |
| ------------------------------ | ------------------------------------------------------------- | ------------------- | -------------- | -------------- | ---------- | ----------- | ---------- | ------------------------- |
| **G1: Kinematic Fit**          | Whole-swing marker RMS $\le 30$ mm, 0 RoM violations          | ✅ Passed (27.3 mm) | 🔄 In Progress | 🔄 In Progress | ⏳ Pending | 🏛️ Baseline | ❌ Blocked | **G1 Milestone Active**   |
| **G2: Dynamic Ground Support** | GRF in support polygon, floating root tracked                 | ✅ Passed (74.6 mm) | 🔄 In Progress | ⏳ Pending     | ⏳ Pending | 🏛️ Baseline | ❌ Blocked | **Partial (MuJoCo only)** |
| **G3: Professional Release**   | Dual-club (driver+iron), all 6 engines, cross-engine verified | ⏳ Pending          | ⏳ Pending     | ⏳ Pending     | ⏳ Pending | ⏳ Pending  | ❌ Blocked | **Open (Blocks Release)** |

### 3. Matched Swing Release Issues Roadmap

| Issue                                                                                | Title                                                        | Tier       | Accountable Role     | Blocker / Dependency                       | Next Executable Action                                 |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------ | ---------- | -------------------- | ------------------------------------------ | ------------------------------------------------------ |
| [**MS-100**](https://github.com/D-sorganization/UpstreamDrift/issues/10374) (#10374) | Fail-closed physical acceptance contract and validator       | Governance | `acceptance-lead`    | None (spec-first)                          | Implement MS-100 schema and fail-closed gate validator |
| [**MS-101**](https://github.com/D-sorganization/UpstreamDrift/issues/10375) (#10375) | Drake native full-body trajectory optimization               | P1         | `drake-agent`        | Drake QP solver setup                      | Port trajectory optimization into Drake adapter        |
| [**MS-102**](https://github.com/D-sorganization/UpstreamDrift/issues/10376) (#10376) | OpenSim Moco full-body muscle-driven tracking                | P1         | `opensim-agent`      | Moco CASADI license & memory budget        | Assemble full-body Moco track problem                  |
| [**MS-103**](https://github.com/D-sorganization/UpstreamDrift/issues/10377) (#10377) | Pinocchio Crocoddyl full-body optimal control integration    | P1         | `pinocchio-agent`    | Two-window terminal cost tuning            | Wire Crocoddyl action models into full pipeline        |
| [**MS-104**](https://github.com/D-sorganization/UpstreamDrift/issues/10378) (#10378) | Driver and 7-iron dual-club G3 coverage across all engines   | P1         | `full-body-lead`     | Single-club evidence on non-MuJoCo engines | Run and record dual-club suites per engine             |
| [**MS-105**](https://github.com/D-sorganization/UpstreamDrift/issues/10379) (#10379) | Cross-engine physical convergence and numerical verification | P2         | `verification-agent` | Step-size and GRF divergence checks        | Run cross-engine step convergence analysis             |
| [**MS-106**](https://github.com/D-sorganization/UpstreamDrift/issues/10380) (#10380) | Professional release gate and verified matched badge         | P2         | `release-auditor`    | G3 multi-engine cross-validation pass      | Sign off release verification audit                    |
| [**MS-107**](https://github.com/D-sorganization/UpstreamDrift/issues/10381) (#10381) | Native automated engine benchmark regression suite           | P2         | `ci-infra`           | Runner execution time limits               | Add nightly automated cross-engine benchmark lane      |
| [**MS-108**](https://github.com/D-sorganization/UpstreamDrift/issues/10382) (#10382) | Matched swing program end-to-end evidence release audit      | P2         | `governance-lead`    | MS-100 through MS-106                      | Final immutable evidence freeze and turnover           |

<!-- end-generated:matched-swing-status -->

---

## 4. Multi-Agent Work Protocol

Before picking any child issue in Epic #10363:

1. **Check Claim:** `python -m scripts.check_agent_claim --repo UpstreamDrift --issue <N>` (run from `Repository_Management`).
2. **Post Lease:** `python -m scripts.post_agent_lease --agent <id> --session <uuid> --repo UpstreamDrift --issue <N>` (TTL 2 h).
3. **Branch & Worktree:** Create an isolated worktree from `origin/main` (`_wt_<agent>_<issue>`). Never work in another agent's worktree.
4. **TDD / DbC / LoD:** Write failing tests first. Use `@precondition`/`@postcondition` on public interfaces. Restrict physics engine SDK imports strictly to engine adapter layers.
5. **Budgets & Conventions:** Verify `check_architecture_budget.py` and `ruff check --fix` pass. Prefix commit with conventional type (`feat:`, `test:`, `docs:`, `chore:`).
6. **PR & Hotspots:** Open PR referencing `Closes #N` with `agent:<id>,spec-exempt`. Resolve hotspots (`DEVELOPMENT_LOG.md`, `HANDOFF.md`) once, last, before merge.

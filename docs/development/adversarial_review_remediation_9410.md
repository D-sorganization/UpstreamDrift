# Adversarial Product Review Remediation Ledger

Tracking: [Epic #9410](https://github.com/D-sorganization/UpstreamDrift/issues/9410)
(program Repository_Management#1505, Phase 2, Pillar P2). It re-parents the
2026-08-21 adversarial product review defects #8820–#8943 plus #8360, #8641,
#8843, #8846, #8853, #8861–#8870, #8874–#8876 and #8894, grouped into six
clusters so each can be one agent wave.

This is a reconciliation ledger, not a completion claim. Every row was
re-checked against `main` at `db4fe88c4` on 2026-09-16 by searching the
first-parent history for the child issue number and, for every residual, by
reading the cited source location. A row is **landed** only when the merge
commit is an ancestor of that SHA; fixes that exist on an unmerged branch are
recorded as residual with the branch named. "Not re-verified" means no commit
references the issue and the defect was not re-read in source for this ledger;
treat it as open.

## Status Summary

| Cluster                          | Children | Landed on `main` | Residual |
| -------------------------------- | -------: | ---------------: | -------: |
| A — Broken at the root           |        6 |                5 |        1 |
| B — Registries and parity        |        7 |                3 |        4 |
| C — Duplicate stacks             |        9 |                1 |        8 |
| D — Destroys work / lies to user |       10 |                7 |        3 |
| E — Provenance and units         |        7 |                6 |        1 |
| F — Launcher UX and performance  |       22 |               12 |       10 |
| **Total**                        |   **61** |           **34** |   **27** |

Cluster D counts #8880 as landed for the shared mechanism only; see its row.

## Cluster A — Broken at the Root

| Issue | Finding                                                | Status on `db4fe88c4`                                                                                              |
| ----- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| #8894 | Shared theme import broken repo-wide                   | Landed — `dba24ceb7` (#9440) resolves the tools-canonical clusters from the pinned Tools tree.                     |
| #8843 | In-app Help broken at root                             | Landed — `3dd62c7d6` (#9990) reconciles help paths and topic mappings.                                             |
| #8846 | 24/25 tools have no help hook; `build_help_menu` dead  | Landed — `2496377cb` (#9992) adopts the shared help menu and calculation-sheet affordances.                        |
| #8641 | Dead first-party imports in `model_generation` facades | Landed — `29bc9513a` (#9993) restores the facades and repairs the imports.                                         |
| #8360 | Launcher splash stalls on optional provider failure    | Landed — `35c69c289` (#9951) bounds splash lifetime and degrades on provider failure.                              |
| #8938 | Launcher startup blocks the GUI thread                 | Residual — no commit references it; #9951 bounds the splash but the startup thread model was not re-verified here. |

## Cluster B — Registries and Parity

| Issue | Finding                                                                | Status on `db4fe88c4`                                                                                                                                                                                                                  |
| ----- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #8853 | Two tile registries plus three code registries                         | Residual — `models.yaml` has 60 tiles, `launcher_manifest.json` 52. `6147a094b` (#8976) and `e5bf6484c` (#8986) narrowed the gap; the single-registry cut (#9412, `1b53a9bf5`) sits on unmerged `readiness/p0-9412-one-tile-registry`. |
| #8861 | Six "parity" features are empty shells                                 | Residual — no commit on `main`; addressed on the same unmerged #9412 branch (`api_only` parity status).                                                                                                                                |
| #8832 | Stale parity docs                                                      | Landed — `c64a7e651` (#9327) and `6147a094b` (#8976).                                                                                                                                                                                  |
| #8833 | Orphaned parity matrix                                                 | Landed — `b1b9c6d6a` (#9987) supersedes the orphaned matrix with the canonical UI policy.                                                                                                                                              |
| #8863 | Six runnable `src/tools` packages in no launcher                       | Landed — `6147a094b` (#8976), `e5bf6484c` (#8986), `5ae322996` (#9420); verify the remaining package set before closing.                                                                                                               |
| #8883 | Video Analyzer placeholder marked `status: ready`; no `maturity` field | Residual — `models.yaml` carries zero `maturity:` keys; the field is introduced on the unmerged #9412 branch.                                                                                                                          |
| #8870 | JaxSim has engine/loader/probe/ADR but no tile                         | Residual — no `jaxsim` entry in `models.yaml` or the README engine table; registered `hidden` on the unmerged #9412 branch.                                                                                                            |

## Cluster C — Duplicate Stacks

Coordinate with the UD-1 seam epic and ADR-0046 before cutting any of these.

| Issue | Finding                                                           | Status on `db4fe88c4`                                                                                                                                                                                                                                                                                                          |
| ----- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| #8864 | Second unmounted FastAPI app; three CORS helpers                  | Residual — six `FastAPI(` constructors (`src/api/server.py`, `src/api/local_server.py`, `calc_backend/app.py`, `model_generation/api/__init__.py`, `motion_pipeline/api.py`, `realtime/ws_pubsub.py`); CORS handled separately in `server.py`, `local_server.py`, `api/config.py`, `typed_settings.py` and `rest_api_core.py`. |
| #8865 | Four C3D readers; GUI upload bypasses the motion pipeline         | Residual on `main` — fix `2dcf42831` exists on unmerged `conductor/issue-8865`.                                                                                                                                                                                                                                                |
| #8867 | ≥9 pose/skeleton classes                                          | Residual — the representation handoff still lists convention consolidation as open.                                                                                                                                                                                                                                            |
| #8866 | 1,600 lines of skeleton extractors nothing imports                | Residual on `main` — deletion `a7c70ddca` exists on unmerged `conductor/issue-8866`.                                                                                                                                                                                                                                           |
| #8868 | Three pub/sub layers; unlaunchable subscriber                     | Residual — not re-verified beyond #8869.                                                                                                                                                                                                                                                                                       |
| #8869 | `REALTIME_TRANSPORT=ws` is a silent no-op                         | Residual — `realtime/api.py` logs "transport not wired in this build" at debug level and falls back to the file transport.                                                                                                                                                                                                     |
| #8875 | Motion-pipeline API advertises formats it rejects                 | Landed — `3c17225ef` (#10018).                                                                                                                                                                                                                                                                                                 |
| #8876 | `SimscapeKinematicsService.get_link_transforms` returns `{}` live | Residual — `pose_interchange/services/simscape.py` still returns an empty dict behind a TODO tied to #6093.                                                                                                                                                                                                                    |
| #8874 | 7 of 11 Rust crates have no install path via extras               | Residual — the `rust` extra in `pyproject.toml` lists three of the eleven `rust_core/` crates.                                                                                                                                                                                                                                 |

## Cluster D — Destroys Work or Lies to the User

| Issue | Finding                                                     | Status on `db4fe88c4`                                                                                                                                                                                                                                |
| ----- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #8880 | Every simulation on the GUI thread                          | Mechanism landed — `6c529a49f` (#9472) adds the shared worker/progress/cancel `async_action` and migrates the first tool; `af0aed1d5` (#9742) migrates Launch Monitor analytics. Coverage of every remaining simulation entry point is not verified. |
| #8881 | Launch Monitor New Project / session removal destroy work   | Landed — `4ec3da337` (#9454) confirm-or-cancel.                                                                                                                                                                                                      |
| #8882 | Pose Studio Save/Load are enabled stubs                     | Landed — `99df48196` (#9462) implements Save/Load with dirty tracking.                                                                                                                                                                               |
| #8887 | ±180° joint limits regardless of engine                     | Residual — `pose_studio/widgets/joint_panel.py` applies `_DEFAULT_DEG_RANGE` everywhere; the comment promises a `JointSlot` override that no code reads.                                                                                             |
| #8888 | "Export for MuJoCo/Drake/Pinocchio" writes one generic URDF | Residual — `model_explorer/gui.py` `export_for_engine` docstring still states the three actions export the same generic URDF.                                                                                                                        |
| #8893 | Golf Environment mock trajectory; synthetic 2-DOF stub      | Residual — `golf_environment/gui.py` still adds "mock trajectories for demo" unlabelled; the cross-engine stub was not re-verified.                                                                                                                  |
| #8884 | Training Controller swallows cancel/pause/resume failures   | Landed — `9623a5662` (#9465).                                                                                                                                                                                                                        |
| #8895 | Docker dialog Close orphans builds                          | Landed — `19a390b78` (#9471), one dirty-close contract.                                                                                                                                                                                              |
| #8896 | Settings Close discards edits                               | Landed — `19a390b78` (#9471).                                                                                                                                                                                                                        |
| #8892 | Web Settings loses edits on navigation                      | Landed — `19a390b78` (#9471).                                                                                                                                                                                                                        |

## Cluster E — Provenance and Units

| Issue | Finding                                                  | Status on `db4fe88c4`                                                                                                                                                                                            |
| ----- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #8820 | Dashboard exports carry no engine/model/run ID           | Landed — `8ef1bec80` (#9995). The industrial-readiness ledger entry U3 still reads `open` and needs its own update under the #9539 contract.                                                                     |
| #8821 | Sidecars only for binary formats                         | Landed with `8ef1bec80` — `export_recording_all_formats` passes provenance to the JSON and CSV writers; `tests/unit/test_dashboard_export_provenance.py` covers CSV headers and embedded JSON. Verify and close. |
| #8822 | `ProvenanceInfo` lacks engine                            | Landed with `8ef1bec80` — `ProvenanceInfo.engine_name` and `run_id`. Verify and close.                                                                                                                           |
| #8886 | mph/ft vs m/s/kg in one chain                            | Residual — not re-verified.                                                                                                                                                                                      |
| #8849 | Uncited sg_optimizer coefficients                        | Landed — `e3402d7c7` (#9988).                                                                                                                                                                                    |
| #8847 | `MethodCitation` surfaced nowhere                        | Landed — `1a5728c8f` (#9994).                                                                                                                                                                                    |
| #8848 | User guide documents a feature that exists only in tests | Landed — `c64a7e651` (#9327).                                                                                                                                                                                    |

## Cluster F — Launcher UX and Performance

| Issue | Finding                                                   | Status on `db4fe88c4`                                                                                   |
| ----- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| #8885 | Hardcoded `setStyleSheet` sites bypass the theme          | Residual — 125 `setStyleSheet` call sites under `src/launchers/` today (the review counted 45).         |
| #8898 | Sync heavy imports in the click handler                   | Landed — `d9f023dab` (#9999).                                                                           |
| #8899 | Workspace layout never persisted                          | Landed — `7e57aaf90` (#10002).                                                                          |
| #8900 | Toasts                                                    | Landed — `308226b0a` (#10004).                                                                          |
| #8901 | Hover-only 18 px buttons                                  | Landed — `eb4a042cb` (#10006).                                                                          |
| #8902 | Wrong shortcut help ×2                                    | Landed — `c7f5e236f` (#10000).                                                                          |
| #8904 | Integrations Health lies                                  | Landed — `1dfaf2af2` (#9997).                                                                           |
| #8905 | Model grid rebuild per keystroke                          | Residual — not re-verified.                                                                             |
| #8907 | Settings in five stores                                   | Residual — not re-verified.                                                                             |
| #8908 | Launcher UX batch                                         | Residual — not re-verified.                                                                             |
| #8891 | Cross-engine web dashboard dt=0 polls forever             | Landed — `d2241041a` (#9774).                                                                           |
| #8922 | Mocap IK re-evaluates `mj_forward` per marker             | Landed — `438bd3282` (#9828).                                                                           |
| #8928 | Pendulum batch accessors                                  | Landed — `cf75b0fb7` (#9831).                                                                           |
| #8929 | Pendulum GUI playback stutters                            | Landed — `5c030174e` (#10061).                                                                          |
| #8940 | Auth dependencies acquire a DB session eagerly            | Landed — `66d80e359` (#9996).                                                                           |
| #8943 | Orchestrator/URDF parse on the event loop; pandas at boot | Landed — `17beca553` (#9727).                                                                           |
| #8930 | Performance                                               | Residual — needs a `benchmarks/` entry before and after (`benchmarks/` holds only `bioptim_parity.py`). |
| #8932 | Performance                                               | Residual — as #8930.                                                                                    |
| #8935 | Performance                                               | Residual — as #8930.                                                                                    |
| #8936 | Performance                                               | Residual — as #8930.                                                                                    |
| #8941 | Performance                                               | Residual — as #8930.                                                                                    |
| #8942 | Performance                                               | Residual — as #8930.                                                                                    |

## Epic Acceptance Against `main`

| Criterion                                     | Status                     | Evidence                                                                                                           |
| --------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| No `maturity: ready` tile opens a placeholder | Not met                    | `models.yaml` has no `maturity` field (#8883); the field and the Video Analyzer downgrade are on the #9412 branch. |
| No simulation runs on the GUI thread          | Partial                    | Shared `async_action` mechanism landed (#8880); per-tool migration is not exhaustively verified.                   |
| No destructive action without confirmation    | Met for the reviewed sites | #8881, #8884, #8892, #8895, #8896 landed with confirm-or-cancel contracts.                                         |
| Every export carries provenance               | Met for dashboard exports  | #8820–#8822 landed in `8ef1bec80`; other export paths were not audited here.                                       |
| One registry                                  | Not met                    | Two tile registries remain on `main`; the cut is on the unmerged #9412 branch.                                     |
| One API factory                               | Not met                    | Six `FastAPI(` constructors (#8864).                                                                               |
| One C3D reader                                | Not met                    | Fix on unmerged `conductor/issue-8865`.                                                                            |
| One pose type                                 | Not met                    | #8867 open; #8866 deletion on unmerged `conductor/issue-8866`.                                                     |

## Ordered Next Waves

1. Land the three ready branches, in this order, rebased on current `main`:
   `readiness/p0-9412-one-tile-registry` (closes #8853, #8861, #8883, #8870
   and the registry acceptance line), `conductor/issue-8865`, then
   `conductor/issue-8866`. Each already carries its own tests.
2. Verify-and-close the rows marked "verify and close" (#8821, #8822, #8863)
   with a comment citing the landed SHA; update readiness ledger entry U3
   for #8820 under the #9539 contract and regenerate its index.
3. Cluster D residuals in one wave: #8887 (read `JointSlot` limits), #8888
   (per-engine export or one honest action), #8893 (label or remove mock
   data). Each is a single-file fix with a unit test.
4. Cluster C residuals after the UD-1 seam ruling: #8864 `create_app(mode=)`
   factory, #8869/#8868 realtime transport, #8876 Simscape transforms, #8874
   extras. Do not start #8867 until #8866 has merged.
5. Cluster F residuals (#8885, #8905, #8907, #8908) plus #8938 from Cluster A
   as the launcher UX batch; then the six performance children, each with a `benchmarks/`
   script committed before the change.

Re-run the first-parent history search and the source reads above before
closing any row; a commit message that names an issue is not, by itself,
evidence that the defect is gone.

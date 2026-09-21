# Industrial Readiness Index

<!-- AUTO-GENERATED — do not edit by hand. -->
<!-- Regenerate with: python3 -m scripts.generate_industrial_readiness_index -->

Generated from [`src/config/industrial_readiness.json`](../../src/config/industrial_readiness.json) (ledger v1.0.0).

Execution index for epic [#9539](https://github.com/D-sorganization/UpstreamDrift/issues/9539), the 2026-09-04
industrial readiness review. The priority children hold the code changes;
this record says which of them landed, what proves it, and which remain
open. It is a software-correctness record only — scientific and human
qualification are recorded separately, through the design-manual
governance pathway.

- **Release status:** 🔴 blocked
- **Reconciled against:** `10caddd219ce213a914fa295661929e4fbf1b686` on 2026-09-09
- **Audit snapshot:** `1f69a51fce997932f04a6ad1dd95bf4d065ba971` (context, not current branch identity)
- **Queue:** 2 merged · 2 open

## Priority Implementation Queue

- **U1** · [#9477](https://github.com/D-sorganization/UpstreamDrift/issues/9477) · P1 · ✅ merged
- **U2** · [#9407](https://github.com/D-sorganization/UpstreamDrift/issues/9407) · P0 · ✅ merged
- **U3** · [#8820](https://github.com/D-sorganization/UpstreamDrift/issues/8820) · P1 · 🔴 open · owner unassigned, depends on [#8822](https://github.com/D-sorganization/UpstreamDrift/issues/8822), [#8821](https://github.com/D-sorganization/UpstreamDrift/issues/8821)
- **U4** · [#9417](https://github.com/D-sorganization/UpstreamDrift/issues/9417) · P3 · 🔴 open · owner unassigned, depends on [#9416](https://github.com/D-sorganization/UpstreamDrift/issues/9416)

### U1 — Pinocchio Ran Free-Fall Only and Cross-Engine 'Total Energy' Had No Mass Term

✅ merged · P1 · [#9477](https://github.com/D-sorganization/UpstreamDrift/issues/9477)

- **Merge SHA:** `ee7792c5873c2265c06ee19f4bdd8ab2098bced9`
- **Implementation:** `src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui_simulation.py` · `src/shared/python/pendulum_simulator/cross_engine_perturbation.py`
- **Tests:** `tests/unit/shared_python/test_cross_engine_perturbation.py` · `tests/unit/test_pinocchio_gui.py`

**Acceptance evidence.** Commanded torque now reaches the Pinocchio stepping loop (`set_commanded_torque` / `_advance_physics` in `gui_simulation.py`), so a user who applies nonzero actuation sees the model driven rather than dropped. Cross-engine total energy resolves through engine-native energy methods, then the mass matrix / inertia / mass fallbacks in `_eval_mass_or_inertia_energy`, so the reported figure is joules rather than a unit-mass velocity proxy.

### U2 — README Install Path Did Not Work on a Clean Machine

✅ merged · P0 · [#9407](https://github.com/D-sorganization/UpstreamDrift/issues/9407)

- **Merge SHA:** `c9b15bb8887c1ed4e6729a91dcf65e45d4860a77` · `a389fcc2b549e17b2d1bad64bcea8bd33ba40783`
- **Implementation:** `install.sh` · `README.md` · `scripts/ci/verify_installation.py`
- **Tests:** `tests/scripts/wave9_scripts_b/test_verify_installation.py` · `tests/unit/test_install_script.py`

**Acceptance evidence.** The documented install path initializes the `vendor/ud-tools` submodule and no longer advertises a Git LFS step that the repository does not use; `scripts/ci/verify_installation.py` executes the documented examples so a clean-machine install is checked rather than asserted. The audit's original missing-init finding must not be re-filed.

### U3 — Engine Dashboard Exports Carry No Engine, Model Path, Run ID, or Timestamp

🔴 open · P1 · [#8820](https://github.com/D-sorganization/UpstreamDrift/issues/8820)

- **Owner:** unassigned
- **Depends on:** [#8822](https://github.com/D-sorganization/UpstreamDrift/issues/8822), [#8821](https://github.com/D-sorganization/UpstreamDrift/issues/8821)
- **Source:** `src/shared/python/dashboard/window.py` · `src/shared/python/dashboard/_recorder_playback.py` · `src/shared/python/data_io/provenance.py` · `src/shared/python/data_io/export.py`
- **Existing tests:** `tests/unit/data_io/test_provenance.py`

**Narrow PR plan.** Confirmed still open against the reconciliation SHA. `UnifiedDashboardWindow.export_data` saves under the default name `swing_data` and dumps `recorder.get_data_dict()`, which stamps only `model_name` and `num_frames`; `_flatten_dict_for_csv` then keeps only per-frame arrays, so the CSV carries no identifying column at all and MuJoCo, Drake and Pinocchio exports are byte-level indistinguishable. Narrow PR order: (1) add the engine field to `ProvenanceInfo` and make `model_path` mandatory at the dashboard call site (#8822); (2) emit a sidecar plus header rows for JSON and CSV using the already-present, currently unused `add_provenance_to_csv` (#8821); (3) stamp engine, model path and hash, timestamp and run ID into the dashboard export dict and assert the round trip from each of the three engine dashboards (#8820). Do not widen the export schema without a reimport test.

### U4 — Ship Installable Artifacts: Wheel, Image, Desktop Bundles, SBOM

🔴 open · P3 · [#9417](https://github.com/D-sorganization/UpstreamDrift/issues/9417)

- **Owner:** unassigned
- **Depends on:** [#9416](https://github.com/D-sorganization/UpstreamDrift/issues/9416)
- **Source:** `deploy` · `installer/windows/build_installer.py`
- **Existing tests:** —

**Narrow PR plan.** Confirmed still open against the reconciliation SHA: `deploy/` is present but empty, and `release.yml` publishes a wheel without the container image, Tauri bundles or per-artifact SBOM the installation docs imply. The companion-manifest dependency (#9416) is merged at 072db089188edc96f365e3a11e3e43bd27efd11d, so this is dependency-ready. Narrow PR order: decide the GA artifact set; add image build/push, bundle upload, SBOM and attestation to the release workflow; resolve the Windows MSI decision against `installer/windows/build_installer.py`; either populate `deploy/` with the API compose/helm or delete it; add a clean-machine smoke install per artifact. Ship the artifact-set decision first — the later slices depend on it.

## Acceptance Criteria

### ✅ Met — Each Priority Child Is Reconciled Against Current Main and Has a Narrow PR Plan.

- **Blockers:** —

All four queue entries were re-read against the reconciliation SHA rather than the audit snapshot: U1 and U2 are merged with the SHAs recorded here, and U3 and U4 were re-confirmed open at the named source locations and carry an ordered PR plan.

### ✅ Met — Every Claimed Completed Capability Links Implementation SHA, Executed Tests and User-Visible Acceptance Evidence.

- **Blockers:** —

The registry contract refuses a `merged` entry that lacks a 40-character merge SHA, a test path, or acceptance evidence, and refuses any implementation or test path that does not exist in the tree.

### 🟡 Partial — Required Release Profiles Pass Installed-Product Journeys; Unsupported Profiles Are Explicit.

- **Blockers:** [#9417](https://github.com/D-sorganization/UpstreamDrift/issues/9417)

Engine support is explicit in `docs/operations/tier-policy.md` (MuJoCo core; Drake and Pinocchio extended; OpenSim and MyoSuite experimental behind a warning), and the source install journey is checked by `scripts/ci/verify_installation.py`. No built artifact other than the wheel has a clean-machine journey, which is exactly the U4 scope.

### ✅ Met — Numerical/Scientific Validity, Software Correctness and Human Qualification Remain Separately Recorded.

- **Blockers:** —

This ledger records software correctness only. Scientific qualification stays in the design-manual governance pathway (`scripts/config/design_manual_governance.json`, `python3 -m scripts.check_design_manual_governance`); a passing entry here is not scientific or human approval, and no registered experiment may be rerun or promoted to satisfy it.

### ✅ Met — Outstanding Blockers Have a Named Owner/Dependency and Do Not Masquerade as Green Release Status.

- **Blockers:** [#8820](https://github.com/D-sorganization/UpstreamDrift/issues/8820), [#9417](https://github.com/D-sorganization/UpstreamDrift/issues/9417)

`release_status` is `blocked` and the contract forbids `ready` while any queue entry is open or any acceptance criterion is unmet. Every open entry must carry an owner and a plan, must carry a dependency when the owner is unassigned, and must appear in at least one acceptance blocker list, so an open item cannot be dropped from the summary.

### 🟡 Partial — Release Evidence Includes Recovery/Rollback Instructions and Preserves User Data.

- **Blockers:** [#9417](https://github.com/D-sorganization/UpstreamDrift/issues/9417)

`docs/operations/release-runbook.md` carries a Rollback section that pins downstream consumers to the previous tag, forbids overwriting or rerunning immutable release assets, and requires an incident issue before a patch release. It covers the wheel; the artifacts U4 adds have no rollback step yet.

## Keeping This Record Honest

`src/config/industrial_readiness_loader.py` refuses a ledger that claims
more than the tree supports. A merged entry must carry a 40-character
merge SHA, at least one test path, and user-visible acceptance evidence;
an open entry must carry no merge SHA, an owner, a plan, and a dependency
when no owner is named. Every referenced path must exist. Every open
issue must appear in an acceptance blocker list, and the release status
cannot read `ready` while any entry is open or any criterion is unmet.

An issue closure, a mock-only success, a changed golden file or a raised
tolerance is not evidence of correctness and must not be recorded here.

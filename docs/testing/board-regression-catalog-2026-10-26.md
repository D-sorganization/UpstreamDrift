# Board Regression Catalog: 2026-10-26

This catalog supplies the exact regression locations for the six resolved rows
in [deferred tracker #13882](https://github.com/D-sorganization/UpstreamDrift/issues/13882)
and [test-hardening epic #13864](https://github.com/D-sorganization/UpstreamDrift/issues/13864).
The epic's original `test_ud_NN` descriptions are proposed test names, not the
names of the tests that ultimately merged. Use the paths and pytest nodes below
when reproducing those regressions or reconciling the remaining listing checkbox.

## Resolved Rows and Existing Tests

Paths and nodes were checked against main `02450ff99d8c917f86601aea2049964e611057e7`.
The linked fix PRs are merged ancestors of that revision. These are existing
software regressions; this catalog introduces no new scientific acceptance rule.

| Finding                                      | Existing Regression                                                                                                                                                                                                                                                                                  | Merged Fix and Merge Commit                                                                                       |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| UD-08: model-specific lift coefficient cap   | [test_flight_models.py](../../tests/unit/test_flight_models.py): `TestMacDonaldHanzelyModel::test_ud_08_spin_ratio_lift_coefficient_respects_model_cl`                                                                                                                                               | [#13958](https://github.com/D-sorganization/UpstreamDrift/pull/13958), `eb8b71400ed95d88bce26e8757e08f00710ccd6e` |
| UD-12: SimTK quaternion documentation        | [test_opensim_coord_map.py](../../tests/engines/opensim/test_opensim_coord_map.py): `test_simtk_quaternion_docs_accuracy`                                                                                                                                                                            | [#13954](https://github.com/D-sorganization/UpstreamDrift/pull/13954), `f838b4e2e92764716435cc85b2a21394cb9cfec1` |
| UD-20: cost helper docstrings                | [test_cost.py](../../tests/unit/motion_matching/test_cost.py): `test_cost_term_helpers_have_docstrings`                                                                                                                                                                                              | [#13948](https://github.com/D-sorganization/UpstreamDrift/pull/13948), `b5182571427a60699dc7b79ef7fa2774c904cbce` |
| UD-23: one-based impact index                | [test_error_timecourse.py](../../tests/unit/engines/pinocchio/test_error_timecourse.py): `test_error_timecourse_impact_idx_is_one_based`, `test_error_timecourse_impact_idx_first_frame`, `test_error_timecourse_impact_idx_last_frame`, `test_error_timecourse_impact_idx_missing_defaults_to_last` | [#13950](https://github.com/D-sorganization/UpstreamDrift/pull/13950), `2f369c5a740bedba630f2f9d27fb62ba935d4696` |
| UD-30: balanced support and transverse force | [test_grip_contact_model.py](../../tests/unit/test_grip_contact_model.py): `TestStaticEquilibrium` covers balanced support, excess support, transverse force and invalid inputs                                                                                                                      | [#13960](https://github.com/D-sorganization/UpstreamDrift/pull/13960), `15e55478f224e074b25310f79ad955dae77b5b18` |
| UD-37: bounded MuJoCo advance completion     | [test_simulate_contract_mujoco.py](../../tests/parity/test_simulate_contract_mujoco.py): `test_advance_to_returns_false_when_substep_budget_exhausted`, `test_advance_to_returns_true_when_budget_sufficient`                                                                                        | [#13963](https://github.com/D-sorganization/UpstreamDrift/pull/13963), `5e855a22c29297f2373957c7b01dc7e8acd44f1f` |

## Focused Reproduction

Run from the repository root with the supported Python environment and the exact
`vendor/ud-tools` gitlink initialized. Follow [the testing guide](testing-guide.md)
for environment setup. In PowerShell, replace `python3` with `py -3` and place
each command on one line.

```bash
python3 -m pytest -q -n 0 --no-cov tests/unit/test_flight_models.py::TestMacDonaldHanzelyModel::test_ud_08_spin_ratio_lift_coefficient_respects_model_cl tests/engines/opensim/test_opensim_coord_map.py::test_simtk_quaternion_docs_accuracy tests/unit/motion_matching/test_cost.py::test_cost_term_helpers_have_docstrings tests/unit/engines/pinocchio/test_error_timecourse.py tests/unit/test_grip_contact_model.py::TestStaticEquilibrium tests/parity/test_simulate_contract_mujoco.py::test_advance_to_returns_false_when_substep_budget_exhausted tests/parity/test_simulate_contract_mujoco.py::test_advance_to_returns_true_when_budget_sufficient
```

Record the actual passed, skipped and failed counts. Optional-engine skips do
not establish native-engine acceptance. The SimTK row verifies documentation;
the index row exercises the visualization's indexing contract. Neither is a new
cross-engine parity qualification. Earlier red-first evidence belongs to the
linked fix PRs; rerunning current tests establishes their present result only.
If an installed optional engine rejects the host's headless backend during
collection, record that error and run the other rows separately. In particular,
Windows MuJoCo can reject `MUJOCO_GL=egl`; a collection error is not a skip or a
passing UD-37 result.

## Remaining Tracker Boundaries

This listing covers UD-08, UD-12, UD-20, UD-23, UD-30 and UD-37 only. Cataloged
findings UD-13, UD-14, UD-26, UD-33, UD-36 and UD-39 retain their child issues;
UD-34 and UD-35 retain their recorded non-reproduction evidence. Listing these
six regressions does not close those children, finish all of #13864, or authorize
any issue closure. Reconcile tracker status against merged evidence separately.

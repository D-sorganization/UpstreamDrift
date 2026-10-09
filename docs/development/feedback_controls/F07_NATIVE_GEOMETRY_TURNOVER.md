# Native OpenSim Geometry Turnover

## Scope and Current Evidence

Child #11903, parent F07 #11791 and F08 #11792. Source branch
`feat/feedback-native-markers-11903`; canonical calculation authority is
`manuals/upstreamdrift/chapters/26-native-opensim-geometry.qmd`. This implements
native geometric callbacks and rejects metadata-only physical placeholders.
It does not complete the generic grip/ground IK adapter or dynamics integration.

TDD first demonstrated that five placeholder operations did not raise. New
native body/offset-frame fixtures verify actual pose and station values,
repeatability, source identity and independent-coordinate restrictions. A real
two-joint native coupler exposed an unchecked dependent-coordinate range; the
new all-coordinate check rejects that case. The achieved-coordinate rejection
test uses a proxy assembly perturbation and does not claim native solver failure.
Six bounded-solver tests first failed for missing API; opt-in finite-source-bound
TRF now retains unreachable residuals and rejects invalid seeds, with default LM
preserved. Native and diagnostic tests are run serially.
An additional RED test exposed premature cost-only termination from an active
bound; the bounded trajectory path now uses the existing pose-TRF stopping policy.

The pinned public model was evaluated without changing its muscles, locks,
constraints, reserves or anatomical parameters. Three original capture samples
produced aggregate 37.704 mm RMS and final-sample 65.192 mm RMS; pelvis rotation
reached its source bound. Native station/transform consistency was within
1.1e-14 m. Earlier unbounded LM failed at the source range and its failure receipt
is retained separately. The first sample is the offset-calibration gauge, so its
near-zero residual is constructed. Details stay in the private capture clone;
public source assets and private observations are not committed here.
The resulting local offset radii are 1.107–1.249 m, which exposes registration
being absorbed into the gauge and rejects any anatomical pelvis-fit interpretation.

## Reproduction and Remaining Work

Use the serial native test and diagnostic commands in chapter26. Windows native
OpenSim 4.6 and ezc3d 1.6.3 DLLs failed when loaded together in either order.
The driver reuses the existing frozen loader in a separate Python process and
the existing TRC exchange; it checks quantization and restores the original
frozen clock. No capture parser or private schema was added. The native runtime
and source-model hashes are recorded in private execution receipts.

Next: obtain source-backed capture/model fixed frame registration from declared
training anchors and freeze it before holdout; qualify actual body-local marker
attachments and anthropometry; retain unknown nonpelvic correspondence and
trial ancestry as blockers. A bound hit alone cannot distinguish registration
error from anatomical infeasibility. No offset refitting on holdout is allowed.
Then integrate qualified geometry with muscle/contact state preparation and
independent excitation replay. Passive loads, 34 rigid tendons, locked wrists,
hand/grip coverage, reserves/couplers and absent source contact remain visible.
Native continuous motion, derivatives, uncertainty and full-horizon scientific
acceptance are not inferred from sampled pose fitting.

The calculation registry stays empty and release blocked. Generated manual
artifacts, semantic parity and human publication approval remain outstanding.

## Local Validation

- 53 scoped native geometry, diagnostic, document-IK and governance tests passed
  in the actual OpenSim 4.6/Python 3.12 runtime; the pre-existing full-stride
  integration test was excluded from this bounded execution.
- Ruff lint/format and five changed source-file mypy checks passed after two
  new typing errors were corrected. Governance reports nine QMD sources,
  zero approved calculations and blocked release.
- Normal commit and push hooks passed without bypass, including mypy, Bandit
  and the repository's pre-push unit-test selection.
- The central `pre_pr.py --base-ref origin/main` broad test mapper selected
  45 OpenSim test modules: 244 passed, 57 skipped, 19 deselected and five failed
  in default Python 3.13. Two failures are the deliberate OS-0 runtime gates
  (OpenSim is installed in the separate native Python 3.12 environment).
  Three reproduce when only unchanged wrapper test modules are collected:
  their module-level OpenSim mock contaminates unavailable-engine tests, and
  their legacy string-loader test patches the public path loader instead of
  the existing `_load_from_path_impl`. Those files and production wrapper are
  unchanged from the branch base. This broad gate is not reported as green.
  Semgrep/import policy and policy/fragment gates passed.
  All three OS-0 qualification-module tests subsequently passed in the actual
  native Python 3.12 runtime; this does not erase the broad default-runtime failure.
- The incomplete fleet inbox read retains its malformed-history/page-limit
  warnings; direct ownership coordination covers this branch and shared-solver
  extension. No absence-of-peer claim is inferred from that read.

## CI Remediation

CI identified three avoidable attribute chains and missing tracking references
for deliberate unsupported physical methods; these are corrected without
changing behavior. The existing eight-parameter trajectory API gains one
optional coordinate_bounds keyword. A focused architecture-budget exception
owned by codex/#11903 expires2026-11-09, preserving caller compatibility pending
coherent solver-options consolidation. It exempts no native range or scientific
gate. Existing private receipts retain their exact earlier provider hashes;
they are not silently relabeled after this structural refactor.

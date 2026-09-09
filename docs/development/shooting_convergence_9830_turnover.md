# Independent Shooting Accuracy: #9830

## Ownership and Scope

- Follow-up to #9756 under #9762; integration relevance to impact program #9700.
- Branch: fix/9830-independent-shooting-convergence. Codex lease session
  impact-acoustics-01a07d8a-shooting9830 expires 2026-09-09T04:11:20Z.
- Reproduction source: protected main 9f54c5e0b79d9ca0e81aebf52f90f07ffac832c0.
  Worktree subsequently fast-forwards to ff0effa5a, retaining identical
  casadi_backend, model_provider, \_swing_models and motion_pipeline/model_bridge.
- The original study preserves the production shooting solver and threshold.
  The later adaptive-reference implementation and corrected interpretation are
  documented below. Results concern a synthetic model, not experimental
  validation or a certificate of a realizable golfer swing.

## Reproduced Failure

Optional-stack run 34292252213/job 102281274047 fails
test_multiple_shooting_satisfies_its_own_dynamics at 0.5190812793829043 rad
against its existing 0.5 rad bound. The own-grid test passes first. The source
predates #9826 (af6923bc7 / #9768); no retry or threshold relaxation is justified.

An isolated Linux run reproduces that exact failure in 110.59 s, retaining the
600-second CI timeout and one finite-difference deprecation warning. Runtime:
Python 3.11.15, CasADi 3.8.0, NumPy 2.4.6, SciPy 1.17.1, pytest 9.1.1,
pytest-timeout 2.4.0, one BLAS/OMP/MKL thread. CI uses Python 3.11.16; do not
describe the whole environment as byte-identical. The numerical dependency
versions match the CI installation log. Exact Tools gitlink eab74a901 is
initialized; no native binding is mocked to obtain this result.

## Exploratory Convergence Evidence

Same default golfer/club, anthropometric inertials, six nodes over 0.6 s,
800-iteration cap, torque/joint limits and sine initial guess as the failing
test. Re-optimize independently at 8, 16 and 32 shooting substeps; each uses
the original initial guess, not a substituted saved solution. For each fixed
candidate, preserve its x and interval torques and compare local interval
rollouts at 8/16/32/64 RK4 substeps with an independent adaptive integration.

Adaptive RHS is [v, forward_dynamics(q,v,tau_k)], holding the original interval
torque constant and restarting at each original node. DOP853 integrates to
the actual interval endpoint, with max_step=0.12/32 s. Compare rtol/atol pairs
(1e-8,1e-10) and (1e-10,1e-12). These are numerical tolerances for coordinates
in rad and rad/s, not measurement uncertainty. The two adaptive settings agree
in the reported maxima to roughly 3e-9 rad and 4e-7 rad/s or better. They share
the same dynamics RHS, so this checks integration, not an independent dynamics
implementation. Local interval defects are not whole-trajectory error bounds.

| Shooting Substeps | Own-Grid Position Defect (rad) | Own-Grid Velocity Defect (rad/s) | Tight Adaptive Position Defect (rad) | Tight Adaptive Velocity Defect (rad/s) |
| ----------------- | ------------------------------ | -------------------------------- | ------------------------------------ | -------------------------------------- |
| 8                 | 4.6546e-11                     | 2.3839e-9                        | 0.4812025640                         | 25.8570082604                          |
| 16                | 2.0514e-12                     | 9.4161e-11                       | 0.1469211437                         | 6.6214611260                           |
| 32                | 3.8677e-10                     | 1.1272e-8                        | 0.0064861939                         | 0.3355566480                           |

For the original eight-substep candidate, the 16/32/64-substep RK4 position
defects are 0.5190812794, 0.4804333166 and 0.4812953138 rad. Thus the nominal
16-substep reference is itself under-resolved. Merely switching references can
pass the old position threshold while retaining a large velocity mismatch.
Re-optimization improves the observed defect in these three runs, but a
nonconvex optimizer need not choose the same local branch or improve
monotonically in general. The 16/32-substep solves converge in 199/317
iterations with objectives -1.3351813163/-1.3174353216; no optimality or physical
realizability is inferred.

## Preserved Local Evidence

Native environment: /home/dieterolson/.cache/codex-impact/shooting9830/venv.
Evidence directory: /home/dieterolson/.cache/codex-impact/shooting9830/evidence.
Full per-interval metrics are in fixed-candidate-study.json (8),
fixed-candidate-study-16.json and fixed-candidate-study-32.json.

- candidate-eight.npz SHA256:
  e2a2bc71dc2d63d87f94c12b5db4d697ae0a7fbc5349214525dd81aa0c3ef157.
- candidate-16.npz SHA256:
  ad04fa01c75a36ac43eac404f47445559300eb8e43d155d5fa45afe4802e7f6d.
- candidate-32.npz SHA256:
  8c0986d83ce46e048582f8da4e959fb376638eac3e66344ebcd3f38cf02563a0.

Exploratory helper: C:/Users/diete/AppData/Local/Temp/impact-9830-study.py.
It reuses the benchmark torque-limit helper and the production symbolic RHS.
The initial run saved the successful eight-substep candidate before Linux Git
failed to interpret the Windows worktree path. Reuse that exact saved candidate
for reference refinements; do not invent its lost iteration/objective metadata
(both are null). Native Windows Git supplies the verified source SHA to later
runs. This helper is not a qualified production implementation. The SciPy
versioned web reader refused its URL; no full-page review or bypass is claimed.

## Adaptive Reference Implementation

The private `_swing_reference.py` endpoint operation now reuses SciPy DOP853
with analytic oscillator, damped-mode and constant-acceleration controls.
`dynamics_defect` reuses the existing symbolic forward dynamics and inverse
dynamics. Its default reference requires two endpoint integrations to agree;
explicit `n_substeps` retains the fixed RK4 discrete-map diagnostic. Position
and velocity defects retain their units and separate summaries. The legacy
four-field dictionary remains compatible; `reference_resolution` additionally
retains endpoint values, component agreement, settings and RHS counts.

Endpoint agreement uses `atol + rtol*max(abs(coarse), abs(fine))` separately
for each component. The initial-state magnitude is deliberately excluded:
a RED control showed that a large initial displacement could otherwise hide
a small near-zero endpoint discrepancy. Default endpoint budgets are rtol
1e-8, position atol 1e-10 rad and velocity atol 1e-10 rad/s. Local solver
tolerances are 10 and 100 times tighter, with step caps interval/32 and
interval/64. These are numerical settings, not experimental uncertainty or
a rigorous global error bound. Unresolved refinement, nonfinite values,
malformed endpoints and exceeded RHS-call budgets refuse qualification.

TDD and live evidence on 2026-09-09:

- Missing endpoint module: RED collection failure before implementation.
- Initial analytic/refusal suite: 52 passed. The initial-magnitude masking
  control then failed before correcting the endpoint scale.
- Backend default routing and strict-domain controls: 13 failed, 53 passed
  before integration; all 66 passed afterward in 6.53 s.
- First complete live run: 1 failed, 7 passed, 3 native-dependency skips in
  99.00 s. Endpoint agreement correctly refused the coarse reference in the
  original swing. The endpoint budget was not relaxed. A separate RED test
  required tighter local tolerances; one budget-separation control passed.
- With local solver tolerances tightened, the combined reference and live
  suite passed: 75 passed, 3 Pinocchio-dependent skips in 101.48 s. The
  original six-node/eight-substep case and 0.5 rad historical ceiling remain;
  assertions now also require the intentionally large velocity discrepancy
  and retained reference-resolution evidence. Seven existing deprecation
  warnings remain.
- Installed Pinocchio 4.1.0 without changing NumPy 2.4.6, SciPy 1.17.1 or
  CasADi 3.8.0. All three previously skipped native RNEA/dynamics comparisons
  passed in 2.78 s (eight unrelated cases deselected). The isolated Windows
  mocked-SDK/degradation and backend-registry suite also passed: 13 tests in
  15.93 s, with nine existing deprecation warnings. Bioptim was not installed
  at that checkpoint; the later import investigation is recorded below.
- Root Ruff 0.15.17 lint/format checks passed (6,769 files); mypy 1.13.0
  passed on both changed source modules. Manual governance passes its
  structural check but still reports `blocked-inventory-required`, with zero
  registered calculations. No scientific/manual release approval is claimed.

The parity narrative now labels its tables as historical fixed-reference
results. It removes unsupported physical-realizability, controlled mesh
convergence, convexity and equivalent-optimum claims. The old numbers are
preserved; the tables have not been regenerated.

## Remaining Acceptance and Delivery

1. Native and degradation checks are complete as recorded above. A combined
   Windows collection of the new reference tests and existing mocked-SDK tests
   also passed all 80 cases in 8.79 s (nine inherited warnings), checking the
   optional-module contamination boundary. Bioptim remains unqualified under
   the concrete compatibility failure in #9842, not a locally passed test.
2. Implementation 17211e201 and SPEC placement ccd64c8b4 are integrated with
   main 403292ca3 in 7c0a84b8865a356cc1d2508397b460706f346b81. The merge
   changes no optimization source or tests relative to the validated code.
   Root Ruff still passes (6,769 files); the fleet SPEC hook passes with exactly
   one new row. The script is absent in this checkout and was run from the
   authoritative Repository_Management sibling. All normal commit/push hooks,
   including mypy, Bandit and the configured unit gate, passed. Remote SHA was
   verified. PR [#9841](https://github.com/D-sorganization/UpstreamDrift/pull/9841)
   is open; protected CI/review are pending. An initial phantom-guard run was
   cancelled before any runner/step; its duplicate is queued. No source failure
   or successful guard execution is inferred from that cancellation.
3. Numerical reference resolution does not qualify the coarse swing for impact
   use. Application-specific state budgets, whole-trajectory convergence,
   physical calibration and acoustic validation remain separate program work.

## CI Type Contract and Bioptim Consumer Investigation

Code-quality job 102306477551 at 05b780f7e fails because `_checked_rhs` returns
a validated array callback but advertises the raw `object` return type. This
reproduces locally only when the CI SciPy stubs are included: mypy 1.20.1,
NumPy 2.2.6, SciPy 1.15.3 and scipy-stubs 1.17.1.4. The un-stubbed check
passing did not cover that contract. Narrowing the callback return annotation
to `np.ndarray` gives a passing two-module type check; 67 reference/routing
tests pass in 10.01 s afterward. No numerical expression or tolerance changes.

The dedicated environment then installed Bioptim 3.4.0 from exact revision
fdafe4d9e50edb3f92a14f1980b08c3f81434545, Matplotlib 3.11.1 and the missing
core import dependencies (structlog 26.1.0, h5py 3.16.0, simpleeval 1.0.7,
pandas 3.0.5 and Jinja2 3.1.6). Initial collection failures identify absent
core dependencies, not reference-solver failures. With those present, the
two real OCP test modules report 8 failed, 5 passed in 56.16 s: pinned
`bioptim/limits/constraints.py` imports `MX_eye` from CasADi 3.8.0, which
does not export it. Installed direct_url metadata verifies the exact source.
The failures precede optimization and defect evaluation. Follow-up
[#9842](https://github.com/D-sorganization/UpstreamDrift/issues/9842) owns
real-process compatibility, pin/probe corrections and consumer validation;
do not claim these OCP paths passed or alter historical study dependencies.

## Bioptim Compatibility and Selected Runtime (#9842)

The real-process matrix test first failed on `MX_eye` (24.22 s). Supplying
MX/SX aliases exposed the additional missing `DM_eye`; an AST inventory of
Bioptim imports confirmed all three. The compatibility helper now installs
only absent aliases from exact CasADi class factories, without replacing
native exports. The absent-distribution fixture explicitly simulates absence
so it also works on hosts with a real SDK. Values, types, zero-size shapes,
MX/SX Jacobians and repeated imports are exercised in a subprocess.

With the aliases, CasADi 3.8 reports 16 passed and one failed in 205.31 s
across factory, contamination and swing/tracking controls. The remaining
eight-node, 1 s RK4 case fails solver success with its original 500-iteration
budget. A separate 1,000-iteration diagnostic also fails, stopping at 635
iterations; more iterations did not resolve it. Raw status is 1 and does not
identify a more specific IPOPT failure cause. The failed result's fallback
warm-start `x` was not analyzed as a final candidate. Diagnostic provenance is
archived in `/home/dieterolson/.cache/codex-impact/bioptim9842/budget1000.json`,
including source hashes for the then-uncommitted compatibility patch.

A separate environment preserves the same 51 packages except CasADi 3.6.7.
All 13 unchanged swing/tracking cases pass in 170.58 s; the native-factory
control passes in 4.41 s. The original 3.8 study environment remains intact.
The Bioptim extra therefore pins CasADi 3.6.7, while the general
`optimal-control` range remains unchanged. Dependency-contract TDD produced
two failures/two passes before the pin and four passes afterward (10.39 s).
No integration substeps, solver budgets or regression ceilings were changed.
The final combined real swing/tracking, factory, module-isolation and dependency
suite passes all 21 cases in 162.06 s (one inherited deprecation warning).
Python 3.11 Linux is the tested numerical scope; other available wheels are
not equivalent validation. The parity guide records that limit and the
unsuccessful 3.8 case. Source docstrings no longer claim convexity guarantees
or a general collocation performance advantage.

The normal push of callback fix 7369ccc2d was rejected by a shared hook cache
combining mypy 1.13 with NumPy 2.5.3 syntax. No hook was skipped and the shared
cache was not modified. A task-isolated PRE_COMMIT_HOME installs the same
configured hooks; its actual mypy hook passes all three changed source
modules. The separate CI-equivalent SciPy-stub type check also passes.

CI's unit gate reports one failure, 14,448 passes and 671 skips: the generated
shared-tools divergence inventory lacks the new private reference module.
The generator preview adds that sole classification entry and preserves the
Tools pin; generated authorship and byte metadata also refresh. Canonical
regeneration is now complete: all ten inventory tests pass in 16.78 s, and
the classification projection differs only by the new reference module.
Commit 658fe3c8a contains runtime/inventory qualification; c2a8d5b6e integrates
main 438bd3282. Only SPEC required manual conflict resolution, retaining both
incoming entries and the reference section. Optimization source/tests,
dependency metadata, optional workflow and generated inventory are unchanged
by this merge. Inventory freshness, pinned root Ruff (6,772 files) and the
SPEC hook pass afterward. PR #9841 remains open; publishing through normal
push hooks and protected CI/review remain pending.

Repository-pinned Ruff 0.15.17 passes root lint and format (6,771 files).
An initial check accidentally used the shared environment's newer 0.16.4;
its unrelated new-rule/Markdown-format findings are not the repository gate,
and no unrelated files were changed. Manual governance and the SPEC hook
pass; manual release remains `blocked-inventory-required`.

## Related Delivery

Tools autonomous decay 58f33e403 and turnover 2d290079b are published through
normal hooks (731 Linux golf/API tests). T2 #5082 has merged as 80d580d57;
its golf source/tests match both reviewed e47fde4e and prior 476eaa98.
Inventory #5103 has advanced to 02b53e2d8 and awaits current CI; its AST
correction is still needed for the new decay calculation. Signal PR #5106 at
c8f3b4d1 still reports a failed private-consumer lane with other checks pending.
Subsequently #5106 merged as 287767dfa60567de136fbadc0da28c7e1ca7edf3
at 2026-09-09T01:41:52Z from c8f3b4d1; merge is verified, not all-checks-green
evidence or equivalence to the older locally validated signal worktree.
At the subsequent #5103 observation, private-consumer job 102292157026 again
fails its repository lookup with Not Found. Python 3.11/3.12 aggregate checks
fail because their rate-of-closure shards were cancelled after about 90 minutes;
the 3.11 log reaches 99% and then has no further test completion before
cancellation. These are not classifier failures or passing suites; source
diagnosis of the unfinished GUI tests and private access remains separate.
Scheduled/completed-ID comparison isolates the unfinished
`test_pyqt_variation_rendered_interactions.py::test_rendered_controls_at_100_and_150_percent_dpi`;
follow-up Tools #5114 records evidence and bounded diagnostic acceptance.
UpstreamDrift #9826 merged as a410ae7059883d7f27f5fb12405b61859267457c
at 2026-09-09T01:12:29Z, from reviewed head fced8c0d6. This was verified via
GitHub; remaining queued auxiliary jobs are not claimed as passed. The
impact/acoustic program remains active; physical calibration and blinded
listener evidence remain open.

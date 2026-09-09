# Independent Shooting Accuracy: #9830

## Ownership and Scope

- Follow-up to #9756 under #9762; integration relevance to impact program #9700.
- Branch: fix/9830-independent-shooting-convergence. Codex lease session
  impact-acoustics-01a07d8a-shooting9830 expires 2026-09-09T02:13:33Z.
- Reproduction source: protected main 9f54c5e0b79d9ca0e81aebf52f90f07ffac832c0.
  Worktree subsequently fast-forwards to ff0effa5a, retaining identical
  casadi_backend, model_provider, \_swing_models and motion_pipeline/model_bridge.
- No production solver, test threshold or scientific claim has been changed.
  Results below are exploratory numerical diagnostics of a synthetic model,
  not experimental validation or a certificate of a realizable golfer swing.

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

## Next Implementation and Acceptance

1. Add failing tests for an independently controlled reference integration,
   including analytic stiff/oscillatory models, endpoint coverage, component
   units, nonfinite inputs/output, solver failure and unresolved refinement.
   Keep a deliberate fixed-grid path for checking shooting constraints.
2. Reuse the existing symbolic dynamics and defect record. The pendulum ODE
   helper deliberately excludes the final endpoint from its output grid; it is
   unsuitable unchanged for this endpoint comparison. The sidekick helper
   expects symbolic expressions; do not duplicate a dynamics model to use it.
3. Separate NLP convergence, discrete feasibility, reference resolution and
   continuous-ODE discrepancy in contracts and documentation. A fixed larger
   substep count alone cannot guarantee an accurate reference for every input.
   Do not report own-grid residuals as physical validation or silently mix
   position and velocity acceptance scales.
4. Preserve the failing candidate and original threshold history. Require a
   converged reference and scientifically stated error budgets; do not increase
   0.5 simply to obtain green CI, skip the case or retry it as flaky.
5. Correct the parity narrative where it promotes OCP outputs to realizable
   swings despite unqualified defects. Then run live integration/optimization
   tests, required lint/type/manual checks and protected CI before merging.

## Related Delivery

Tools autonomous decay 58f33e403 and turnover 2d290079b are published through
normal hooks (731 Linux golf/API tests). T2 #5082 has merged as 80d580d57;
its golf source/tests match both reviewed e47fde4e and prior 476eaa98.
Inventory #5103 has advanced to 02b53e2d8 and awaits current CI; its AST
correction is still needed for the new decay calculation. Signal PR #5106 at
c8f3b4d1 still reports a failed private-consumer lane with other checks pending.
UpstreamDrift #9826 publishes c7a88299d and is integrating current main to
remove the shallow-diff false deletion failure. The impact/acoustic program
remains active; physical calibration and blinded listener evidence remain open.

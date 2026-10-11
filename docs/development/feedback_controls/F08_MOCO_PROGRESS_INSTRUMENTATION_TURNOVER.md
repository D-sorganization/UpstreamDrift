# F08 Moco Progress Instrumentation Turnover

## Scope

Issue #11898 adds measurement-only visibility to the maintained native Moco
runner. It preserves the existing native_moco_progress.json latest-stage
fields and adds an ordered events journal to each atomically replaced
snapshot. Events remain bound to request and source SHA-256 values and retain
the build callback's existing eight-stage order.

The runner records monotonic, wall and CPU clocks, plus the current-input
audit duration. The existing elapsed wall/CPU fields retain their established
origin after \_require_current_inputs and the native body audit. The separate
monotonic clock supports comparison with the outer supervisor's child-start
clock; it does not reinterpret the existing elapsed fields.

At the optimizing event, after the existing set_parallel(0) call and
immediately before study.solve(), the runner records actual MocoCasADiSolver
getter outcomes for solver, Hessian, sparsity, finite-difference scheme,
transcription, dynamics mode, mesh intervals, verbosity, parallelism, iteration/tolerance and
IPOPT print properties. Returned scalars are distinguished from missing,
error, nonfinite and non-scalar outcomes. No missing value is filled from a
constructor default.

## Limits

The observer does not change request bytes, model/source, mesh, bounds,
objective, optimizer settings, dynamics, integration accuracy, or solver-call
count. The optimizing stage proves entry into the study solve API only. It
does not prove CasADi/IPOPT preprocessing finished, an IPOPT iteration began,
or that the solve returned. Timeouts remain measurements or unknowns. Expected RuntimeError/ValueError getter failures are retained as error receipts; unexpected exceptions propagate so the observer does not hide programming failures. None of these outcomes is a convergence, feasibility, or qualification claim.
All 17 required production model/variant rows across six ecosystems remain
unqualified.

## Validation

The focused unit tests use a fake study and solver to verify the unchanged
stage order, the post-set_parallel(0) getter timing, request/source identity,
atomic replacement, and durable optimizing state when study.solve() raises.
Additional cases preserve getter missing/error/nonfinite/non-scalar results, verify unexpected descriptor and getter exceptions propagate, and confirm failed serialization leaves the prior complete snapshot intact.
No OpenSim SDK import, model construction, native solve, or native execution was
performed for this instrumentation change.

The validation worktree used the repository-pinned Tools submodule commit
6b6e4be55e7086a33db43b36d2632765be619bde through a junction to an existing
clean donor checkout. The donor is not modified by this branch; keep it
available until validation using this worktree is complete, then remove only
the owned junction. Do not remove or modify the donor as part of this change.

# F03 Synthetic Collocation Spike Turnover (#11787)

F03 work is isolated on `feat/f03-constrained-ocp-11787`, based on F01 PR
#11808. It uses the existing multiple-shooting solver as one synthetic
comparator and a bounded sparse midpoint inverse-dynamics collocation spike
as the other. No new general solver framework is introduced. The study's
analytic Jacobian has an independent finite-difference test; a stiff coarse
transcription demonstrates that tiny node defects can still fail fresh
forward replay. Both candidates use the same one-coordinate rotary truth
plant, horizon and position target, but shooting has one constant input DOF
versus collocation's per-interval inputs. Their defect units and solver
objectives differ; a production winner cannot be selected from this fixture.

The benchmark API records predeclared cold/warm starts, rejected attempts,
node objective, hard constraint violations, fresh ZOH replay gap, observation
RMSE, solve/replay/total callback wall time and Python-tracked peak allocation with platform and
library versions. The timer covers the synthetic callback only. It excludes
private capture preparation, F06 native validation, artifact export and a
full-body engine. The private data protocol D02 and early F06 replay validator
are integration prerequisites before capture fitting or a production backend
decision. Keep this issue open until those receipts exist.

Reproduce with `python -m pytest
tests/unit/motion_matching/test_sparse_collocation_spike.py -q`, then run
`python scripts/ci/check_architecture_budget.py`, scoped Ruff/mypy and manual
governance. The F03 PR body records final commit, CI results and any unresolved
provider or private-data blockers. No private capture artifacts are committed.

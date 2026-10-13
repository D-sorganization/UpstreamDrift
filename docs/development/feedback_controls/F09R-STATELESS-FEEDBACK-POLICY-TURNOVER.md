# Stateless Feedback Policy Turnover (#12390)

## Change

DistributedFeedbackPolicy now computes an immutable FeedbackRequest from the
existing schedule, exact state, task providers, and policy configuration. The
request retains time, step duration, schedule index, phase/task provenance,
nominal input, feedback correction, total selected input, full-nv generalized
effort, channel IDs, generalized indices, and a selected-coordinate mask. The
caller supplies the expected time, so request evaluation remains stateless.

DistributedFeedbackController subclasses the policy and retains its existing
public constructor and command_for_step interface. It still owns the applied
input history and step clock; those advance only after the allocator returns a
finite correctly sized applied input. The allocation boundary receives the same
full generalized-effort vector as before.

## Validation

The new policy tests use a shape-only nq=4/nv=3 adapter fixture (not a native
quaternion geometry test) for hand-computed permuted nq/nv effort mapping,
immutable request values and provenance, repeatable evaluation with one task
sample per call, disabled-policy nominal-only requests, nonfinite overflow
refusal before allocation, preserved allocator-failure retry behavior, and
invalid allocator outputs that do not advance controller history. The existing
distributed-feedback tests remain in the focused selection.

Focused command (supported Python 3.12 environment):
$env:PYTHONPATH = '$PWD\src\shared\python'
python -m pytest -q tests/unit/motion_matching/test_distributed_feedback_policy.py tests/unit/motion_matching/test_distributed_feedback.py

Result: 23 passed, one pre-existing state_manager deprecation warning. Ruff check
and format-check passed for the two Python files; targeted mypy passed for the
source and test files. Manual governance and the existing change-fragment validator
also passed. No native simulation ran for this change.

## Limits and Next Step

The request boundary is a software refactor of the torque controller's existing
policy calculation. It does not add a muscle allocator or engine-specific
actuation contract, infer coordinate IDs or units, integrate a plant, or validate
capture matching. Native muscle qualification and the 17-row model inventory
remain unqualified; this change does not make a parity claim. A future muscle
bridge must bind its own explicit coordinate identities/units and preserve the
separation between requested generalized effort and applied muscle commands.

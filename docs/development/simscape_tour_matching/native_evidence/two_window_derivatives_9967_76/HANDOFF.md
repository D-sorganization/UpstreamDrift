# Two-Window Stage 2 Audit 76 — Derivative Gates Unresolved

**Pause here. No optimizer is authorized by this result.** The bounded audit
completed all16 signed trials and returned process exit0, but scientific status
is `failed_derivative_gates`. Overall qualification is not established. No
retry, threshold relaxation or subsequent optimization occurred. Call time was
225.112 s and launch-to-terminal250.611 s. All processes are terminal.

## Fixed Fixture and Exact Providers

Candidate remains exact returned73:
`786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a`.
Runtime73, original model/clock and75's saved0.6 node/basis remain unchanged.
The chart reference is the saved uninterrupted state. The second-window base
state is existing zero retraction's state, with its matching implicit54x42
Jacobian; its tiny shift from the reference is recorded. The first-window
initial state remains original q0/qd0. Global absolute-time sextic inputs are
perturbed with existing increment_native_bernstein at basis_duration0.85;
the baseline coefficients are never reconstructed before sensitivity replay.

First-window sensitivities have81 B4/B5/B6 columns. Second-window sensitivities
have81 control plus42 initial-node columns. Both passed primal marker agreement:
first115853 evaluations/44.664 s, discrepancy6.16963e-11 m; second48383
evaluations/17.007 s, discrepancy2.68474e-12 m. Settings: grouped error control,
rtol1e-10, atol1e-12, max_step0.0000625, cap200000 per augmented call. Independent
and finite primal replays retain rtol1e-11/atol1e-13 and the same step limit.

Four exact direction vectors are saved in receipt.json: LSInputX B6 unit
coefficient; top right singularvector of node Dq; right-nullspace vector of
Dq with position norm2.25911e-17 and rate norm1; and deterministic seed996776
normalized mixed81-control plus node-position/node-velocity direction.
Node-only trials reuse the first-window baseline. Controls use two replays
per sign; no more than16 signed trials were executed. Each perturbed node is
retracted by the existing provider with75's scales/radius/tolerance.

## Results and Limits

Every comparison includes observed marker XYZ, both physical endpoints and
first-window-end minus next-node continuity. Endpoint and continuity q/qd
blocks are reported separately, plus full54D norms scaled with75's recorded
state scales. Analytic continuity explicitly subtracts the next-node implicit
Jacobian. Relative L2 denominator is the analytic norm; the unchanged resolved
block gate is1e-3. Absolute errors are also preserved.

| Direction     |         Step | Result                                                                   |
| ------------- | -----------: | ------------------------------------------------------------------------ |
| LSInputX B6   |         1e-5 | Continuity q1.209e-3, qd1.614e-3, scaled1.496e-3 fail; other blocks pass |
| LSInputX B6   |         1e-4 | All blocks pass                                                          |
| Node Position | 1e-5 and1e-4 | All resolved blocks pass; weak continuity-qd block unresolved            |
| Node Velocity | 1e-5 and1e-4 | All resolved blocks pass; weak continuity-q block unresolved             |
| Mixed         |         1e-7 | Resolved relative errors3.812e-3–5.011e-3 fail                           |
| Mixed         |         1e-6 | All blocks pass                                                          |

Node-position continuity-qd analytic norm is4.995e-14; finite-replay errors
are1.665e-10 and8.374e-11. Node-velocity continuity-q analytic norm is2.254e-17;
errors are1.527e-11 and3.275e-12. These weak blocks cannot support a meaningful
relative-accuracy claim. The diagnostic flags analytic norms below1e-12 as
weak for reporting, and does not promote them to passes except exact zeros.
This reporting threshold is not a new physical acceptance threshold.

Step dependence is compatible with numerical resolution limits but does not
prove their cause. Even-response norms combine curvature and numerical noise;
they are explicitly not claimed as measured pure integration-error floors.
The next agent should examine preserved control-polynomial perturbation accuracy,
endpoint resolution and weak-block absolute magnitudes at this fixed fixture
before deciding on another audit. Do not discard smaller-step failures, waive
the gate, or infer that larger-step passes qualify the entire optimizer chain.

## Complete Reproduction and Turnover

`audit.py`, `receipt.json`, `summary.json` and `raw-run.zip` preserve exact
directions, steps, commands/environment, sources, inputs and all28 raw outputs.
The ZIP contains both marker/state sensitivity tensors **and actual primal
q/qd/markers**, all16 signed trial q/qd/marker arrays, perturbed coefficients,
retracted nodes, analytic/central comparison arrays, node chart and runtime.
Source/input/output/archive hashes were verified, including identity with the
qualified runtime73. Raw archive SHA256:
`7348dab56fdd1ccb39e3c2ee2e86a8c568b0f2a4a57eddd64beb96dfacf819e5`.
Force-add the ignored36.3 MB ZIP when committing. Driver Ruff format/check
passed before execution. Do not edit archived bytes to alter qualification.

Remote output is `/mnt/c/Users/diete/native-two-window-derivatives-9967-76`;
runtime is `/home/dieterolson/native-regularized-fit-9967-73`. The exact launch
vector is in summary.json. Any future experiment requires a new output path.
The user requested a pause and cheaper-agent handoff: no further computation
or optimization should be inferred from this document. Root owns final commits
and the next-agent execution prompt. No production edits or commits were made
by this execution agent.

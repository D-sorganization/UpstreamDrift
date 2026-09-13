# Two-Window Stage 1 Preflight 75 — Passed

The zero-displacement fixture passed all existing trajectory and closure gates.
No optimization, window sensitivity integration or directional derivative audit
was performed. The run73 candidate remains a rejected 0–0.85 s swing fit.

Exact candidate SHA256:
`786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a`.
Original model raw SHA256:
`b817fea76407f71b29e42aeb188890df53d2d86874c7e5875d14d353f4a1a248`.
Coefficients were used directly: no Bernstein conversion or coefficient
roundtrip. The actual run73 state NPZ hash was checked against its archived
summary, and coordinate inventory, original q0/qd0 and capture clock matched.

## Fixture and Comparison

The shared sampled_shooting_windows selects exactly sample index216 at 0.6 s:
windows contain 217 and 91 samples, with the boundary shared. Window0 starts
at original q0/qd0. Window1 starts at the **saved uninterrupted run73 state**
at 0.6 s, not window0's computed endpoint. The discrepancy between that
endpoint and the saved node is measured independently. This is a shooting
fixture, not a declaration that independently started windows replace final
uninterrupted acceptance. Both windows evaluate the exact polynomial at
absolute global times using existing replay_window.

Settings match73: rtol 1e-11, atol 1e-13, max_step 0.0000625, closure bound
1e-7. Measured comparison against the saved uninterrupted73 trajectory:

| Quantity                              | Maximum Difference |              Existing Gate |
| ------------------------------------- | -----------------: | -------------------------: |
| Native q                              |        4.42211e-11 |                       1e-6 |
| Native qd                             |         8.80385e-9 |                       1e-4 |
| Marker Euclidean Distance             |      5.36901e-12 m |                     1e-7 m |
| First Endpoint Minus Saved Node       |        9.76996e-14 | Measured Within q/qd Gates |
| Refreshed Segmented Pose/Rate Closure |        6.15783e-11 |                       1e-7 |

Window integration counts were 115853 and 48383; times 12.085 and 5.642 s.
The full preflight took 18.268 s; launch-to-terminal took 24.874 s, exit 0.

## Full Position/Velocity Node Chart

The reference is the saved uninterrupted0.6 node, selected through existing
select_shooting_states and independently closure-checked (max5.83267e-12).
The native closure_trajectory_linearization supplies its first12 pose/rate
rows and q/v columns, using its configured centered step1e-6. The shared
scaled_tangent_basis and retract_node create a **42-dimensional** chart in
the full54-dimensional q/qd state.

Explicit inherited numerical scales: q0.1 (native metres/radians), qd1.0
(native metres/s or radians/s); closure pose rows0.01, rate rows0.1; radius0.5
in scaled chart norm, retraction tolerance1e-8. These are numerical scaling
choices, not physical motion limits. Scaled closure singular values range
3.53595–31.84446. Zero-coordinate retraction moves the node by at most
4.74865e-12 in physical components (scaled norm3.98181e-11), with scaled
closure1.77636e-14. That corrected state was **not** injected into either
window replay. The implicit state Jacobian is archived for subsequent testing;
its directional accuracy was not independently qualified in Stage1.

## Evidence and Next Gate

`preflight.py`, `receipt.json`, `summary.json` and `raw-run.zip` preserve exact
driver, command/environment, original inputs and runtime source. The archive
contains both actual window q/qd/marker arrays, per-time differences, saved
and reference node, basis, closure Jacobian, retraction Jacobian, scales and
refreshed closure arrays. Archive SHA256:
`56d6e76b03291db562656c6b3f98e7c2e35b158f863e2fa22eea9a5e5a75a09b`.
All input/output/archive hashes and identity with frozen qualified runtime73
were verified. No new runtime or numerical-provider replacement was needed.
Runtime73's qualification and namespace-scaffolding limits remain applicable.
Force-add the ignored ZIP. Driver Ruff format/check passed before execution.

Remote output: `/mnt/c/Users/diete/native-two-window-preflight-9967-75`.
Runtime: `/home/dieterolson/native-regularized-fit-9967-73`. Both remain
unchanged. Root reviews Stage1 before authorizing Stage2 endpoint, marker,
node-retraction and continuity derivative checks. No optimizer should run
before that preflight passes. No production edits or commits were made here;
all processes from75 are terminal.

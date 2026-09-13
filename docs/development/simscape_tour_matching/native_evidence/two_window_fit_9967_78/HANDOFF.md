# Two-Window Direct-Node SLSQP Trial 78 — Parity Passed, Budget Exhausted

Run78 is TERMINAL (exit0, launch-to-terminal857.6 s, driver771.2 s,10 residual
evaluations,22 window sensitivity solves). It returned the exact run73
candidate unchanged: the evaluation budget was exhausted before any SLSQP
iterate satisfied the projected continuity equality within1e-7, so the shared
backend's fallback selected the only feasible evaluated point, the start. No
improvement is claimed. The exploratory candidate remains rejected run73.

## Zero-Displacement Parity (Gate Passed)

Audit-only run78-audit and the fit's own preflight both evaluate the two
windows (0–0.6 s from original q0/qd0,0.6–0.85 s from the zero-retracted
saved node) with once-only shared-boundary observations, terminal weight10 and
the run73 effort penalty (weight0.01, scales100 N/20 Nm, built on parent19
with the recovered run73 increment). Segmented score with effort18.3001613827
versus run73's returned18.3001613827: relative difference1.88e-12. Effort cost
1.5453277580617921 matches run73 exactly. Markers versus the saved run73
replay differ by at most1.8e-11 m; the initial scaled defect is4.0e-11.
Restart roundoff: recovered Bernstein increment versus10\*(x73−1) agrees to
3.9e-14, and theta is applied to the exact run73 coefficients (zero theta
returns the identical candidate hash). Projected continuity constraint
(basis.T times scaled [window0 endpoint sensitivity, −node Jacobian]) has full
rank42 with singular values1.00 to349.8.

## Formulation and Bounds

Theta: 81 physical B4/B5/B6 Bernstein increments over run73, bounds
10*(0.8−x73) to10*(1.2−x73) from run73's saved parameters (zero interior).
Node: 42 chart coordinates at0.6 s, zero initial, box ±0.05 (chart norm
≤0.32 <0.5 radius), retraction radius0.5/tolerance1e-8/step1e-6 as in
run75/76. window_jacobian_state_coordinates="node", shared_boundary_policy=
"once", constraint_projection=basis.T on scaled defects, defect scales0.1/1.0,
defect_tolerance1e-4, equality_tolerance1e-7, variable_scales10 for theta and
1 for nodes, max_iterations6, max_nfev10. Sensitivities: grouped control,
rtol1e-10/atol1e-12, max_step0.0000625, cap200000; primal1e-11/1e-13.
Runtime78 = runtime77 plus the once-only shooting option (135 qualification
tests, driver --help). All22 sensitivity solves used exactly115853 (window0)
and48383 (window1) evaluations; worst primal agreement2.2e-10 m.

## Optimizer Path and Iterate Audit

SLSQP took four accepted line-search steps in ten evaluations (full steps were
rejected each time). The bounded iterate audit (`audit_iterates.py`, primal
replays only, one uninterrupted original-state replay per iterate):

| Eval | theta max (N/Nm) | node chart norm | segmented obj | uninterrupted obj | whole mm | early mm | terminal mm | club mm | scaled defect | projected violation |
| ---: | ---------------: | --------------: | ------------: | ----------------: | -------: | -------: | ----------: | ------: | ------------: | ------------------: |
|    1 |            0.000 |          0.0000 |       18.3002 |           18.3002 |   28.105 |   10.860 |      65.398 |  30.041 |       4.0e-11 |             5.3e-14 |
|    4 |            0.055 |          0.0056 |       18.2605 |           18.2603 |   28.147 |   10.875 |      65.222 |  31.049 |       1.6e-04 |             9.4e-05 |
|    6 |            0.399 |          0.0318 |       17.9822 |           17.9681 |   28.301 |   10.913 |      64.113 |  29.905 |       1.6e-03 |             7.0e-04 |
|    8 |            0.857 |          0.0663 |       17.5537 |           17.5179 |   28.322 |   11.006 |      62.664 |  24.670 |       1.9e-02 |             1.0e-02 |
|   10 |            1.122 |          0.0848 |       17.2780 |           17.2979 |   28.241 |   11.072 |      62.070 |  30.866 |       3.0e-02 |             1.7e-02 |

Reading: the terminal-weighted objective falls monotonically (−5.5 % at eval
10, uninterrupted) and terminal RMS improves3.3 mm, but whole-swing RMS rises
0.14 mm and early RMS0.21 mm, so this is the run73 objective trading whole
error for terminal error, not a better swing. Projected violations grow with
step length (SLSQP satisfies only the linearized continuity), so no iterate
was feasible to1e-7 and the returned point is the start. Active bounds: zero
theta and zero node bounds at the returned point. Nothing here relaxes the
25/35 mm acceptance gates; the head-marker rigidity floor still applies.

## Cost Observation and Next Bounded Step

The driver integrated sensitivities for every residual evaluation (68 s each),
so ten evaluations consumed the whole budget in four iterations. Run79 keeps
the identical formulation with primal-only residual replays (sensitivities
only on Jacobian requests), max_iterations15/max_nfev45, and records the
objective without defect rows plus the first-Jacobian linear-model prediction
for explicit predicted-versus-actual reduction.

## Evidence

`fit.py` (SHA25678e27203…, identical to the executed remote copy),
`audit_iterates.py`, `config.json`, `receipt.json`, `returned.json`,
`returned-nodes.json`, `evaluations.jsonl`, `jacobians.jsonl`,
`parity-receipt.json`, `parity-launch.json`, `launch.json`,
`runtime-receipt.json`, `iterates-receipt.json`, `iterates-launch.json`,
`raw-run.zip` (SHA256
`fda4959f1cfa650766281f194bb91e99a992e0ff8038f1516fbd1c398ba0ef02`, both
output folders, logs, launch vectors, driver, overlay manifests and full
runtime78 source) and `raw-iterates.zip` (SHA256
`a5516909b5cd8f824896464aa16b8be78919ba1ceb265fdd530567a860d2f476`). Remote
outputs `/mnt/c/Users/diete/native-two-window-fit-9967-78{,-audit,-iterates}`;
runtime `/home/dieterolson/native-two-window-fit-9967-78`. All processes are
terminal. Future experiments need new output paths.

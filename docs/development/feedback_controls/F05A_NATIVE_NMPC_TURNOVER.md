# F05a Native NMPC Turnover (#11904; Parent #11789)

This child contributes an optional bounded robust direct-shooting controller,
one-hinge native MuJoCo execution adapter, and independently replayed paired
comparison with the existing F02 TVLQR controller. The source-controlled
supported-provider measurement is
`F05_NATIVE_PAIRED_RECEIPT_MJ38.json`; an earlier 3.3.4 run is retained in
`F05_NATIVE_PAIRED_RECEIPT.json` and is development-only because project
requirements specify MuJoCo >=3.6.0. Neither receipt promotes the F05 parent.
The supported 3.8.0 first run and repeat are both retained: initial 0.4 rad
timed out on 8/8 calls in each and had NMPC/TVLQR RMSE 0.400/0.348 rad;
initial 0.7 rad optimized 2/8 calls in each and had 0.617/0.622 rad RMSE.
NMPC p50 controller-call time was approximately 1.001 to 1.002 s per trial
against a 0.010 s native integration step. The unsupported 3.3.4 development
run had different optimized counts, so success
rate is not treated as a stable provider comparison.

The implementation was test-first: missing controller and native adapter
imports were observed RED, followed by bounded/unsafe-fallback tests and
native controlled-versus-frozen replay GREEN. Native actuation is post-limit
direct unit motor torque, held at each 10 ms integration step. The native
adapter preflights F06 model, exact state and Tools T01 bundle policy, then
copies the executed torque history into an independent time-only replay. It
checks position, velocity and full native integration state within 1e-12.
The F02 baseline gets the same replay check. Both controllers start from
0.4 and 0.7 rad on a 40% mass-perturbed execution plant. NMPC predicts through
both nominal and perturbed plants, with hard state/input bounds, finite
evaluation budget, cooperative wall budget and labeled fallback on no result.

The evaluation disposition is **retain TVLQR, reject SciPy shooting NMPC for
10 ms native-step or full-swing operation**. The shooting solve can use much
longer than a native step; warm-start/fallback counts vary with machine load.
The receipt preserves every predeclared trial rather than selecting a winning
run. This is a software and runtime result, not a human-control or motion
capture finding. The replacement candidate should use native derivatives and
hard bounded optimal control, e.g. the existing Crocoddyl BoxFDDP or a
properly integrated acados path, with a predeclared latency and contact policy.
Do not infer it will win before paired native receipts exist.

The new adapter is intentionally one hinge, one direct torque motor and no
contact. It rejects other topology/actuator mapping rather than converting
hidden muscle activation, passive force or contact reaction to a torque input.
The controller admits only named Euclidean tangent states; nq != nv and native
muscle/contact states need their own explicit provider. The benchmark uses
exact simulated-state observations and a synthetic target, so private capture
timing, state estimation, held-out trials, contact/grip, six engines, muscle
excitation replay and full swing remain open. Keep F05 #11789 open.

Reproduction on a Python environment with MuJoCo >=3.6.0:

```powershell
$env:F05_BENCHMARK_RECEIPT='1'
python -m pytest -q -n 0 -o junit_family=legacy tests/unit/motion_matching/test_bounded_nmpc_native.py --junitxml native-paired.xml
python scripts/f05_native_paired_receipt.py --junit native-paired.xml --host-alias local-host --output docs/development/feedback_controls/F05_NATIVE_PAIRED_RECEIPT_MJ38.json
python -m pytest -q -n 0 tests/unit/motion_matching/test_bounded_nmpc.py
```

The exact source digest and Tools pin are in the receipt. The canonical
calculation and caveats live in
`manuals/upstreamdrift/chapters/23-native-bounded-nmpc.qmd`; the manual
calculation registry remains blocked.

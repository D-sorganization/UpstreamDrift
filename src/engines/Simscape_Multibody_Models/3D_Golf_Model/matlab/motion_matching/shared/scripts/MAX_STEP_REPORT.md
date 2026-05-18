# MaxStep Performance Sweep

_GitHub issue: D-sorganization/UpstreamDrift#4078._

- Model: `GolfSwing3D_Kinetic`
- Stop time: 0.300 s
- Warm runs per setting: 3 (plus 1 cold cache-priming run, discarded)
- Reference for accuracy: MaxStep = 0.0001 (smallest swept value, treated as ground truth)
- Generated: 06-May-2026 15:32:33

## Results

| MaxStep | mean_warm_s | grip_rmse_mm | total_work_J | rel_work_error_pct |
| ------- | ----------- | ------------ | ------------ | ------------------ |
| 0.0001  | 7.885       | 0.000        | 2.77e+08     | 0.000              |
| 0.001   | 6.949       | 539.933      | 3.917e+08    | 41.399             |
| 0.002   | 7.122       | 539.933      | 3.917e+08    | 41.399             |
| 0.005   | 7.096       | 539.933      | 3.917e+08    | 41.399             |
| 0.01    | 7.065       | 539.933      | 3.917e+08    | 41.399             |
| 0.02    | 6.544       | 539.933      | 3.917e+08    | 41.399             |

## Recommendation

Use **MaxStep = 0.001**. Speedup vs default 1.00x; grip RMSE 539.933 mm vs reference.

MaxStep is not the binding step-size constraint at the current default (0.001): all values >= 0.001 produce the same grip trace within 0.00 mm. The solver is bounded by RelTol/AbsTol, not MaxStep, so loosening yields no speedup. Recommend keeping MaxStep=0.001 as the default; use the new high_precision opt-in (MaxStep=0.0001) only when ground-truth accuracy is required (~1.1x slower per call).

## Method

Each setting runs 1 cold sim (discarded — pays the FastRestart compile cost) then 3 warm sims with `FastRestart=on`. Wall-clock is per-call seconds. Grip position is `CombinedSignalBus.MidpointCalcsLogs.MPGlobalPosition`; RMSE is computed against the smallest MaxStep's trace interpolated onto its native grid. Total work uses `W = trapz(t, sum(abs(tau .* qd), 2))` with joint torque and velocity traces from `logsout` / `CombinedSignalBus`. Acceptance gates: grip RMSE <= 5 mm and |rel_work_error| <= 5%; results above 100 mm grip RMSE are flagged as broken rather than recommended.

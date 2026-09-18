# Native Pinocchio Iteration Probe

All runs: 654 frames, 1.813889 s full driver capture, identical model/capture and weights, Pinocchio 4.1. Only per-frame iteration budget changes. Address continuation uses its existing fixed 60-iteration stages. All results remain REJECTED_DIAGNOSTIC.

| Iterations | IK Time (s) | Whole RMS (mm) | Club RMS (mm) | Peak Effort (N m) | Acceleration Parity Residual |
| ---------- | ----------- | -------------- | ------------- | ----------------- | ---------------------------- |
| 15         | 14.21       | 133.531        | 50.204        | 9196.23           | 0.00209782                   |
| 60         | 28.94       | 95.209         | 25.082        | 4433.36           | 0.0131845                    |
| 180        | 97.32       | 79.595         | 18.122        | 5229.96           | 0.00112351                   |

The 15-iteration fast fit is not sufficient to estimate attainable marker accuracy. More iterations materially reduce residuals, but neither convergence to a global optimum nor anatomical calibration is established. Force plausibility does not improve monotonically with marker fit.

## Next Implementation

PF-02: report projected-gradient/KKT residual, cost reduction, accepted/rejected steps, active bounds and termination reason. Run selected difficult-frame multi-start and closure-continuation probes before anatomical calibration changes. Allocate more work to nonconverged frames and short overlapping windows; warm-start neighboring frames. Do not merely increase the global default to 180. Preserve all group metrics and verify analytical Jacobians against finite differences.

PF-02/PF-05: replace independent post-filtering with closure-consistent trajectory refinement and consistent q/v/a. Audit translation (m) and rotation (rad) separately; the current combined closure norm is not a physical millimeter metric. Confirm contact-law compatibility before optimizing forces.

PF-07/PF-09: require uninterrupted native forward replay. The related MuJoCo PR #10448 reports same-pose FK agreement but rejected replay (571/654 frames, G1 RMS 0.940137 m); this confirms geometry transfer alone is insufficient.

PF-10: native model plus capture/IK/replay layers, coordinate-frame metadata, exact candidate/model hashes, physical-time scrub and truthful qualification. The existing generic overlay swaps Y/Z; do not use it as proof of native Z-up rendering. Attached diagnostic rendering explicitly uses native Z-up and labels kinematic playback.

Source correction: PR #10451 commit cbd72595d. Structured metrics and hashes: pf09_ik_iteration_probe.json. Full-swing acceptance, realistic GRFs, iron, and all-engine qualification remain open.

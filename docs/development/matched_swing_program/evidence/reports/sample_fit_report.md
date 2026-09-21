# Fit-Quality & Acceptance Report: MUJOCO (Driver)

> **Overall Run Verdict: REJECTED**

## Qualification & Audit Notes

> [!IMPORTANT] > **Qualification:** kinematic IK and computed-torque tracking milestone on the MuJoCo full-body model (anthropometric geometry (AN-1); unqualified until Simscape parity); not a fit, not acceptance
>
> **Rejection / Gate Reason:** None

## Shared Standardized Metrics

| Standardized Metric  | Measured Value      |
| -------------------- | ------------------- |
| Whole Marker RMSE    | 52.27 mm (0.0523 m) |
| Early Marker RMSE    | —                   |
| Terminal Marker RMSE | —                   |
| Club Marker RMSE     | 40.48 mm (0.0405 m) |
| Pelvis Yaw RMSE      | —                   |

## Quantitative Acceptance Gates

_No formal quantitative acceptance gates evaluated for this receipt._

## Cryptographic Provenance (#8820 / U3)

| Provenance Field  | Value                                                              |
| ----------------- | ------------------------------------------------------------------ |
| Physics Engine    | `mujoco` (version: `3.3.4`)                                        |
| Git Commit        | `69b95c32c`                                                        |
| Candidate SHA-256 | `3f94aa92f28acde61b6b2f8c871ad536f93eef62cbbae5d0ffe7c85e4b41a6ad` |
| Receipt SHA-256   | `3f0a0809ba61c2dfa5a17c1c67f905b157c71f7c9a5fa8c880c117d6f2890d1d` |
| Capture / Lane    | `driver` / `ground_support`                                        |
| Overall Verdict   | **REJECTED**                                                       |
| Generated (UTC)   | `2026-09-21T03:54:39Z`                                             |

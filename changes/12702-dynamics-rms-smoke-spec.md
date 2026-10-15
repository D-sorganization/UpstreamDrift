---
issue: 12702
summary: "Smoke test dynamics RMS 0.51 m was not a code regression: test_mujoco_native_match_smoke[driver] ran the pipeline's default native v2 spec, not the canonical anthro_driver document its 0.15 m bound and 0.089 m figure come from; the driver now passes --spec full_body_spec_anthro_driver.json (current main reproduces receipt anthro_driver: IK 0.0465, dynamics 0.0719), pinned by a fast unit test; bounds unchanged"
dl_state: "in_review"
next_step: "Merge the PR; put the slow smoke in a nightly lane (it runs in no CI lane today)."
branch: "fix/12702-dynamics-rms-regression"
---

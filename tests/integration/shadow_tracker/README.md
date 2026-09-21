# Shadow Tracker Integration Test Home

Reserved for real camera/render/state/engine/service boundary tests under
[Validation](../../../docs/plans/shadow_tracker/VALIDATION.md).
`test_model_probe.py` exercises fresh real MuJoCo diagnostic runs. Its success
means the probe works, not that the golfer motion is valid. Mark optional
engines explicitly and report skips. Mocks may test orchestration but cannot
qualify real dynamics. Live tests must verify full-interval continuous replay,
realized controls, camera conventions and contact/grip audits.

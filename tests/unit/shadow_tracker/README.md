# Shadow Tracker Unit Test Home

Reserved for test-first implementation under
[Work Packages](../../../docs/plans/shadow_tracker/WORK_PACKAGES.md).
`test_pilot_metrics.py` tests unit-separated diagnostics for the ST-01 probe.
For product records, start with the frozen ST-02A/C/B packets in
[Ready Tasks](../../../docs/plans/shadow_tracker/READY_TASKS.md).
Unit tests must run without a GPU, GUI, network, model download or physics engine.
Use independent expected outcomes and immutable tiny fixtures.

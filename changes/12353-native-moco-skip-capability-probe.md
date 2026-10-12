---
issue: 12353
summary: "Allow native Moco preparation to skip only the short capability probe while recording runtime as not attempted and keeping solve readiness false"
dl_state: "in_review"
next_step: "Root review of implementation, focused tests, and manual-governance updates."
branch: "feat/native-no-probe-12353-20261012"
paths: "src/engines/physics_engines/opensim/python/tour_matching/native_moco_preparation.py,src/engines/physics_engines/opensim/python/tour_matching/cli.py,tests/unit/engines/opensim/test_native_moco_probe_mode.py,tests/unit/engines/opensim/test_native_path_optimizer_domains.py,manuals/upstreamdrift/chapters/56-native-diagnostic-execution-contract.qmd,manuals/upstreamdrift/calculation-registry.json,SPEC.md,AGENT_HANDOFF.md,docs/development/feedback_controls/F07_NATIVE_NO_PROBE_PREPARATION_TURNOVER.md,docs/agent_context/README.md,docs/agent_context/index.html"
---

The explicit `run_capability_probe=False` API mode and `moco-native --prepare-only --skip-capability-probe` CLI mode preserve the probe-enabled default. A skipped probe is serialized as requested=false with runtime `not-attempted`, and the software-solve readiness property remains false. No forward integration runs on this preparation path. Ordinary preparation audits still run and may require native model finalization, state restoration, or initialization. The mode does not bypass other blockers or establish runtime viability, solver execution, physiology, replay, or parity. Focused verification uses fake native bindings; no native run was performed.

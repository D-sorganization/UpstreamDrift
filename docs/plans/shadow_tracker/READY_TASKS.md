# Lower-Cost Agent Dispatch Queue

## Restart Queue — 2026-09-16

Read [Current Turnover](TURNOVER_CURRENT.md) first. #10231 is merged through
#10253; do not repeat its completed work. Renderer #10232 has an existing,
conflicted PR #10264 with a known import collection failure; its owner resumes
that PR. The next independent small-agent task is #10233 revision integrity and
persistence. #10273 tracks native PTS and stored timing authority. Then deliver
#10134's real manual-evidence launcher journey. Older queues below are historical
where their ordering conflicts with this section.

## Ready to Assign

Planning PR #10136 is merged. The image-only contract decisions and measured
model findings in this handoff must also be present on the checkout used by the
worker. Use [Handoff PR #10144](https://github.com/D-sorganization/UpstreamDrift/pull/10144)
or its merged contents; never infer that the complete
ST-01 scientific qualification has passed.

| Packet | Issue                                                                   | Readiness                                                                               | Allowed Work                                                    |
| ------ | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| A      | [#10137](https://github.com/D-sorganization/UpstreamDrift/issues/10137) | Implemented in [PR #10145](https://github.com/D-sorganization/UpstreamDrift/pull/10145) | SourceAsset, FrameIdentity, sequence validation                 |
| C      | [#10139](https://github.com/D-sorganization/UpstreamDrift/issues/10139) | Implemented in [PR #10147](https://github.com/D-sorganization/UpstreamDrift/pull/10147) | Convert existing camera types with explicit direction           |
| B      | [#10138](https://github.com/D-sorganization/UpstreamDrift/issues/10138) | Implemented in [PR #10148](https://github.com/D-sorganization/UpstreamDrift/pull/10148) | Immutable binary masks, correction lineage and content identity |
| D      | [#10140](https://github.com/D-sorganization/UpstreamDrift/issues/10140) | Implemented in [PR #10164](https://github.com/D-sorganization/UpstreamDrift/pull/10164) | Fix IK coordinate packing, establish forward kinematics parity  |
| E      | [#10141](https://github.com/D-sorganization/UpstreamDrift/issues/10141) | Implemented in [PR #10165](https://github.com/D-sorganization/UpstreamDrift/pull/10165) | Separate production closure units and acceptance status         |

## Current Dispatch

[Current Turnover](TURNOVER_CURRENT.md) supersedes older pickup instructions.
#10231: source timestamps, physical-time provenance and genuinely bounded decode (implemented in PR #10253).
#10232: articulated geometry/state binding, clipping and calibrated filled masks (implemented; landing in PR).
#10233: revision collision/parent integrity and persistence; independent of physics.
Use [Continuation Prompt](CONTINUATION_PROMPT.md), then all ST-07–ST-12 gates.
Merged #10205/#10212/#10214 establish partial progress, not whole-stage acceptance.

## Commands by Packet

From the target UpstreamDrift checkout, after installing normal development
dependencies and initializing the exact pinned Tools provider:

```bash
# Add focused red/green tests for the selected corrective task.
python3 -m pytest tests/unit/shadow_tracker tests/integration/shadow_tracker --timeout=60
python3 -m ruff check src/shared/python/shadow_tracker tests/unit/shadow_tracker
python3 -m ruff format --check src/shared/python/shadow_tracker tests/unit/shadow_tracker
python3 -m mypy src/shared/python/shadow_tracker --follow-imports=silent
python3 -m scripts.shared_tools.divergence_inventory --write
python3 -m pytest tests/unit/scripts/test_divergence_inventory.py --no-cov -q
```

Commands targeting future test files become runnable when the packet creates
those files; their absence now is intentional. Do not commit xfailed placeholders.
Packet C additionally runs the existing camera-observation and pipeline contract
tests found by the discovery workflow. CI coverage and repository gates still
apply; a passing focused test alone does not complete a packet.
Commit both generated shared-tools inventory files whenever adding source files.
The inventory includes source-directory Markdown files as well as Python modules.

## Review and Turnover Checklist

- Public fields and constructor/serialization signatures match the frozen spec.
- Invalid data fail with the specified exception and field; no coercion or repair.
- Original vs processed time, unknown physical time and coordinate direction survive.
- Immutable ownership tests include mutation of constructor inputs.
- Tiny fixtures are redistributable and do not require a network or GPU.
- All acceptance rows for that packet pass; source/consumer tests pass too.
- No hidden model imports, copied camera math, deep object access or unrelated edits.
- Evidence and current issue state are updated; pending parent requirements remain.

## Still Blocked for General Worker Dispatch

ST-01's physical gate profile and full model qualification, ST-06 initialization,
ST-07 full-body rollout integration, ST-08 control fitting and ST-09 scientific
uncertainty remain specialist work. The model probe is repeatable but detects
invalid cross-boundary pose semantics. Generic 41-element array shape checks
cannot certify those mappings. A/B/C completion must not unlock those stages.

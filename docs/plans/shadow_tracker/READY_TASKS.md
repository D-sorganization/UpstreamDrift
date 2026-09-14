# Lower-Cost Agent Dispatch Queue

## Ready to Assign

Planning PR #10136 is merged. The image-only contract decisions and measured
model findings in this handoff must also be present on the checkout used by the
worker. Use [Handoff PR #10144](https://github.com/D-sorganization/UpstreamDrift/pull/10144)
or its merged contents; never infer that the complete
ST-01 scientific qualification has passed.

| Packet | Issue                                                                   | Readiness                                        | Allowed Work                                                    |
| ------ | ----------------------------------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------- |
| A      | [#10137](https://github.com/D-sorganization/UpstreamDrift/issues/10137) | Implemented in Packet A Topic Branch             | SourceAsset, FrameIdentity, sequence validation                 |
| C      | [#10139](https://github.com/D-sorganization/UpstreamDrift/issues/10139) | Ready After This Handoff Lands; Independent of A | Convert existing camera types with explicit direction           |
| B      | [#10138](https://github.com/D-sorganization/UpstreamDrift/issues/10138) | Ready After Packet A Merges                      | Immutable binary masks, correction lineage and content identity |
| D      | [#10140](https://github.com/D-sorganization/UpstreamDrift/issues/10140) | Specialist Review Required                       | Fix IK coordinate packing, then regenerate invalidated evidence |
| E      | [#10141](https://github.com/D-sorganization/UpstreamDrift/issues/10141) | Specialist Review Required                       | Separate production closure units and acceptance status         |

A and C can be assigned to separate agents because they have disjoint source
files. Serialize façade changes or let the integration reviewer reconcile those
few exports. B waits for A's actual merged API. This is an assignment plan;
no background agents have been started by the handoff.

## Exact Worker Prompt

```text
Implement Shadow Tracker Packet <A/C/B>, issue #<10137/10139/10138>.
Read repository AGENTS.md and CLAUDE.md, then:
  docs/plans/shadow_tracker/CONTRACT_FREEZE.md
  docs/plans/shadow_tracker/READY_TASKS.md
  docs/plans/shadow_tracker/AGENT_HANDOFF.md
Claim the child issue using Repository_Management before editing.
Implement only the chosen packet and its unit tests. Use the frozen field names,
types, schema tags, error policy and acceptance examples exactly. Write and run
the failing tests before implementation. Record red and green output in the PR.
No decoder, neural model, dynamics engine or scientific gate changes are allowed.
Use existing camera/state/contract utilities; no third camera convention.
Keep each PR small. Do not close #10122, #10124 or #10125 with this slice.
Update the handoff with commit, tests, remaining dependency and exact next task.
If the frozen API cannot be satisfied using the existing provider, report the
specific mismatch instead of changing the contract or returning mock success.
```

## Commands by Packet

From the target UpstreamDrift checkout, after installing normal development
dependencies and initializing the exact pinned Tools provider:

```bash
# Replace source_records with camera_bridge or mask_records for C or B.
python3 -m pytest tests/unit/shadow_tracker/test_source_records.py -n auto --timeout=60
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

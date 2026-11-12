# NSX-04 Torque Capacity Verification Status

Refs issue **#12553**. The `capacity_for` loader and
`src/config/loop_nullspace_registry.json` torque-capacity table are **code-complete
on main**. Remaining work is **source verification**: reading cited tables and
either promoting `UNVERIFIED` slots to `VERIFIED` or keeping them `ASSUMED` with
an explicit rationale.

## Inventory (Re-Anchor by Symbol)

These counts are pinned by
`tests/unit/loop_nullspace/test_nsq04_verification_status_12553.py`. They supersede
the stale snapshot in issue **#12553** (82 entries / 30 verified / 52
unverified).

| Measure | Value |
| --- | --- |
| Total `torque_capacity` entries | 94 |
| `VERIFIED` | 58 |
| `UNVERIFIED` (all `assumed: true`) | 36 |

Invariants enforced in CI:

- Every `VERIFIED` row has a non-empty `source.citation_id`.
- Every `UNVERIFIED` row has `citation_id: null` and non-empty `source.note` plus
  `rationale` (see `test_every_unverified_slot_has_a_specific_reason` in
  `tests/unit/loop_nullspace/test_capacity.py`).

## Open Work

Per issue **#12553**: hip adduction and hip rotation anatomical slots are now
`VERIFIED` (Swake 2026 male frontal-plane; Gonçalves 2021 Biodex rotation).
MTP extension (negative) and virtual input slots remain open.
Hill-type muscle/tendon force–length data (board items UD-09/10) are tracked
elsewhere (**#12723** / **#12593**), not in this issue.

## Related Contracts

- Registry schema and loader postconditions:
  `docs/research/loop_nullspace/contracts.md` (section **NSX-04 Torque
  Capacity**).
- Implementation: `src/shared/python/loop_nullspace/capacity.py`.

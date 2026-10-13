---
issue: 12435
summary: "Unblock Collate Change Fragments: the MOSAIC-13 multi-trial #11545 fragment no longer reopens shipped DL-#11545 as in_review; it records against the shipped entry"
branch: "claude/fix-dl-11545-shipped-fragments"
paths: "changes/11545-mosaic-13-closed-form-multi-trial-benchm.md"
---

Collation of changes/\*.md on main was aborting at the #11545 multi-trial fragment (dl_state in_review on a shipped entry). Fixed by dropping its dl_state/next_step (#11694 is merged). Remaining #11545 work stays tracked on the open issue; future #11545 fragments must not set a live dl_state - open a new issue with its own DL entry instead.

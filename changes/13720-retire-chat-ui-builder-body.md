---
issue: 13720
summary: "Retire the child chat UI builder body under the verified pinned Tools module alias."
dl_state: "in_review"
next_step: "Review the focused provider-origin, export, parity, inventory, and canonical validation results before publishing this physical-retirement slice."
title: "Retire Child Chat UI Builder Body"
owner: "codex"
branch: "codex/13720-ui-builder-retire"
paths: "src/shared/python/chat/_qt/ui_builder.py,tests/unit/repo_hygiene/test_chat_child_copy_sync.py,docs/shared_tools/divergence_inventory.v2.jsonl"
---

Branch: codex/13720-ui-builder-retire. This slice removes only the 352-line local build_chat_dock_ui body. The pinned-module alias contract and import-order identity checks remain active; the remaining local declarations are reserved for a later retirement slice. Validation recorded with the ready receipt.

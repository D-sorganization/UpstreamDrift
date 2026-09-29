# Review Comments Archive - 2026-05-09

Generated: 2026-05-09T20:37:06.810156

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4992: src/launchers/launcher_layout_manager.py:324

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Keep category labels aligned with grid section buckets**

`_get_model_category` now emits labels like "Physics Engines"/"Tools & Data", but `rebuild_grid()` still buckets cards under legacy keys ("Core Physics Engines", "Analysis Tools", "Utilities", etc. in `src/launchers/launcher_layout_manager.py`). Because of that mismatch, cards are routed to the fallback "Other" bucket and the new taxonomy does not rend...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4992#discussion_r3214243318)

---

### PR #4992: src/launchers/launcher_layout_manager.py:318

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Keep engine sidebar filter compatible with renamed category**

Changing the engine label to "Physics Engines" breaks the sidebar Engines route, because `_on_sidebar_routed` still sets `current_category_filter` to "Core Physics Engines" (`src/launchers/launcher_ui_setup.py`). After this change, clicking Engines produces an empty grid since no model category matches that filter string.

Useful? React with 👍 / 👎...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4992#discussion_r3214243319)

---

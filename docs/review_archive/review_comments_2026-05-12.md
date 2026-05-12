# Review Comments Archive - 2026-05-12

Generated: 2026-05-12T14:14:54.567354

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #5305: src/shared/python/ai/gui/assistant_panel.py:320

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Initialize chat UI before loading a persisted session**

The panel now connects `session_loaded` and immediately calls `_load_history()` during `__init__`, but `_load_history()` calls `ChatSessionManager.load_session()`, which emits `session_loaded` synchronously. That invokes `_on_session_loaded()` before `_setup_ui()` has created `_message_layout`, so users with an existing saved chat can hit an `AttributeE...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/5305#discussion_r3229881496)

---

### PR #5305: src/launchers/launcher_ui_setup.py:1066

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Route AI panel close through toggle state handler**

The close button wiring now directly sets `ai_panel` max width to 0, bypassing `toggle_ai_assistant()`, which is the only path that synchronizes `_ai_visible`, splitter sizes, and toggle button state. After closing via the panel's ✕, the sidebar toggle remains checked, so the next click applies another hide action and users typically need two clicks to reop...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/5305#discussion_r3229881500)

---


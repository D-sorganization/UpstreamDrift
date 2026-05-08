# Review Comments Archive - 2026-05-07

Generated: 2026-05-07T16:50:20.546989

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4370: src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/starting_pose_matcher.py:1160

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Keep Rx/Ry unlock setting when running shaft snap**

When `Rx/Ry` editing is currently enabled (`lock_xy_rotation == False`), calling `_snap_shaft` forces `cb_lock_xy` to unchecked via `setChecked(False)`, which re-locks XY rotation and disables those controls. This contradicts the comment (“leave as-is for user”) and unexpectedly changes user state after every snap, so subsequent manual alignment cannot cont...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4370#discussion_r3205307354)

---

### PR #4370: src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/starting_pose_matcher.py:1384

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Restore pose-event combo with matching item text**

Session load writes raw event keys (`"A"/"T"/"I"/"F"`) into a combo whose items are formatted as `"K - Label"`, so `setCurrentText` does not select the saved item. The displayed selection can therefore stay at a default while `slot.target_event` is set differently, and the next `_on_pose_event_changed` call rewrites all slots from the stale UI values, silent...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4370#discussion_r3205307358)

---


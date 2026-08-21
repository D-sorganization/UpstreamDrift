# UI/UX, Performance, and Proximal–Distal Modeling Review — 2026-08-21

Second review wave requested by the repository owner, following the
integration/organization review (PR #8815, issues #8816–#8876). Three
axes this wave, each verified against the code before filing:

1. **UI/UX** — desktop launcher and shared UI infrastructure; tool GUIs
   and the React/Tauri web app. Responsiveness, error-state UX,
   feedback, consistency, accessibility, state persistence.
2. **Performance** — runtime hot paths (vectorization, per-frame
   allocation, plotting), startup cost, API latency, caching, IPC, and
   Rust-acceleration opportunities.
3. **Proximal–distal modeling** — critical technical review of the
   energy-transfer research program (epics #8557/#8668/#8684):
   assumptions, energy accounting, friction modeling, cross-engine
   parity, experimental design, claim governance.

Dedup notes: `.scratch/uiux-issues/` holds 28 pre-drafted web-app UI/UX
issues; at least drafts 01–03 were filed as #7415–#7417 (closed). New
findings were checked against those drafts and against open issues
before filing.

## Findings and Filed Issues

_(Populated as the review proceeds; each row is a filed GitHub issue.)_

| # | Issue | Area | Severity | Title |
|---|-------|------|----------|-------|
| 1 | #8879 | UX tools | Critical | Pose Studio crashes on construction (act_undo used before create_menu_bar) |
| 2 | #8880 | UX tools | High | Every simulation runs on the GUI thread; no progress, no cancel |
| 3 | #8881 | UX tools | High | Launch Monitor New Project destroys unsaved work; dirty flag reset defeats close guard |
| 4 | #8882 | UX tools | High | Pose Studio Save/Load are tooltip stubs; is_dirty hardcoded False; no closeEvent |
| 5 | #8883 | UX tools | High | Video Analyzer tile opens a placeholder label but is advertised ready |
| 6 | #8884 | UX tools | High | Training Controller swallows cancel/pause/resume failures; no cancel confirm |
| 7 | #8885 | UX tools | Med-High | 45 hardcoded stylesheets bypass the theme; four colors for the run button |
| 8 | #8886 | UX tools | Medium | Units inconsistent across tools and mixed in result panes |
| 9 | #8887 | UX tools | Medium | Pose Studio ±180° limits regardless of engine; silent edit rejection |
| 10 | #8888 | UX tools | Medium | Model Explorer per-engine exports write identical files |
| 11 | #8889 | UX tools | Medium | Pose Matcher Clear overrides: no confirm + stale events without xlsx path |
| 12 | #8890 | UX tools | Medium | Terrain Engine/Model Explorer raise bare exceptions from Qt slots |
| 13 | #8891 | UX web | Medium | Cross-Engine dashboard accepts dt=0; polls forever with no cancel |
| 14 | #8892 | UX web | Medium | Web Settings discards unsaved edits on navigation |
| 15 | #8893 | UX tools | Low-Med | Golf Environment unlabeled mock trajectory; embed drops selector |
| 16 | #8894 | UX launcher | Critical | Shared theme import broken; THEME_AVAILABLE always False in 5 widgets |
| 17 | #8895 | UX launcher | High | Docker dialog Close bypasses build-cancel guard; untimed wait() freeze |
| 18 | #8896 | UX launcher | High | Settings mixes live-apply and Apply-required tabs; Close discards edits |
| 19 | #8897 | UX launcher | High | Layout: Locked caption never updates; Edit Tiles disabled without reason |
| 20 | #8898 | UX launcher | High | Check Windows Dependencies imports pydrake in the clicked slot (5-15 s freeze) |
| 21 | #8899 | UX launcher | High | Embedded-host workspace never persisted; state_snapshot/restore_state dead |
| 22 | #8900 | UX launcher | High | Toasts unbounded, undismissable, window-detached, color-only type coding |
| 23 | #8901 | UX launcher | High | Tile Favourite/Info hover-only 18px NoFocus; arrow-key nav documented but absent |
| 24 | #8902 | UX launcher | Med-High | Two shortcut dialogs both wrong (unbound list vs "(shortcut)" labels) |
| 25 | #8903 | UX launcher | Med-High | WSL checkbox and engine probes run blocking subprocesses on GUI thread |
| 26 | #8904 | UX launcher | Medium | Integrations Health: failure=empty, sticky Copied!, no empty state, contrast |
| 27 | #8905 | UX launcher | Medium | Empty filter results blank; per-keystroke full grid rebuild with smooth rescale |
| 28 | #8906 | UX launcher | Medium | Skeleton cards animate nothing; forever timers |
| 29 | #8907 | UX launcher | Medium | Five settings stores incl. dead branding; no window geometry persistence |
| 30 | #8908 | UX launcher | Low-Med | Batch: unnamed modal, dark table in QMessageBox, global Esc, jargon, overlay resize, processEvents |

## Status

Review in progress. If usage runs out, the filed issues are the durable
output; the table above is the index.

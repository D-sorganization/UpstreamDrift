## Problem

When the Tauri window closes, the Python backend process spawned by the Rust side keeps running. On relaunch the backend port can be occupied and startup fails (or the user accumulates orphaned backend processes). Found during the UI/UX architecture review of `ui/src-tauri/src/lib.rs` — there is no `on_window_event`/exit handler that kills the spawned backend child.

## Fix

1. In `ui/src-tauri/src/lib.rs`, hold the spawned backend `Child` in managed state (if not already) and kill it on shutdown:
   ```rust
   .on_window_event(|window, event| {
     if let tauri::WindowEvent::Destroyed = event {
       if let Some(state) = window.app_handle().try_state::<BackendProcess>() {
         if let Some(mut child) = state.0.lock().unwrap().take() {
           let _ = child.kill();
         }
       }
     }
   })
   ```
   Also handle `RunEvent::ExitRequested` for multi-window safety.
2. Prefer a graceful shutdown first if the backend exposes one (POST `/api/shutdown` with a short timeout), then `kill()` as fallback.
3. Defensive startup: if the port is already bound at launch, detect it and either reuse the healthy backend (health-check endpoint) or pick a free port — at minimum show a clear error instead of a silent failure.
4. Manual verification on Windows: launch app → Task Manager shows backend python → close window → process exits within ~2s; relaunch succeeds.

## Acceptance criteria

- No orphaned backend process after closing the app; relaunch always succeeds.

Part of the UI/UX overhaul epic (see tracking issue).

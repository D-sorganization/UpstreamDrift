use serde::Serialize;
use std::process::Command;
use std::sync::Mutex;
use std::time::Duration;
use tauri::{Manager, State};

mod backend_supervisor;
mod health_probe;

use backend_supervisor::{BackendState, BackendSupervisor};
use health_probe::probe_backend_health;

/// Managed state holding the Python backend server process.
struct BackendProcess(Mutex<BackendSupervisor>);

#[derive(Serialize, Clone)]
struct BackendStatus {
    running: bool,
    pid: Option<u32>,
    port: u16,
    state: String,
    error: Option<String>,
    stderr_tail: Option<String>,
}

fn backend_state_label(state: &BackendState) -> String {
    match state {
        BackendState::NotStarted => "not_started",
        BackendState::Starting { .. } => "starting",
        BackendState::Ready { .. } => "ready",
        BackendState::Exited { .. } => "exited",
        BackendState::Failed { .. } => "failed",
        BackendState::Stopping => "stopping",
    }
    .to_string()
}

fn backend_status_from(supervisor: &BackendSupervisor) -> BackendStatus {
    let state_label = backend_state_label(&supervisor.state);
    match &supervisor.state {
        BackendState::Ready { pid, port } => BackendStatus {
            running: true,
            pid: Some(*pid),
            port: *port,
            state: state_label,
            error: None,
            stderr_tail: None,
        },
        BackendState::Starting { port } => BackendStatus {
            running: false,
            pid: supervisor.child.as_ref().map(std::process::Child::id),
            port: *port,
            state: state_label,
            error: None,
            stderr_tail: None,
        },
        BackendState::Exited { code, stderr_tail } => BackendStatus {
            running: false,
            pid: None,
            port: supervisor.running_port(),
            state: state_label,
            error: Some(format!("Backend exited with code: {code}")),
            stderr_tail: Some(stderr_tail.clone()),
        },
        BackendState::Failed { reason } => BackendStatus {
            running: false,
            pid: None,
            port: supervisor.running_port(),
            state: state_label,
            error: Some(reason.clone()),
            stderr_tail: Some(supervisor.stderr_tail_text()),
        },
        _ => BackendStatus {
            running: false,
            pid: None,
            port: supervisor.running_port(),
            state: state_label,
            error: None,
            stderr_tail: None,
        },
    }
}

#[derive(Serialize, Clone)]
struct DiagnosticInfo {
    backend: BackendStatus,
    python_found: bool,
    python_version: Option<String>,
    repo_root: Option<String>,
    local_server_found: bool,
}

/// Find the repository root by walking up from the Tauri binary location.
fn find_repo_root() -> Option<std::path::PathBuf> {
    // In dev mode, the binary is at ui/src-tauri/target/debug/
    // Walk up looking for a directory containing "src/api/local_server.py"
    let exe = std::env::current_exe().ok()?;
    let mut dir = exe.parent()?;
    for _ in 0..8 {
        let candidate = dir.join("src").join("api").join("local_server.py");
        if candidate.exists() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
    // Also check CWD
    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join("src").join("api").join("local_server.py");
        if candidate.exists() {
            return Some(cwd);
        }
        // Check parent dirs from CWD
        let mut dir = cwd.as_path();
        for _ in 0..5 {
            if let Some(parent) = dir.parent() {
                let candidate = parent.join("src").join("api").join("local_server.py");
                if candidate.exists() {
                    return Some(parent.to_path_buf());
                }
                dir = parent;
            }
        }
    }
    None
}

/// Detect available Python command.
fn find_python() -> Option<(String, String)> {
    for cmd in &["python", "python3", "py"] {
        let output = Command::new(cmd).arg("--version").output();
        if let Ok(out) = output {
            if out.status.success() {
                let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
                let version = if version.is_empty() {
                    String::from_utf8_lossy(&out.stderr).trim().to_string()
                } else {
                    version
                };
                return Some((cmd.to_string(), version));
            }
        }
    }
    None
}

fn lock_supervisor(
    process: &BackendProcess,
) -> Result<std::sync::MutexGuard<'_, BackendSupervisor>, String> {
    process.0.lock().map_err(|e| e.to_string())
}

/// Poll child state while holding the mutex briefly; run HTTP health probes outside it.
fn supervisor_refresh(process: &BackendProcess) -> Result<(), String> {
    let probe_port = {
        let mut guard = lock_supervisor(process)?;
        guard.poll();
        if matches!(guard.state, BackendState::Ready { .. }) {
            return Ok(());
        }
        guard.starting_port()
    };
    let Some(port) = probe_port else {
        return Ok(());
    };
    let probe = probe_backend_health(port);
    let mut guard = lock_supervisor(process)?;
    guard.apply_health_probe(probe);
    guard.poll();
    Ok(())
}

fn backend_is_ready(process: &BackendProcess) -> Result<bool, String> {
    supervisor_refresh(process)?;
    let guard = lock_supervisor(process)?;
    Ok(matches!(guard.state, BackendState::Ready { .. }))
}

fn snapshot_backend_status(process: &BackendProcess) -> Result<BackendStatus, String> {
    let guard = lock_supervisor(process)?;
    Ok(backend_status_from(&guard))
}

/// Start the Python backend server.
#[tauri::command]
fn start_backend(state: State<'_, BackendProcess>) -> Result<BackendStatus, String> {
    {
        let mut guard = lock_supervisor(&state)?;
        guard.poll();
        let already_running = matches!(
            guard.state,
            BackendState::Ready { .. } | BackendState::Starting { .. }
        );
        if already_running {
            drop(guard);
            supervisor_refresh(&state)?;
            return snapshot_backend_status(&state);
        }

        let repo_root = find_repo_root().ok_or("Could not find repository root")?;
        let (python_cmd, _) = find_python().ok_or("Python not found on PATH")?;
        let script_path = repo_root.join("src").join("api").join("local_server.py");
        if !script_path.exists() {
            return Err(format!(
                "Server script not found: {}",
                script_path.display()
            ));
        }

        log::info!(
            "Starting backend: {} {} (cwd: {})",
            python_cmd,
            script_path.display(),
            repo_root.display()
        );

        guard.start(&python_cmd, &script_path, &repo_root)?;
    }

    for _ in 0..25 {
        if backend_is_ready(&state)? {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    supervisor_refresh(&state)?;
    snapshot_backend_status(&state)
}

/// Kill the managed backend child if one is running. Shared by the
/// `stop_backend` command and the window-close / app-exit shutdown hooks
/// (issue #7436) so the Python process never outlives the app.
fn kill_backend(process: &BackendProcess) {
    // Lock defensively: a poisoned mutex must not prevent shutdown.
    let mut guard = match process.0.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    let _ = guard.stop();
    log::info!("Backend server stopped");
}

/// Stop the Python backend server.
#[tauri::command]
fn stop_backend(state: State<'_, BackendProcess>) -> Result<BackendStatus, String> {
    kill_backend(&state);
    snapshot_backend_status(&state)
}

/// Get current backend status.
#[tauri::command]
fn backend_status(state: State<'_, BackendProcess>) -> Result<BackendStatus, String> {
    supervisor_refresh(&state)?;
    snapshot_backend_status(&state)
}

/// Get comprehensive diagnostic information.
#[tauri::command]
fn get_diagnostics(state: State<'_, BackendProcess>) -> Result<DiagnosticInfo, String> {
    supervisor_refresh(&state)?;
    let backend = snapshot_backend_status(&state)?;
    let python_info = find_python();
    let repo_root = find_repo_root();

    let local_server_found = repo_root
        .as_ref()
        .map(|r| r.join("src").join("api").join("local_server.py").exists())
        .unwrap_or(false);

    Ok(DiagnosticInfo {
        backend,
        python_found: python_info.is_some(),
        python_version: python_info.map(|(_, v)| v),
        repo_root: repo_root.map(|p| p.to_string_lossy().to_string()),
        local_server_found,
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(BackendProcess(Mutex::new(BackendSupervisor::default())))
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            // Auto-start the Python backend when the app window opens (issue #6637).
            // Errors are logged rather than propagated so a missing Python install
            // does not hard-crash the UI — the diagnostic screen will surface it.
            let state = app.state::<BackendProcess>();
            match start_backend(state) {
                Ok(status) => log::info!(
                    "Backend auto-start: running={} pid={:?}",
                    status.running,
                    status.pid
                ),
                Err(e) => log::warn!(
                    "Backend auto-start failed (diagnostic screen will show details): {}",
                    e
                ),
            }
            Ok(())
        })
        // #7436: when a window is destroyed, kill the spawned Python backend so
        // it does not outlive the app and block the port on relaunch.
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(process) = window.app_handle().try_state::<BackendProcess>() {
                    kill_backend(&process);
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            start_backend,
            stop_backend,
            backend_status,
            get_diagnostics,
        ])
        .build(tauri::generate_context!())
        .expect("error while running Golf Modeling Suite")
        // #7436: belt-and-suspenders for multi-window / non-window exits —
        // ensure the backend is killed on any app exit, not just a window
        // Destroyed event.
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(process) = app_handle.try_state::<BackendProcess>() {
                    kill_backend(&process);
                }
            }
        });
}

#[cfg(test)]
mod version_sync_tests {
    #[test]
    fn cargo_and_tauri_versions_match() {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("cargo");
        let cargo_ver = manifest
            .lines()
            .find(|line| line.trim().starts_with("version ="))
            .and_then(|line| line.split('"').nth(1))
            .expect("cargo version");
        let conf = std::fs::read_to_string(manifest_dir.join("tauri.conf.json")).expect("conf");
        let parsed: serde_json::Value = serde_json::from_str(&conf).expect("json");
        assert_eq!(cargo_ver, parsed["version"].as_str().unwrap());
    }
}

use serde::Serialize;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::State;

/// Managed state holding the Python backend server process.
struct BackendProcess(Mutex<Option<Child>>);

#[derive(Serialize, Clone)]
struct BackendStatus {
    running: bool,
    pid: Option<u32>,
    port: u16,
    error: Option<String>,
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

/// Start the Python backend server.
#[tauri::command]
fn start_backend(state: State<'_, BackendProcess>) -> Result<BackendStatus, String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;

    // Check if already running
    if let Some(ref mut child) = *guard {
        match child.try_wait() {
            Ok(None) => {
                return Ok(BackendStatus {
                    running: true,
                    pid: Some(child.id()),
                    port: 8000,
                    error: None,
                });
            }
            _ => {
                // Process exited, clear it
                *guard = None;
            }
        }
    }

    let repo_root = find_repo_root().ok_or("Could not find repository root")?;
    let (python_cmd, _) = find_python().ok_or("Python not found on PATH")?;

    let server_script = repo_root.join("launch_golf_suite.py");
    let script_path = if server_script.exists() {
        server_script
    } else {
        // Fallback: run local_server directly
        repo_root.join("src").join("api").join("local_server.py")
    };

    if !script_path.exists() {
        return Err(format!("Server script not found: {}", script_path.display()));
    }

    log::info!(
        "Starting backend: {} {} (cwd: {})",
        python_cmd,
        script_path.display(),
        repo_root.display()
    );

    let child = Command::new(&python_cmd)
        .arg(&script_path)
        .arg("--api-only")
        .current_dir(&repo_root)
        .env("PYTHONPATH", &repo_root)
        .spawn()
        .map_err(|e| format!("Failed to start backend: {}", e))?;

    let pid = child.id();
    *guard = Some(child);

    Ok(BackendStatus {
        running: true,
        pid: Some(pid),
        port: 8000,
        error: None,
    })
}

/// Stop the Python backend server.
#[tauri::command]
fn stop_backend(state: State<'_, BackendProcess>) -> Result<BackendStatus, String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;

    if let Some(ref mut child) = *guard {
        let _ = child.kill();
        let _ = child.wait();
        log::info!("Backend server stopped");
    }
    *guard = None;

    Ok(BackendStatus {
        running: false,
        pid: None,
        port: 8000,
        error: None,
    })
}

/// Get current backend status.
#[tauri::command]
fn backend_status(state: State<'_, BackendProcess>) -> Result<BackendStatus, String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;

    if let Some(ref mut child) = *guard {
        match child.try_wait() {
            Ok(None) => {
                return Ok(BackendStatus {
                    running: true,
                    pid: Some(child.id()),
                    port: 8000,
                    error: None,
                });
            }
            Ok(Some(status)) => {
                let error = if status.success() {
                    None
                } else {
                    Some(format!("Backend exited with code: {:?}", status.code()))
                };
                *guard = None;
                return Ok(BackendStatus {
                    running: false,
                    pid: None,
                    port: 8000,
                    error,
                });
            }
            Err(e) => {
                return Ok(BackendStatus {
                    running: false,
                    pid: None,
                    port: 8000,
                    error: Some(format!("Failed to check backend status: {}", e)),
                });
            }
        }
    }

    Ok(BackendStatus {
        running: false,
        pid: None,
        port: 8000,
        error: None,
    })
}

/// Get comprehensive diagnostic information.
#[tauri::command]
fn get_diagnostics(state: State<'_, BackendProcess>) -> Result<DiagnosticInfo, String> {
    let backend = backend_status(state)?;
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
        .manage(BackendProcess(Mutex::new(None)))
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_backend,
            stop_backend,
            backend_status,
            get_diagnostics,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Golf Modeling Suite");
}

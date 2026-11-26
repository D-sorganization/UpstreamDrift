use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;

use crate::health_probe::HealthProbeResult;

pub const STDERR_TAIL_LINES: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackendState {
    NotStarted,
    Starting { port: u16 },
    Ready { pid: u32, port: u16 },
    Exited { code: i32, stderr_tail: String },
    Failed { reason: String },
    Stopping,
}

#[derive(Debug)]
pub struct BackendSupervisor {
    pub state: BackendState,
    pub(crate) child: Option<Child>,
    port: u16,
    stderr_tail: VecDeque<String>,
    stderr_rx: Option<mpsc::Receiver<String>>,
}

impl Default for BackendSupervisor {
    fn default() -> Self {
        Self {
            state: BackendState::NotStarted,
            child: None,
            port: 8000,
            stderr_tail: VecDeque::new(),
            stderr_rx: None,
        }
    }
}

pub fn pick_ephemeral_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("could not bind ephemeral port: {e}"))?;
    listener
        .local_addr()
        .map(|addr| addr.port())
        .map_err(|e| format!("could not read ephemeral port: {e}"))
}

impl BackendSupervisor {
    pub fn append_stderr_line(&mut self, line: String) {
        self.stderr_tail.push_back(line);
        while self.stderr_tail.len() > STDERR_TAIL_LINES {
            self.stderr_tail.pop_front();
        }
    }

    pub fn stderr_tail_text(&self) -> String {
        self.stderr_tail
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn drain_stderr_channel(&mut self) {
        let Some(rx) = self.stderr_rx.as_ref() else {
            return;
        };
        let mut lines = Vec::new();
        while let Ok(line) = rx.try_recv() {
            lines.push(line);
        }
        for line in lines {
            self.append_stderr_line(line);
        }
    }

    /// Fast poll: child exit + stderr only (no HTTP probe — callers probe outside the mutex).
    pub fn poll(&mut self) {
        self.drain_stderr_channel();
        let Some(child) = self.child.as_mut() else {
            return;
        };
        match child.try_wait() {
            Ok(Some(status)) => {
                let code = status.code().unwrap_or(-1);
                self.stderr_tail
                    .push_back(format!("exited with code {code}"));
                self.state = BackendState::Exited {
                    code,
                    stderr_tail: self.stderr_tail_text(),
                };
                self.child = None;
                self.stderr_rx = None;
            }
            Ok(None) => {}
            Err(err) => {
                self.state = BackendState::Failed {
                    reason: format!("Failed to poll backend child: {err}"),
                };
            }
        }
    }

    pub fn apply_health_probe(&mut self, probe: HealthProbeResult) {
        if !probe.is_ready() {
            return;
        }
        let BackendState::Starting { port } = self.state else {
            return;
        };
        let Some(child) = self.child.as_ref() else {
            return;
        };
        self.state = BackendState::Ready {
            pid: child.id(),
            port,
        };
    }

    pub fn starting_port(&self) -> Option<u16> {
        if let BackendState::Starting { port } = self.state {
            Some(port)
        } else {
            None
        }
    }

    pub fn start(
        &mut self,
        python_cmd: &str,
        script_path: &std::path::Path,
        repo_root: &std::path::Path,
    ) -> Result<&BackendState, String> {
        self.poll();
        if matches!(
            self.state,
            BackendState::Ready { .. } | BackendState::Starting { .. }
        ) {
            return Ok(&self.state);
        }
        let port = pick_ephemeral_port()?;
        self.port = port;
        self.state = BackendState::Starting { port };

        let mut child = Command::new(python_cmd)
            .arg(script_path)
            .current_dir(repo_root)
            .env("PYTHONPATH", repo_root)
            .env("GOLF_PORT", port.to_string())
            .env("PYTHONUNBUFFERED", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                self.state = BackendState::Failed {
                    reason: format!("Failed to start backend on port {port}: {e}"),
                };
                format!("Failed to start backend: {e}")
            })?;

        let stderr = child.stderr.take();
        if let Some(stderr_handle) = stderr {
            let (tx, rx) = mpsc::channel::<String>();
            self.stderr_rx = Some(rx);
            thread::spawn(move || {
                use std::io::BufRead;
                let reader = std::io::BufReader::new(stderr_handle);
                for line in reader.lines().map_while(Result::ok) {
                    let _ = tx.send(line);
                }
            });
        }

        self.child = Some(child);
        Ok(&self.state)
    }

    pub fn stop(&mut self) -> BackendState {
        self.state = BackendState::Stopping;
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.stderr_rx = None;
        self.state = BackendState::NotStarted;
        self.state.clone()
    }

    pub fn running_port(&self) -> u16 {
        self.port
    }
}

pub fn spawn_immediate_exit_child() -> std::io::Result<Child> {
    if cfg!(windows) {
        Command::new("cmd").args(["/C", "exit", "0"]).spawn()
    } else {
        Command::new("sh").args(["-c", "exit 0"]).spawn()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ephemeral_port_is_nonzero() {
        let port = pick_ephemeral_port().expect("port");
        assert!(port > 0);
    }

    #[test]
    fn stderr_tail_is_bounded() {
        let mut supervisor = BackendSupervisor::default();
        for index in 0..50 {
            supervisor.append_stderr_line(format!("line-{index}"));
        }
        assert_eq!(supervisor.stderr_tail.len(), STDERR_TAIL_LINES);
        assert!(supervisor.stderr_tail_text().contains("line-49"));
    }

    #[test]
    fn state_transitions_to_exited_when_child_exits() {
        let mut supervisor = BackendSupervisor::default();
        let child = spawn_immediate_exit_child().expect("spawn immediate exit");
        supervisor.child = Some(child);
        supervisor.port = pick_ephemeral_port().expect("port");
        supervisor.state = BackendState::Starting {
            port: supervisor.port,
        };
        for _ in 0..20 {
            supervisor.poll();
            if matches!(supervisor.state, BackendState::Exited { .. }) {
                break;
            }
            thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(matches!(supervisor.state, BackendState::Exited { .. }));
    }

    #[test]
    fn health_probe_false_for_closed_port() {
        let port = pick_ephemeral_port().expect("port");
        assert!(!crate::health_probe::probe_backend_health(port).is_ready());
    }
}

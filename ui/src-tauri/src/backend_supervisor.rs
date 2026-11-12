use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

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
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("could not bind ephemeral port: {e}"))?;
    listener
        .local_addr()
        .map(|addr| addr.port())
        .map_err(|e| format!("could not read ephemeral port: {e}"))
}

pub fn health_ready(port: u16) -> bool {
    use std::net::ToSocketAddrs;
    let Ok(mut addrs) = format!("127.0.0.1:{port}").to_socket_addrs() else {
        return false;
    };
    let Some(socket_addr) = addrs.next() else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&socket_addr, Duration::from_millis(400))
    else {
        return false;
    };
    let request =
        format!("GET /api/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut buffer = [0_u8; 2048];
    let Ok(read_bytes) = stream.read(&mut buffer) else {
        return false;
    };
    let body = String::from_utf8_lossy(&buffer[..read_bytes]);
    body.contains("healthy")
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
            Ok(None) => {
                if matches!(self.state, BackendState::Starting { .. }) && health_ready(self.port) {
                    self.state = BackendState::Ready {
                        pid: child.id(),
                        port: self.port,
                    };
                }
            }
            Err(err) => {
                self.state = BackendState::Failed {
                    reason: format!("Failed to poll backend child: {err}"),
                };
            }
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
    fn state_transitions_to_exited_when_child_drops() {
        let mut supervisor = BackendSupervisor::default();
        let child = Command::new("true").spawn().expect("spawn true");
        supervisor.child = Some(child);
        supervisor.port = 9_999;
        supervisor.state = BackendState::Starting { port: 9_999 };
        supervisor.poll();
        assert!(matches!(supervisor.state, BackendState::Exited { .. }));
    }

    #[test]
    fn health_ready_false_for_closed_port() {
        let port = pick_ephemeral_port().expect("port");
        assert!(!health_ready(port));
    }
}

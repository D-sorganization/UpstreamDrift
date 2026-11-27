use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant};

pub const HEALTH_PROBE_BUDGET: Duration = Duration::from_millis(500);
const CONNECT_TIMEOUT: Duration = Duration::from_millis(400);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthProbeResult {
    Ready,
    NotReady,
    TransportError,
}

impl HealthProbeResult {
    pub fn is_ready(self) -> bool {
        matches!(self, HealthProbeResult::Ready)
    }
}

fn remaining_budget(start: Instant, budget: Duration) -> Option<Duration> {
    budget.checked_sub(start.elapsed())
}

fn set_timeouts(stream: &TcpStream, remaining: Duration) -> Result<(), std::io::Error> {
    stream.set_read_timeout(Some(remaining))?;
    stream.set_write_timeout(Some(remaining))?;
    Ok(())
}

fn read_http_response(
    stream: &mut TcpStream,
    start: Instant,
    budget: Duration,
) -> Result<Vec<u8>, std::io::Error> {
    let mut buffer = Vec::new();
    let mut scratch = [0_u8; 1024];
    loop {
        let Some(remaining) = remaining_budget(start, budget) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "health probe budget exhausted",
            ));
        };
        set_timeouts(stream, remaining)?;
        match stream.read(&mut scratch) {
            Ok(0) => break,
            Ok(n) => buffer.extend_from_slice(&scratch[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "health probe read timed out",
                ));
            }
            Err(e) => return Err(e),
        }
        if let Some(idx) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            let header_end = idx + 4;
            let headers = &buffer[..header_end];
            let body_start = header_end;
            if let Some(body_len) = parse_content_length(headers) {
                if buffer.len() >= body_start + body_len {
                    break;
                }
            } else if !buffer[body_start..].is_empty() {
                // Chunked/close-delimited fallbacks: another read or EOF ends the loop.
                continue;
            }
        }
    }
    Ok(buffer)
}

fn parse_content_length(headers: &[u8]) -> Option<usize> {
    let text = String::from_utf8_lossy(headers);
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            return rest.trim().parse().ok();
        }
    }
    None
}

fn parse_status_code(headers: &[u8]) -> Option<u16> {
    let text = String::from_utf8_lossy(headers);
    let first = text.lines().next()?;
    let mut parts = first.split_whitespace();
    let _http = parts.next()?;
    parts.next()?.parse().ok()
}

fn split_headers_body(response: &[u8]) -> Option<(&[u8], &[u8])> {
    let idx = response.windows(4).position(|w| w == b"\r\n\r\n")?;
    let header_end = idx + 4;
    Some((&response[..header_end], &response[header_end..]))
}

fn validate_health_json(body: &[u8], expected_port: u16) -> bool {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
        return false;
    };
    let Some(status) = value.get("status").and_then(|v| v.as_str()) else {
        return false;
    };
    if status != "healthy" {
        return false;
    }
    let Some(bind_port) = value.get("bind_port").and_then(|v| v.as_u64()) else {
        return false;
    };
    bind_port == u64::from(expected_port)
}

/// Probe ``GET /api/health`` with a strict deadline, status code, JSON body, and port identity.
pub fn probe_backend_health(expected_port: u16) -> HealthProbeResult {
    let start = Instant::now();
    let addr: SocketAddr = format!("127.0.0.1:{expected_port}")
        .parse()
        .unwrap_or_else(|_| SocketAddr::from(([127, 0, 0, 1], expected_port)));
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) else {
        return HealthProbeResult::TransportError;
    };
    let Some(remaining) = remaining_budget(start, HEALTH_PROBE_BUDGET) else {
        return HealthProbeResult::TransportError;
    };
    if set_timeouts(&stream, remaining).is_err() {
        return HealthProbeResult::TransportError;
    }
    let request = format!(
        "GET /api/health HTTP/1.1\r\nHost: 127.0.0.1:{expected_port}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return HealthProbeResult::TransportError;
    }
    let Ok(response) = read_http_response(&mut stream, start, HEALTH_PROBE_BUDGET) else {
        return HealthProbeResult::TransportError;
    };
    let Some((headers, body)) = split_headers_body(&response) else {
        return HealthProbeResult::NotReady;
    };
    let Some(code) = parse_status_code(headers) else {
        return HealthProbeResult::NotReady;
    };
    if code != 200 {
        return HealthProbeResult::NotReady;
    }
    if validate_health_json(body, expected_port) {
        HealthProbeResult::Ready
    } else {
        HealthProbeResult::NotReady
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    fn spawn_stub_server(
        write_body: impl FnOnce(u16, &mut TcpStream) + Send + 'static,
    ) -> Result<u16, String> {
        let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            let mut stream = loop {
                if Instant::now() > deadline {
                    let _ = tx.send(());
                    return;
                }
                match listener.accept() {
                    Ok((mut s, _)) => break s,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => {
                        let _ = tx.send(());
                        return;
                    }
                }
            };
            let mut junk = [0_u8; 512];
            let _ = stream.read(&mut junk);
            write_body(port, &mut stream);
            let _ = stream.shutdown(std::net::Shutdown::Write);
            let _ = tx.send(());
        });
        Ok(port)
    }

    #[test]
    fn rejects_503_even_when_body_contains_healthy_substring() {
        let port = spawn_stub_server(|_port, stream| {
            let body = r#"{"status":"unhealthy","detail":"still starting"}"#;
            let response = format!(
                "HTTP/1.1 503 Service Unavailable\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        })
        .expect("server");
        assert_eq!(probe_backend_health(port), HealthProbeResult::NotReady);
    }

    #[test]
    fn accepts_body_that_arrives_after_headers() {
        let port = spawn_stub_server(|port, stream| {
            let body = format!(r#"{{"status":"healthy","mode":"local","bind_port":{port}}}"#);
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(headers.as_bytes());
            thread::sleep(Duration::from_millis(40));
            let _ = stream.write_all(body.as_bytes());
        })
        .expect("server");
        assert_eq!(probe_backend_health(port), HealthProbeResult::Ready);
    }

    #[test]
    fn rejects_slow_peer_beyond_probe_budget() {
        let port = spawn_stub_server(|port, stream| {
            let body = format!(r#"{{"status":"healthy","mode":"local","bind_port":{port}}}"#);
            thread::sleep(Duration::from_millis(800));
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        })
        .expect("server");
        assert_eq!(
            probe_backend_health(port),
            HealthProbeResult::TransportError
        );
    }

    #[test]
    fn requires_bind_port_instance_identity() {
        let port = spawn_stub_server(|_port, stream| {
            let wrong_body = r#"{"status":"healthy","mode":"local","bind_port":1}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{wrong_body}",
                wrong_body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        })
        .expect("server");
        assert_eq!(probe_backend_health(port), HealthProbeResult::NotReady);
    }
}

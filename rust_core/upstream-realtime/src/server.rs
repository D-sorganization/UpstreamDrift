//! Tokio + tokio-tungstenite WebSocket pub-sub server.
//!
//! The server exposes a single endpoint, `GET /realtime/subscribe?channel=...`,
//! that upgrades to WebSocket and forwards every message published on the
//! channel as a text frame. Publishes happen in-process via [`Server::publish`]
//! rather than over HTTP (the Rust path doesn't host an HTTP API surface —
//! the Python facade calls `publish` directly).
//!
//! ## Lifecycle
//!
//! * [`Server::start`] binds, spawns the accept loop, returns a
//!   [`ServerHandle`].
//! * [`ServerHandle::publish`] is the hot path (no awaits in the Python ABI).
//! * [`ServerHandle::subscribe_local`] returns an in-process subscriber that
//!   bypasses the WebSocket entirely — used by the Python facade so a single
//!   process can publish and consume without going out to the loopback
//!   interface.
//! * [`ServerHandle::stop`] cancels the accept loop and closes all sockets.

use crate::channels::{validate_channel, ChannelRegistry};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::runtime::Runtime;
use tokio::sync::broadcast;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;

/// Owns the Tokio runtime and the channel registry. The Python facade keeps
/// exactly one `Server` per process.
pub struct Server {
    runtime: Arc<Runtime>,
    registry: Arc<ChannelRegistry>,
}

/// Handle to a started server. Cheap to clone.
#[derive(Clone)]
pub struct ServerHandle {
    runtime: Arc<Runtime>,
    registry: Arc<ChannelRegistry>,
    bound_addr: SocketAddr,
    shutdown: Arc<tokio::sync::Mutex<Option<oneshot::Sender<()>>>>,
    accept_task: Arc<tokio::sync::Mutex<Option<JoinHandle<()>>>>,
}

impl Server {
    /// Construct a new server with its own multi-threaded Tokio runtime.
    pub fn new() -> io::Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("upstream-realtime")
            .build()?;
        Ok(Self {
            runtime: Arc::new(runtime),
            registry: Arc::new(ChannelRegistry::new()),
        })
    }

    /// Bind to `host:port` and start the accept loop.
    pub fn start(self, host: &str, port: u16) -> io::Result<ServerHandle> {
        let registry = self.registry.clone();
        let runtime = self.runtime.clone();

        // Bind synchronously so callers see bind errors immediately.
        let addr: SocketAddr = format!("{host}:{port}")
            .parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("{e}")))?;
        let listener = runtime.block_on(async { TcpListener::bind(addr).await })?;
        let bound_addr = listener.local_addr()?;

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let registry_for_task = registry.clone();
        let accept = runtime.spawn(async move {
            accept_loop(listener, registry_for_task, shutdown_rx).await;
        });

        Ok(ServerHandle {
            runtime,
            registry,
            bound_addr,
            shutdown: Arc::new(tokio::sync::Mutex::new(Some(shutdown_tx))),
            accept_task: Arc::new(tokio::sync::Mutex::new(Some(accept))),
        })
    }
}

impl ServerHandle {
    /// Bound address (host:port) for the listener.
    pub fn bound_addr(&self) -> SocketAddr {
        self.bound_addr
    }

    /// Publish a payload to all subscribers (both WebSocket and in-process).
    /// Returns the number of receivers at the moment of send.
    pub fn publish(&self, channel: &str, payload_json: String) -> Result<usize, String> {
        validate_channel(channel)?;
        Ok(self.registry.publish(channel, payload_json))
    }

    /// Subscribe in-process — no socket, no asyncio. The returned
    /// [`Subscriber`] is `Send + Sync` and exposes a blocking `recv`.
    pub fn subscribe_local(&self, channel: &str) -> Result<Subscriber, String> {
        validate_channel(channel)?;
        let rx = self.registry.subscribe(channel);
        Ok(Subscriber::new(self.runtime.clone(), rx))
    }

    /// Stop accepting new connections and wait for the accept task to exit.
    pub fn stop(&self) {
        let mut guard = self.runtime.block_on(self.shutdown.lock());
        if let Some(tx) = guard.take() {
            let _ = tx.send(());
        }
        drop(guard);

        let mut task_guard = self.runtime.block_on(self.accept_task.lock());
        if let Some(task) = task_guard.take() {
            let _ = self.runtime.block_on(task);
        }
    }
}

/// Blocking subscriber handle. Holds an Arc to the runtime so it can drive
/// the broadcast receiver from a Python thread without an asyncio loop.
pub struct Subscriber {
    runtime: Arc<Runtime>,
    rx: tokio::sync::Mutex<broadcast::Receiver<String>>,
}

impl Subscriber {
    fn new(runtime: Arc<Runtime>, rx: broadcast::Receiver<String>) -> Self {
        Self {
            runtime,
            rx: tokio::sync::Mutex::new(rx),
        }
    }

    /// Block until a payload arrives or `timeout_secs` elapses. Returns
    /// `Ok(Some(json))` on success, `Ok(None)` on timeout, or `Err` if the
    /// sender side was dropped.
    pub fn recv_blocking(&self, timeout_secs: Option<f64>) -> Result<Option<String>, String> {
        self.runtime.block_on(async {
            let fut = async {
                let mut rx = self.rx.lock().await;
                loop {
                    match rx.recv().await {
                        Ok(payload) => return Ok(Some(payload)),
                        Err(broadcast::error::RecvError::Lagged(_)) => {
                            // Skip dropped messages, keep waiting.
                            continue;
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            return Err("channel closed".to_string());
                        }
                    }
                }
            };
            match timeout_secs {
                None => fut.await,
                Some(t) => {
                    match tokio::time::timeout(std::time::Duration::from_secs_f64(t), fut).await {
                        Ok(res) => res,
                        Err(_) => Ok(None),
                    }
                }
            }
        })
    }
}

async fn accept_loop(
    listener: TcpListener,
    registry: Arc<ChannelRegistry>,
    mut shutdown: oneshot::Receiver<()>,
) {
    loop {
        tokio::select! {
            _ = &mut shutdown => {
                tracing::info!("upstream-realtime: shutdown requested");
                return;
            }
            accept = listener.accept() => {
                match accept {
                    Ok((stream, peer)) => {
                        let reg = registry.clone();
                        tokio::spawn(async move {
                            if let Err(e) = serve_connection(stream, reg).await {
                                tracing::debug!("upstream-realtime: peer {peer} ended: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        tracing::warn!("upstream-realtime: accept failed: {e}");
                    }
                }
            }
        }
    }
}

// The handshake callback returns an `ErrorResponse` on rejection, which the
// `result_large_err` clippy lint flags. Boxing it would force tokio-tungstenite
// to change its signature; the value is short-lived and only constructed on
// the reject path, so we silence the lint here.
#[allow(clippy::result_large_err)]
async fn serve_connection(stream: TcpStream, registry: Arc<ChannelRegistry>) -> Result<(), String> {
    // Read the HTTP request to extract the path + query. tokio-tungstenite
    // exposes `accept_hdr_async` which gives the request; we use it to
    // parse `/realtime/subscribe?channel=...` and validate the channel.
    let mut requested_channel: Option<String> = None;

    let ws_stream = tokio_tungstenite::accept_hdr_async(
        stream,
        |req: &tokio_tungstenite::tungstenite::handshake::server::Request,
         resp: tokio_tungstenite::tungstenite::handshake::server::Response| {
            let uri = req.uri();
            let path = uri.path();
            if path != "/realtime/subscribe" {
                tracing::debug!("upstream-realtime: rejecting path {path}");
                return Err(http_400("unknown path"));
            }
            let query: HashMap<&str, &str> = uri
                .query()
                .map(|q| {
                    q.split('&')
                        .filter_map(|kv| {
                            let mut it = kv.splitn(2, '=');
                            Some((it.next()?, it.next().unwrap_or("")))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let channel = query
                .get("channel")
                .copied()
                .ok_or_else(|| http_400("missing channel"))?;
            if validate_channel(channel).is_err() {
                return Err(http_400("invalid channel"));
            }
            requested_channel = Some(channel.to_string());
            Ok(resp)
        },
    )
    .await
    .map_err(|e| format!("ws handshake failed: {e}"))?;

    let channel = match requested_channel {
        Some(c) => c,
        None => return Err("handshake passed without channel".into()),
    };

    let mut rx = registry.subscribe(&channel);
    let (mut sink, mut source) = ws_stream.split();

    loop {
        tokio::select! {
            // Forward broadcast → WS
            payload = rx.recv() => {
                match payload {
                    Ok(text) => {
                        if sink.send(Message::Text(text)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            // Drain inbound (we don't expect any, but keep the socket healthy)
            incoming = source.next() => {
                match incoming {
                    None => break,
                    Some(Err(_)) => break,
                    Some(Ok(Message::Close(_))) => break,
                    Some(Ok(_)) => continue,
                }
            }
        }
    }

    Ok(())
}

fn http_400(msg: &str) -> tokio_tungstenite::tungstenite::handshake::server::ErrorResponse {
    use tokio_tungstenite::tungstenite::http::{Response, StatusCode};
    let body = msg.to_string();
    Response::builder()
        .status(StatusCode::BAD_REQUEST)
        .body(Some(body))
        .expect("static response is valid")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};

    #[test]
    fn start_publish_subscribe_local_round_trip() {
        let server = Server::new().expect("runtime");
        let handle = server.start("127.0.0.1", 0).expect("bind");
        let sub = handle.subscribe_local("pose/canonical").expect("subscribe");

        // Publish from a separate thread to mimic the Python facade.
        let h = handle.clone();
        let _t = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            let n = h
                .publish("pose/canonical", "{\"frame\":1}".into())
                .expect("publish");
            assert_eq!(n, 1);
        });

        let got = sub
            .recv_blocking(Some(2.0))
            .expect("recv")
            .expect("payload");
        assert_eq!(got, "{\"frame\":1}");
        handle.stop();
    }

    #[test]
    fn websocket_subscribe_receives_published_payload() {
        let server = Server::new().expect("runtime");
        let handle = server.start("127.0.0.1", 0).expect("bind");
        let addr = handle.bound_addr();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let h = handle.clone();
        rt.block_on(async move {
            let url = format!("ws://{addr}/realtime/subscribe?channel=pose/canonical");
            let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
                .await
                .expect("connect");

            // Give the server a tick to register the broadcast subscriber.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;

            let n = h
                .publish("pose/canonical", "{\"hello\":\"ws\"}".into())
                .expect("publish");
            assert!(n >= 1);

            let msg = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
                .await
                .expect("timeout")
                .expect("stream closed")
                .expect("frame err");
            assert_eq!(msg.into_text().unwrap().as_str(), "{\"hello\":\"ws\"}");
            let _ = ws.send(Message::Close(None)).await;
        });
        handle.stop();
    }

    #[test]
    fn bad_channel_rejects_handshake() {
        let server = Server::new().expect("runtime");
        let handle = server.start("127.0.0.1", 0).expect("bind");
        let addr = handle.bound_addr();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let url = format!("ws://{addr}/realtime/subscribe?channel=BAD/Name");
            let res = tokio_tungstenite::connect_async(&url).await;
            assert!(res.is_err(), "expected handshake rejection");
        });
        handle.stop();
    }
}

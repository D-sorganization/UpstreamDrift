//! # upstream-realtime — Tokio WebSocket pub-sub
//!
//! High-performance Rust replacement for the FastAPI/asyncio autostart server
//! used by `src/shared/python/realtime/ws_pubsub.py`. Wire-protocol compatible:
//!
//! - `POST /realtime/publish` body `{channel, payload}` is replaced by an
//!   in-process [`Server::publish`] call from the Rust side; the Python facade
//!   exposes a sync `publish(channel, payload)` that calls into Rust.
//! - `WS /realtime/subscribe?channel=...` — clients connect and receive every
//!   payload published on that channel as a JSON object frame.
//!
//! Acceptance target: < 10 ms median, < 50 ms p99 one-hop. The hot path is
//! pure Rust + Tokio + `tokio::sync::broadcast`; the Python ABI is *sync*
//! (no asyncio) so the GIL is released across the wait.

pub mod channels;
pub mod protocol;
pub mod server;

#[cfg(feature = "python")]
mod bindings;

pub use channels::{validate_channel, ChannelRegistry};
pub use server::{Server, ServerHandle, Subscriber};

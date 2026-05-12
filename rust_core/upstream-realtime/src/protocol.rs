//! Wire-protocol primitives shared between Rust core and the Python facade.
//!
//! Frame format on the wire is identical to the FastAPI server it replaces:
//! a publish delivers the raw JSON `payload` object to every WS subscriber
//! as a single text frame. The `{channel, payload}` envelope is only used
//! by the HTTP `/realtime/publish` route; for the Rust path the channel is
//! a function-call argument and never travels over the WebSocket.

use serde::{Deserialize, Serialize};

/// HTTP `/realtime/publish` body, mirrored from
/// `src/api/routes/realtime.py::PublishRequest`. Retained for compatibility
/// shims and tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishEnvelope {
    pub channel: String,
    #[serde(default)]
    pub payload: serde_json::Value,
}

/// HTTP `/realtime/publish` response, mirrored from the FastAPI handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishAck {
    pub channel: String,
    pub delivered: usize,
}

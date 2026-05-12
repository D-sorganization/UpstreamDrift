//! Channel registry + validation rules.
//!
//! Mirrors `src/shared/python/realtime/protocol.py::validate_channel`. The
//! rule is `^[a-z][a-z0-9_]*(/[a-z0-9_]+)+$` — at least two segments, first
//! starts with a lowercase letter.
//!
//! The registry maps a channel name to a `tokio::sync::broadcast::Sender`
//! so any number of subscribers can fan-out independently with backpressure
//! confined to the slowest consumer's lag window.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Mutex;
use tokio::sync::broadcast;

static CHANNEL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[a-z][a-z0-9_]*(/[a-z0-9_]+)+$").expect("valid regex"));

/// Returns `Ok(())` iff `name` matches the canonical channel pattern.
pub fn validate_channel(name: &str) -> Result<(), String> {
    if !CHANNEL_RE.is_match(name) {
        return Err(format!(
            "invalid channel name {name:?}; must match '^[a-z][a-z0-9_]*(/[a-z0-9_]+)+$' (scope/topic pattern)"
        ));
    }
    Ok(())
}

/// Default per-channel broadcast capacity. Lagging subscribers drop oldest.
/// Sized to comfortably hold a 10k-message burst (the latency benchmark
/// in `tests/unit/realtime/test_latency.py`) without forcing the
/// publisher to pace itself; the Python facade is single-consumer-per-
/// subscriber and drives a dedicated thread that re-reads as fast as
/// `json.loads` + the user callback allow.
pub const DEFAULT_CAPACITY: usize = 65_536;

/// Per-channel broadcast registry. Cheaply cloneable handle; the inner
/// map is wrapped in `Mutex` because contention is one-shot at first
/// publish/subscribe per channel.
#[derive(Debug, Default)]
pub struct ChannelRegistry {
    inner: Mutex<HashMap<String, broadcast::Sender<String>>>,
    capacity: usize,
}

impl ChannelRegistry {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            capacity: DEFAULT_CAPACITY,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            capacity,
        }
    }

    /// Return (or create) the broadcast sender for `channel`.
    pub fn sender(&self, channel: &str) -> broadcast::Sender<String> {
        let mut guard = self.inner.lock().expect("registry mutex poisoned");
        if let Some(tx) = guard.get(channel) {
            return tx.clone();
        }
        let (tx, _rx) = broadcast::channel(self.capacity);
        guard.insert(channel.to_string(), tx.clone());
        tx
    }

    /// Subscribe to `channel`. Returns a fresh receiver.
    pub fn subscribe(&self, channel: &str) -> broadcast::Receiver<String> {
        self.sender(channel).subscribe()
    }

    /// Publish `payload_json` on `channel`. Returns the number of live
    /// subscribers at the moment of send. Mirrors the FastAPI `delivered`
    /// count, with two differences:
    /// * In-process WS subscribers that are connected through this same
    ///   process count too (Rust path is unified).
    /// * Lagging receivers that get dropped by tokio's broadcast still
    ///   count as delivered for the purpose of the ack — the Rust side
    ///   has no equivalent of FastAPI's "send_json may raise."
    pub fn publish(&self, channel: &str, payload_json: String) -> usize {
        let tx = self.sender(channel);
        tx.send(payload_json).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_canonical_names() {
        assert!(validate_channel("pose/canonical").is_ok());
        assert!(validate_channel("engine/drake/state").is_ok());
        assert!(validate_channel("target/active").is_ok());
    }

    #[test]
    fn rejects_bad_names() {
        assert!(validate_channel("BAD/Name").is_err());
        assert!(validate_channel("no-segments").is_err());
        assert!(validate_channel("1leading/digit").is_err());
        assert!(validate_channel("").is_err());
    }

    #[tokio::test]
    async fn registry_round_trip() {
        let reg = ChannelRegistry::new();
        let mut rx = reg.subscribe("pose/canonical");
        let n = reg.publish("pose/canonical", "{\"x\":1}".into());
        assert_eq!(n, 1);
        let got = rx.recv().await.unwrap();
        assert_eq!(got, "{\"x\":1}");
    }

    #[tokio::test]
    async fn registry_isolates_channels() {
        let reg = ChannelRegistry::new();
        let mut rx_a = reg.subscribe("a/one");
        let mut rx_b = reg.subscribe("b/two");
        let _ = reg.publish("a/one", "{\"a\":1}".into());
        let _ = reg.publish("b/two", "{\"b\":2}".into());
        assert_eq!(rx_a.recv().await.unwrap(), "{\"a\":1}");
        assert_eq!(rx_b.recv().await.unwrap(), "{\"b\":2}");
    }
}

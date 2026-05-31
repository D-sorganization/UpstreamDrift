//! PyO3 bindings — exposes a sync `PyServer` + `PySubscriber` to the
//! Python facade. The Rust side owns the Tokio runtime; Python code never
//! sees an event loop.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use crate::server::{Server, ServerHandle, Subscriber};

#[pyclass(name = "Server")]
pub struct PyServer {
    handle: Arc<Mutex<Option<ServerHandle>>>,
}

#[pymethods]
impl PyServer {
    /// Bind to `host:port` and start the accept loop. The returned object
    /// is the handle; the underlying runtime stays alive until `stop()` is
    /// called or the object is dropped.
    #[new]
    #[pyo3(signature = (host="127.0.0.1", port=8765))]
    fn new(host: &str, port: u16) -> PyResult<Self> {
        let server = Server::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let handle = server
            .start(host, port)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            handle: Arc::new(Mutex::new(Some(handle))),
        })
    }

    /// Return the bound port (resolves port 0 to the OS-assigned port).
    fn bound_port(&self) -> PyResult<u16> {
        let guard = self
            .handle
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let h = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("server stopped"))?;
        Ok(h.bound_addr().port())
    }

    /// Publish `payload_json` (already-serialised JSON) on `channel`.
    /// Returns the number of receivers at the moment of send.
    fn publish(&self, channel: &str, payload_json: &str) -> PyResult<usize> {
        let guard = self
            .handle
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let h = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("server stopped"))?;
        h.publish(channel, payload_json.to_string())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Subscribe in-process (no socket, no asyncio). Returns a
    /// `PySubscriber` whose `recv(timeout)` is a blocking call.
    fn subscribe(&self, channel: &str) -> PyResult<PySubscriber> {
        let guard = self
            .handle
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let h = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("server stopped"))?;
        let sub = h
            .subscribe_local(channel)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PySubscriber {
            inner: Arc::new(sub),
        })
    }

    /// Stop accepting new connections and tear down the listener.
    fn stop(&self) -> PyResult<()> {
        let mut guard = self
            .handle
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if let Some(h) = guard.take() {
            h.stop();
        }
        Ok(())
    }
}

#[pyclass(name = "Subscriber")]
pub struct PySubscriber {
    inner: Arc<Subscriber>,
}

#[pymethods]
impl PySubscriber {
    /// Block until a payload arrives or `timeout` seconds elapses. Returns
    /// the JSON-encoded payload string, or `None` on timeout. The GIL is
    /// released across the wait.
    #[pyo3(signature = (timeout=None))]
    fn recv(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<Option<String>> {
        let sub = self.inner.clone();
        py.allow_threads(move || sub.recv_blocking(timeout))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyfunction]
fn validate_channel(name: &str) -> PyResult<()> {
    crate::channels::validate_channel(name).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn benchmark_recorded_swing_json(payload_json: &str, config_json: &str) -> PyResult<String> {
    let swing: crate::moving_horizon::RecordedSwing =
        serde_json::from_str(payload_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let config: crate::moving_horizon::MovingHorizonConfig =
        serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let report = crate::moving_horizon::benchmark_recorded_swing(&swing, &config)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&report).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn upstream_realtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyServer>()?;
    m.add_class::<PySubscriber>()?;
    m.add_function(wrap_pyfunction!(validate_channel, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_recorded_swing_json, m)?)?;
    Ok(())
}

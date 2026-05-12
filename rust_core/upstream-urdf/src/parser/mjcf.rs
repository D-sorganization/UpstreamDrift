//! MJCF (MuJoCo XML) parser — staged for a follow-up PR.
//!
//! The historical Python MJCF converter at
//! `src/shared/python/model_generation/converters/mjcf_converter.py` covers
//! a non-trivial subset of MuJoCo's XML format (geom/joint/site/sensor/
//! actuator/contact). Bringing that surface to byte-perfect round-trip
//! parity in Rust is more than fits in this PR; per the stop-conditions for
//! UD #5215 we scope this drop to URDF only and file the MJCF half as a
//! follow-up.
//!
//! This module is intentionally empty in this revision.

use crate::ast::Robot;
use crate::error::{UrdfError, UrdfResult};

/// Placeholder. Returns a "not implemented" error and is not exposed to
/// Python bindings until the MJCF follow-up issue is in flight.
pub fn parse_mjcf_str(_xml: &str) -> UrdfResult<Robot> {
    Err(UrdfError::Parse(
        "MJCF parsing is staged for a follow-up to UD #5215".into(),
    ))
}

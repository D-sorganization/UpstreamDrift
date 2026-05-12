//! MJCF writer — staged for a follow-up PR. See `parser::mjcf` for the
//! rationale behind deferring this surface.

use crate::ast::Robot;
use crate::error::{UrdfError, UrdfResult};

pub fn write_mjcf(_robot: &Robot) -> UrdfResult<String> {
    Err(UrdfError::Write(
        "MJCF writing is staged for a follow-up to UD #5215".into(),
    ))
}

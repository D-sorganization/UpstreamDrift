//! # upstream-urdf — Rust-backed URDF parser + writer
//!
//! Single Rust crate providing a typed AST, a `quick-xml`-based parser, and a
//! writer that round-trips URDF content with the same byte-significant
//! structure as the historical Python parsers under:
//!
//! - `src/shared/python/model_generation/converters/urdf_parser.py`
//! - `src/shared/python/humanoid_character_builder/generators/urdf_generator.py`
//! - `src/engines/.../mujoco_humanoid_golf/urdf_parser.py`
//!
//! MJCF support is staged in `parser::mjcf` / `writer::mjcf` but is intentionally
//! minimal in this initial drop — see the deferred-items section of the PR body
//! for UD #5215.

pub mod ast;
pub mod error;
pub mod parser;
pub mod writer;

#[cfg(feature = "python")]
mod bindings;

pub use ast::{
    Geometry, GeometryKind, Inertial, Joint, JointDynamics, JointKind, JointLimits, Link, Material,
    Origin, Robot, VisualOrCollision,
};
pub use error::{UrdfError, UrdfResult};
pub use parser::urdf::parse_urdf_str;
pub use writer::urdf::write_urdf;

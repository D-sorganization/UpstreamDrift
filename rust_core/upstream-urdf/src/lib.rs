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
//! MJCF support lives in [`mjcf_ast`] + [`parser::mjcf`] + [`writer::mjcf`].
//! It covers the 80% case used by the historical `mjcf_converter.py` shim —
//! bodies, joints, geoms, inertials, assets, actuators — and preserves the
//! remaining 20% (sensors, contacts, equality constraints, tendons,
//! keyframes) verbatim via [`mjcf_ast::RawSection`] so round-tripping does
//! not silently drop data. See UD #5243 for the deferred semantic coverage.

pub mod ast;
pub mod error;
pub mod mjcf_ast;
pub mod parser;
pub mod writer;

#[cfg(feature = "python")]
mod bindings;

pub use ast::{
    Geometry, GeometryKind, Inertial, Joint, JointDynamics, JointKind, JointLimits, Link, Material,
    Origin, Robot, VisualOrCollision,
};
pub use error::{UrdfError, UrdfResult};
pub use mjcf_ast::{
    Actuator as MjActuator, Asset as MjAsset, Body as MjBody, Compiler as MjCompiler,
    Geom as MjGeom, Inertial as MjInertial, Joint as MjJoint, MjOption, MujocoDocument,
    RawSection as MjRawSection, Site as MjSite, Worldbody as MjWorldbody,
};
pub use parser::mjcf::parse_mjcf_str;
pub use parser::urdf::parse_urdf_str;
pub use writer::mjcf::write_mjcf;
pub use writer::urdf::write_urdf;

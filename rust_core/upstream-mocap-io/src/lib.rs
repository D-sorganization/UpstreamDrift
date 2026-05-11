//! # upstream-mocap-io — Native mocap I/O adapters
//!
//! High-performance Rust parsers for the motion-pipeline source formats:
//! C3D (binary), BVH (text joint hierarchy), TRC (OpenSim text markers).
//!
//! The Rust side returns a [`MarkerData`] (or [`JointData`] for BVH) struct:
//! a flat `f32` buffer of marker positions plus labels, fps, units. The
//! Python facade in `motion_pipeline/sources/{c3d,bvh,trc}_adapter.py` is
//! responsible for constructing `MarkerTrajectory` / `JointTrajectory`
//! pydantic objects from those arrays.
//!
//! Numerical-fidelity contract: byte-identical canonical output on golden
//! files vs. the pure-Python reference implementation
//! (issue #5213, opportunity 2 in `upstreamdrift_rust_opportunities.md`).
//!
//! Scope this PR: marker data only. C3D analog / event sections are not
//! parsed (deferred — see issue #5213 follow-up).

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod bvh;
pub mod c3d;
pub mod trc;

#[cfg(feature = "python")]
mod bindings;

use std::fmt;

/// Parsed marker trajectory (C3D, TRC). Positions are stored as a flat
/// row-major `f32` buffer of length `n_frames * n_markers * 3` in **meters**
/// after the per-format unit conversion (mm → m). Occluded markers are
/// encoded as `f32::NAN` so the Python facade can drop them when building
/// `MarkerFrame`s.
#[derive(Debug, Clone)]
pub struct MarkerData {
    pub names: Vec<String>,
    /// `n_frames * n_markers * 3`, row-major: frame-major, then marker, then xyz.
    pub positions: Vec<f32>,
    pub n_frames: usize,
    pub n_markers: usize,
    pub fps: f32,
    /// Raw units string from the source (e.g. `"mm"`, `"m"`). The
    /// `positions` buffer is already converted to meters.
    pub units: String,
}

/// Parsed joint trajectory (BVH). Per-frame channel values are returned in
/// **radians** with one row per frame, `num_dofs` columns. Channel-order
/// reconstruction (rotation order, translation channels) is the
/// responsibility of the Python facade.
#[derive(Debug, Clone)]
pub struct JointData {
    /// Hierarchy in pre-order; each entry is `(name, parent_index_or_None, channels)`.
    pub joints: Vec<JointInfo>,
    /// `n_frames * num_dofs`, row-major. Angles in radians, translations in source units.
    pub motion: Vec<f32>,
    pub n_frames: usize,
    pub num_dofs: usize,
    pub fps: f32,
}

#[derive(Debug, Clone)]
pub struct JointInfo {
    pub name: String,
    pub parent: Option<usize>,
    pub channels: Vec<String>,
}

#[derive(Debug)]
pub enum ParseError {
    Io(std::io::Error),
    Format(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::Io(e) => write!(f, "I/O error: {e}"),
            ParseError::Format(s) => write!(f, "Format error: {s}"),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<std::io::Error> for ParseError {
    fn from(e: std::io::Error) -> Self {
        ParseError::Io(e)
    }
}

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
//! C3D marker data is always parsed. Event, analog, and force-platform
//! metadata are parsed when present.

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
    /// Optional C3D EVENT group metadata, expressed in seconds.
    pub events: Vec<C3dEvent>,
    /// Optional C3D analog channel samples, expressed in scaled source units.
    pub analog: Option<C3dAnalogData>,
    /// Optional C3D FORCE_PLATFORM group metadata.
    pub force_platforms: Vec<C3dForcePlatform>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct C3dEvent {
    pub label: String,
    pub context: String,
    pub time_s: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct C3dAnalogData {
    pub labels: Vec<String>,
    pub units: Vec<String>,
    /// `n_frames * samples_per_frame * n_channels`, row-major by 3D frame,
    /// analog sub-sample, then channel.
    pub values: Vec<f32>,
    pub n_frames: usize,
    pub samples_per_frame: usize,
    pub n_channels: usize,
    pub rate: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct C3dForcePlatform {
    pub platform_type: i16,
    /// One-based analog channel numbers as stored in FORCE_PLATFORM:CHANNEL.
    pub channels: Vec<i16>,
    /// Four xyz corner triplets, in source C3D units.
    pub corners: Vec<[f32; 3]>,
    /// xyz origin triplet, in source C3D units.
    pub origin: [f32; 3],
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

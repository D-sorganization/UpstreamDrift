//! Typed AST for URDF (and, where it overlaps, MJCF) bodies.
//!
//! Field names intentionally mirror the URDF XML attribute names. The
//! `serde` derives let the Python bindings convert to/from dicts with a
//! single `serde_json` round-trip.

use serde::{Deserialize, Serialize};

/// A `<robot>` document — the URDF root element.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Robot {
    pub name: String,
    #[serde(default)]
    pub links: Vec<Link>,
    #[serde(default)]
    pub joints: Vec<Joint>,
    #[serde(default)]
    pub materials: Vec<Material>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Origin {
    /// xyz translation in metres. URDF default: `0 0 0`.
    #[serde(default = "Origin::default_xyz")]
    pub xyz: [f64; 3],
    /// roll/pitch/yaw in radians. URDF default: `0 0 0`.
    #[serde(default = "Origin::default_rpy")]
    pub rpy: [f64; 3],
}

impl Origin {
    fn default_xyz() -> [f64; 3] {
        [0.0; 3]
    }
    fn default_rpy() -> [f64; 3] {
        [0.0; 3]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Inertial {
    #[serde(default)]
    pub origin: Origin,
    pub mass: f64,
    pub ixx: f64,
    pub iyy: f64,
    pub izz: f64,
    #[serde(default)]
    pub ixy: f64,
    #[serde(default)]
    pub ixz: f64,
    #[serde(default)]
    pub iyz: f64,
}

impl Default for Inertial {
    fn default() -> Self {
        Self {
            origin: Origin::default(),
            mass: 0.0,
            ixx: 0.0,
            iyy: 0.0,
            izz: 0.0,
            ixy: 0.0,
            ixz: 0.0,
            iyz: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "params", rename_all = "lowercase")]
pub enum GeometryKind {
    /// `<box size="x y z"/>`
    Box { size: [f64; 3] },
    /// `<cylinder radius="r" length="l"/>`
    Cylinder { radius: f64, length: f64 },
    /// `<sphere radius="r"/>`
    Sphere { radius: f64 },
    /// `<mesh filename="…" scale="x y z"/>`
    Mesh {
        filename: String,
        #[serde(default = "GeometryKind::default_scale")]
        scale: [f64; 3],
    },
}

impl GeometryKind {
    fn default_scale() -> [f64; 3] {
        [1.0; 3]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    #[serde(flatten)]
    pub kind: GeometryKind,
}

/// A `<visual>` or `<collision>` block on a link. They share a layout in
/// URDF so we factor them here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VisualOrCollision {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub origin: Origin,
    pub geometry: Option<Geometry>,
    /// Visual blocks may reference a material by name (resolved against the
    /// robot-level `materials` table) or define one inline. Collision blocks
    /// leave this `None`.
    #[serde(default)]
    pub material: Option<Material>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Link {
    pub name: String,
    #[serde(default)]
    pub inertial: Option<Inertial>,
    #[serde(default)]
    pub visuals: Vec<VisualOrCollision>,
    #[serde(default)]
    pub collisions: Vec<VisualOrCollision>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JointKind {
    Fixed,
    Revolute,
    Continuous,
    Prismatic,
    Planar,
    Floating,
}

impl JointKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            JointKind::Fixed => "fixed",
            JointKind::Revolute => "revolute",
            JointKind::Continuous => "continuous",
            JointKind::Prismatic => "prismatic",
            JointKind::Planar => "planar",
            JointKind::Floating => "floating",
        }
    }
}

impl std::str::FromStr for JointKind {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "fixed" => JointKind::Fixed,
            "revolute" => JointKind::Revolute,
            "continuous" => JointKind::Continuous,
            "prismatic" => JointKind::Prismatic,
            "planar" => JointKind::Planar,
            "floating" => JointKind::Floating,
            _ => return Err(()),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointLimits {
    pub lower: f64,
    pub upper: f64,
    pub effort: f64,
    pub velocity: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct JointDynamics {
    #[serde(default)]
    pub damping: f64,
    #[serde(default)]
    pub friction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Joint {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: JointKind,
    pub parent: String,
    pub child: String,
    #[serde(default)]
    pub origin: Origin,
    /// Default `[0, 0, 1]` to match URDF spec.
    #[serde(default = "Joint::default_axis")]
    pub axis: [f64; 3],
    #[serde(default)]
    pub limits: Option<JointLimits>,
    #[serde(default)]
    pub dynamics: JointDynamics,
}

impl Joint {
    fn default_axis() -> [f64; 3] {
        [0.0, 0.0, 1.0]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub name: String,
    #[serde(default)]
    pub color: Option<[f64; 4]>,
    #[serde(default)]
    pub texture: Option<String>,
}

//! Typed AST for URDF (and, where it overlaps, MJCF) bodies.
//!
//! Field names intentionally mirror the URDF XML attribute names. The
//! `serde` derives let the Python bindings convert to/from dicts with a
//! single `serde_json` round-trip.

use serde::{Deserialize, Serialize};

use crate::error::{UrdfError, UrdfResult};

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

impl Robot {
    /// Validate biomechanics/physics domain invariants on the parsed robot.
    ///
    /// Enforced constraints (issue #7659):
    /// - every link inertial has `mass > 0`;
    /// - the inertia tensor has a positive diagonal and is symmetric
    ///   positive-definite (checked via Sylvester's leading-minor criterion);
    /// - every joint limit has `lower <= upper` and non-negative
    ///   `effort`/`velocity`.
    ///
    /// Returns [`UrdfError::Schema`] describing the first violation found.
    pub fn validate(&self) -> UrdfResult<()> {
        for link in &self.links {
            if let Some(inert) = &link.inertial {
                inert.validate(&link.name)?;
            }
        }
        for joint in &self.joints {
            if let Some(limits) = &joint.limits {
                limits.validate(&joint.name)?;
            }
        }
        Ok(())
    }
}

/// Strict, NaN-rejecting positivity test (`x > 0` and finite-ish): returns
/// `false` for `0.0`, negatives, and `NaN`.
fn is_strictly_positive(x: f64) -> bool {
    x > 0.0
}

impl Inertial {
    /// Validate mass positivity and inertia-tensor positive-definiteness for a
    /// single link. `link` names the owning link for diagnostics.
    pub(crate) fn validate(&self, link: &str) -> UrdfResult<()> {
        if !is_strictly_positive(self.mass) {
            return Err(UrdfError::Schema(format!(
                "link {link:?}: mass must be positive, got {}",
                self.mass
            )));
        }
        // Diagonal must be strictly positive.
        for (name, v) in [("ixx", self.ixx), ("iyy", self.iyy), ("izz", self.izz)] {
            if !is_strictly_positive(v) {
                return Err(UrdfError::Schema(format!(
                    "link {link:?}: inertia {name} must be positive, got {v}"
                )));
            }
        }
        // Symmetric positive-definiteness via Sylvester's criterion on the
        // leading principal minors of
        //   [ ixx  ixy  ixz ]
        //   [ ixy  iyy  iyz ]
        //   [ ixz  iyz  izz ].
        let (ixx, iyy, izz) = (self.ixx, self.iyy, self.izz);
        let (ixy, ixz, iyz) = (self.ixy, self.ixz, self.iyz);
        let minor2 = ixx * iyy - ixy * ixy;
        let det = ixx * (iyy * izz - iyz * iyz) - ixy * (ixy * izz - iyz * ixz)
            + ixz * (ixy * iyz - iyy * ixz);
        if !is_strictly_positive(minor2) || !is_strictly_positive(det) {
            return Err(UrdfError::Schema(format!(
                "link {link:?}: inertia tensor must be positive-definite \
                 (leading minors: {minor2}, {det})"
            )));
        }
        Ok(())
    }
}

impl JointLimits {
    /// Validate `lower <= upper` and non-negative effort/velocity for a single
    /// joint. `joint` names the owning joint for diagnostics.
    pub(crate) fn validate(&self, joint: &str) -> UrdfResult<()> {
        if self.lower > self.upper {
            return Err(UrdfError::Schema(format!(
                "joint {joint:?}: limit lower ({}) must be <= upper ({})",
                self.lower, self.upper
            )));
        }
        if self.effort < 0.0 {
            return Err(UrdfError::Schema(format!(
                "joint {joint:?}: limit effort must be >= 0, got {}",
                self.effort
            )));
        }
        if self.velocity < 0.0 {
            return Err(UrdfError::Schema(format!(
                "joint {joint:?}: limit velocity must be >= 0, got {}",
                self.velocity
            )));
        }
        Ok(())
    }
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

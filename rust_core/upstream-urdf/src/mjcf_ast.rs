//! Typed AST for MJCF (MuJoCo XML) documents.
//!
//! MJCF and URDF overlap conceptually — both describe a rigid-body tree —
//! but they differ enough in shape (nested `<body>` recursion, half-sized
//! geom dimensions, hinge/slide/ball/free joint kinds, asset references
//! by name) that we keep them in distinct AST modules. The
//! [`MujocoDocument`] root mirrors the order of the top-level sections
//! MuJoCo's XSD defines: `compiler`, `option`, `default`, `asset`,
//! `worldbody`, `tendon`, `actuator`, `sensor`, `equality`, `contact`,
//! `keyframe`, etc. We model the subset that the historical Python
//! `MJCFConverter` in `src/shared/python/model_generation/converters/
//! mjcf_converter.py` actually produces and consumes — body / joint /
//! geom / inertial / asset(material+mesh) / actuator — and preserve the
//! remainder verbatim via [`Body::extras`].

use serde::{Deserialize, Serialize};

/// Top-level `<mujoco>` document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MujocoDocument {
    /// `model` attribute on the root element. Defaults to `""`.
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub compiler: Option<Compiler>,
    #[serde(default)]
    pub option: Option<MjOption>,
    /// Verbatim copy of the `<default>` block; the historical Python
    /// converter passes this through without semantic interpretation.
    #[serde(default)]
    pub default_xml: Option<String>,
    #[serde(default)]
    pub assets: Vec<Asset>,
    #[serde(default)]
    pub worldbody: Worldbody,
    #[serde(default)]
    pub actuators: Vec<Actuator>,
    /// Verbatim contents of unknown / out-of-scope sections (sensor,
    /// tendon, contact, equality, keyframe, custom, …). Stored as raw XML
    /// so a round-trip preserves them.
    #[serde(default)]
    pub extras: Vec<RawSection>,
}

/// `<compiler>` element attributes commonly emitted by `mjcf_converter.py`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Compiler {
    #[serde(default)]
    pub angle: Option<String>,
    #[serde(default)]
    pub coordinate: Option<String>,
    #[serde(default)]
    pub inertiafromgeom: Option<String>,
    #[serde(default)]
    pub meshdir: Option<String>,
    #[serde(default)]
    pub texturedir: Option<String>,
    /// Other compiler attributes preserved verbatim for round-trip.
    #[serde(default)]
    pub extra_attrs: Vec<(String, String)>,
}

/// `<option>` element attributes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MjOption {
    /// `gravity="x y z"` parsed into floats. `None` means "not set".
    #[serde(default)]
    pub gravity: Option<[f64; 3]>,
    #[serde(default)]
    pub timestep: Option<f64>,
    #[serde(default)]
    pub extra_attrs: Vec<(String, String)>,
}

/// Entry in the `<asset>` section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Asset {
    Material {
        name: String,
        #[serde(default)]
        rgba: Option<[f64; 4]>,
        #[serde(default)]
        specular: Option<f64>,
        #[serde(default)]
        shininess: Option<f64>,
        #[serde(default)]
        texture: Option<String>,
        #[serde(default)]
        extra_attrs: Vec<(String, String)>,
    },
    Mesh {
        name: String,
        #[serde(default)]
        file: Option<String>,
        #[serde(default)]
        scale: Option<[f64; 3]>,
        #[serde(default)]
        extra_attrs: Vec<(String, String)>,
    },
    Texture {
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        file: Option<String>,
        #[serde(default)]
        type_: Option<String>,
        #[serde(default)]
        extra_attrs: Vec<(String, String)>,
    },
}

/// `<worldbody>` — the recursive root of the kinematic tree. The
/// worldbody is itself implicit and is **not** a `<body>` element; it
/// carries direct child geoms / sites / lights / bodies.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Worldbody {
    #[serde(default)]
    pub bodies: Vec<Body>,
    #[serde(default)]
    pub geoms: Vec<Geom>,
    #[serde(default)]
    pub sites: Vec<Site>,
    /// Verbatim `<light>` and other rarely-touched elements at the
    /// worldbody level.
    #[serde(default)]
    pub extras: Vec<RawSection>,
}

/// A `<body>` element. Bodies nest to arbitrary depth.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Body {
    pub name: String,
    /// `pos="x y z"`. Defaults to origin.
    #[serde(default)]
    pub pos: [f64; 3],
    /// Optional `quat="w x y z"` orientation. `None` means the MuJoCo
    /// default identity.
    #[serde(default)]
    pub quat: Option<[f64; 4]>,
    /// Optional `euler="x y z"` orientation (radians or degrees per the
    /// compiler `angle` attribute — we do not convert).
    #[serde(default)]
    pub euler: Option<[f64; 3]>,
    /// `childclass` for inheriting `<default>` settings.
    #[serde(default)]
    pub childclass: Option<String>,
    #[serde(default)]
    pub inertial: Option<Inertial>,
    #[serde(default)]
    pub joints: Vec<Joint>,
    #[serde(default)]
    pub geoms: Vec<Geom>,
    #[serde(default)]
    pub sites: Vec<Site>,
    #[serde(default)]
    pub bodies: Vec<Body>,
    /// Other attributes / nested elements preserved for round-trip
    /// (e.g. `<freejoint>`, `<camera>`, `<light>`).
    #[serde(default)]
    pub extras: Vec<RawSection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Inertial {
    pub mass: f64,
    pub pos: [f64; 3],
    /// Diagonal inertia: `[ixx, iyy, izz]`. Mutually exclusive with
    /// `full`.
    #[serde(default)]
    pub diaginertia: Option<[f64; 3]>,
    /// Full inertia: `[ixx, iyy, izz, ixy, ixz, iyz]`.
    #[serde(default)]
    pub fullinertia: Option<[f64; 6]>,
    #[serde(default)]
    pub quat: Option<[f64; 4]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Joint {
    #[serde(default)]
    pub name: Option<String>,
    /// `type` attribute: `hinge`, `slide`, `ball`, `free`.
    #[serde(default = "Joint::default_type")]
    pub type_: String,
    /// `axis="x y z"`. URDF/MJCF default `0 0 1`.
    #[serde(default = "Joint::default_axis")]
    pub axis: [f64; 3],
    #[serde(default)]
    pub pos: Option<[f64; 3]>,
    #[serde(default)]
    pub range: Option<[f64; 2]>,
    #[serde(default)]
    pub damping: Option<f64>,
    #[serde(default)]
    pub frictionloss: Option<f64>,
    #[serde(default)]
    pub armature: Option<f64>,
    #[serde(default)]
    pub stiffness: Option<f64>,
    #[serde(default)]
    pub class: Option<String>,
    #[serde(default)]
    pub limited: Option<String>,
    #[serde(default)]
    pub extra_attrs: Vec<(String, String)>,
}

impl Joint {
    fn default_type() -> String {
        "hinge".to_string()
    }
    fn default_axis() -> [f64; 3] {
        [0.0, 0.0, 1.0]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Geom {
    #[serde(default)]
    pub name: Option<String>,
    /// `type` attribute. Defaults to `sphere` per MuJoCo spec.
    #[serde(default = "Geom::default_type")]
    pub type_: String,
    /// `size` attribute as a list of floats (length depends on type).
    #[serde(default)]
    pub size: Vec<f64>,
    #[serde(default)]
    pub pos: Option<[f64; 3]>,
    #[serde(default)]
    pub quat: Option<[f64; 4]>,
    /// `fromto="x1 y1 z1 x2 y2 z2"` — alternative way to specify
    /// capsule/cylinder ends.
    #[serde(default)]
    pub fromto: Option<[f64; 6]>,
    #[serde(default)]
    pub rgba: Option<[f64; 4]>,
    #[serde(default)]
    pub material: Option<String>,
    /// Mesh name (when `type="mesh"`).
    #[serde(default)]
    pub mesh: Option<String>,
    #[serde(default)]
    pub mass: Option<f64>,
    #[serde(default)]
    pub density: Option<f64>,
    #[serde(default)]
    pub class: Option<String>,
    #[serde(default)]
    pub group: Option<i64>,
    #[serde(default)]
    pub contype: Option<i64>,
    #[serde(default)]
    pub conaffinity: Option<i64>,
    #[serde(default)]
    pub friction: Option<Vec<f64>>,
    #[serde(default)]
    pub extra_attrs: Vec<(String, String)>,
}

impl Geom {
    fn default_type() -> String {
        "sphere".to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Site {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub pos: Option<[f64; 3]>,
    #[serde(default)]
    pub size: Vec<f64>,
    #[serde(default)]
    pub type_: Option<String>,
    #[serde(default)]
    pub rgba: Option<[f64; 4]>,
    #[serde(default)]
    pub extra_attrs: Vec<(String, String)>,
}

/// An `<actuator>` child element — `motor`, `position`, `velocity`,
/// `general`, etc. We carry the element name and attributes verbatim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Actuator {
    /// XML tag (e.g. `motor`, `position`).
    pub tag: String,
    pub attrs: Vec<(String, String)>,
}

/// An out-of-scope XML subtree preserved verbatim so we can round-trip
/// it back out. This is the escape hatch for the 20% of MJCF surface we
/// do not model semantically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawSection {
    /// Top-level tag of this preserved subtree.
    pub tag: String,
    /// Full XML text for the subtree, including the open and close tag.
    pub xml: String,
}

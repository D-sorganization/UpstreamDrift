//! URDF (Unified Robot Description Format) parser.
//!
//! Single-pass `quick-xml` reader → typed AST. Tracks an element stack so
//! that nested attributes (e.g. `inertial > origin xyz=…`) land in the right
//! AST slot.

use std::str::FromStr;

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

use crate::ast::{
    Geometry, GeometryKind, Inertial, Joint, JointDynamics, JointKind, JointLimits, Link, Material,
    Origin, Robot, VisualOrCollision,
};
use crate::error::{UrdfError, UrdfResult};

/// Parse a URDF document supplied as a string. The input must be valid XML
/// with `<robot>` as the root element.
pub fn parse_urdf_str(xml: &str) -> UrdfResult<Robot> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut robot = Robot::default();

    // Element stack tracks the path so child attribute parsers know what
    // outer block they belong to.
    let mut stack: Vec<String> = Vec::new();
    // Working buffers for the currently-open link / joint / visual / etc.
    let mut cur_link: Option<Link> = None;
    let mut cur_joint: Option<Joint> = None;
    let mut cur_inertial: Option<Inertial> = None;
    let mut cur_vis: Option<VisualOrCollision> = None;
    let mut cur_col: Option<VisualOrCollision> = None;
    let mut cur_material: Option<Material> = None;
    // `true` when the current `<material>` element is at the robot's top
    // level (its definition goes into `robot.materials`); `false` when the
    // `<material>` is nested inside a `<visual>` block.
    let mut material_at_root = false;
    let mut seen_root = false;

    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf)? {
            Event::Eof => break,
            Event::Decl(_)
            | Event::Comment(_)
            | Event::PI(_)
            | Event::Text(_)
            | Event::CData(_) => {}
            Event::DocType(_) => {}
            Event::Start(e) => {
                handle_open(
                    &e,
                    &mut stack,
                    &mut robot,
                    &mut cur_link,
                    &mut cur_joint,
                    &mut cur_inertial,
                    &mut cur_vis,
                    &mut cur_col,
                    &mut cur_material,
                    &mut material_at_root,
                    &mut seen_root,
                )?;
            }
            Event::Empty(e) => {
                handle_open(
                    &e,
                    &mut stack,
                    &mut robot,
                    &mut cur_link,
                    &mut cur_joint,
                    &mut cur_inertial,
                    &mut cur_vis,
                    &mut cur_col,
                    &mut cur_material,
                    &mut material_at_root,
                    &mut seen_root,
                )?;
                // Empty tags do not enter the stack; emit a synthetic close.
                let name = String::from_utf8(e.name().as_ref().to_vec())
                    .map_err(|err| UrdfError::Xml(err.to_string()))?;
                close_element(
                    &name,
                    &mut stack,
                    &mut robot,
                    &mut cur_link,
                    &mut cur_joint,
                    &mut cur_inertial,
                    &mut cur_vis,
                    &mut cur_col,
                    &mut cur_material,
                    &mut material_at_root,
                )?;
            }
            Event::End(e) => {
                let name = String::from_utf8(e.name().as_ref().to_vec())
                    .map_err(|err| UrdfError::Xml(err.to_string()))?;
                close_element(
                    &name,
                    &mut stack,
                    &mut robot,
                    &mut cur_link,
                    &mut cur_joint,
                    &mut cur_inertial,
                    &mut cur_vis,
                    &mut cur_col,
                    &mut cur_material,
                    &mut material_at_root,
                )?;
            }
        }
        buf.clear();
    }

    if !seen_root {
        return Err(UrdfError::Schema(
            "expected <robot> root element".to_string(),
        ));
    }
    Ok(robot)
}

#[allow(clippy::too_many_arguments)]
fn handle_open(
    e: &BytesStart<'_>,
    stack: &mut Vec<String>,
    robot: &mut Robot,
    cur_link: &mut Option<Link>,
    cur_joint: &mut Option<Joint>,
    cur_inertial: &mut Option<Inertial>,
    cur_vis: &mut Option<VisualOrCollision>,
    cur_col: &mut Option<VisualOrCollision>,
    cur_material: &mut Option<Material>,
    material_at_root: &mut bool,
    seen_root: &mut bool,
) -> UrdfResult<()> {
    let name = std::str::from_utf8(e.name().as_ref())?.to_string();
    let attrs = collect_attrs(e)?;
    let stack_top = stack.last().cloned();

    match name.as_str() {
        "robot" => {
            *seen_root = true;
            robot.name = attrs.get("name").cloned().unwrap_or_default();
        }
        "link" => {
            *cur_link = Some(Link {
                name: attrs.get("name").cloned().unwrap_or_default(),
                ..Default::default()
            });
        }
        "joint" => {
            // Only at the robot level do we treat <joint> as a top-level joint.
            // Some URDF dialects embed transmission-related <joint> stubs which
            // we ignore.
            if matches!(stack_top.as_deref(), Some("robot")) {
                let kind = attrs
                    .get("type")
                    .and_then(|s| JointKind::from_str(s).ok())
                    .unwrap_or(JointKind::Fixed);
                *cur_joint = Some(Joint {
                    name: attrs.get("name").cloned().unwrap_or_default(),
                    kind,
                    parent: String::new(),
                    child: String::new(),
                    origin: Origin::default(),
                    axis: [0.0, 0.0, 1.0],
                    limits: None,
                    dynamics: JointDynamics::default(),
                });
            }
        }
        "parent" => {
            if let Some(j) = cur_joint.as_mut() {
                if let Some(link_name) = attrs.get("link") {
                    j.parent = link_name.clone();
                }
            }
        }
        "child" => {
            if let Some(j) = cur_joint.as_mut() {
                if let Some(link_name) = attrs.get("link") {
                    j.child = link_name.clone();
                }
            }
        }
        "axis" => {
            if let Some(j) = cur_joint.as_mut() {
                if let Some(xyz) = attrs.get("xyz").and_then(|s| parse_vec3(s).ok()) {
                    j.axis = xyz;
                }
            }
        }
        "limit" => {
            if let Some(j) = cur_joint.as_mut() {
                j.limits = Some(JointLimits {
                    lower: attr_f64(&attrs, "lower", -std::f64::consts::PI),
                    upper: attr_f64(&attrs, "upper", std::f64::consts::PI),
                    effort: attr_f64(&attrs, "effort", 1000.0),
                    velocity: attr_f64(&attrs, "velocity", 10.0),
                });
            }
        }
        "dynamics" => {
            if let Some(j) = cur_joint.as_mut() {
                j.dynamics = JointDynamics {
                    damping: attr_f64(&attrs, "damping", 0.0),
                    friction: attr_f64(&attrs, "friction", 0.0),
                };
            }
        }
        "inertial" => {
            *cur_inertial = Some(Inertial::default());
        }
        "mass" => {
            if let Some(inert) = cur_inertial.as_mut() {
                inert.mass = attr_f64(&attrs, "value", 0.0);
            }
        }
        "inertia" => {
            if let Some(inert) = cur_inertial.as_mut() {
                inert.ixx = attr_f64(&attrs, "ixx", 0.0);
                inert.iyy = attr_f64(&attrs, "iyy", 0.0);
                inert.izz = attr_f64(&attrs, "izz", 0.0);
                inert.ixy = attr_f64(&attrs, "ixy", 0.0);
                inert.ixz = attr_f64(&attrs, "ixz", 0.0);
                inert.iyz = attr_f64(&attrs, "iyz", 0.0);
            }
        }
        "origin" => {
            let xyz = attrs
                .get("xyz")
                .and_then(|s| parse_vec3(s).ok())
                .unwrap_or([0.0; 3]);
            let rpy = attrs
                .get("rpy")
                .and_then(|s| parse_vec3(s).ok())
                .unwrap_or([0.0; 3]);
            let origin = Origin { xyz, rpy };
            // Dispatch by parent element on the stack.
            match stack_top.as_deref() {
                Some("inertial") => {
                    if let Some(inert) = cur_inertial.as_mut() {
                        inert.origin = origin;
                    }
                }
                Some("visual") => {
                    if let Some(v) = cur_vis.as_mut() {
                        v.origin = origin;
                    }
                }
                Some("collision") => {
                    if let Some(c) = cur_col.as_mut() {
                        c.origin = origin;
                    }
                }
                Some("joint") => {
                    if let Some(j) = cur_joint.as_mut() {
                        j.origin = origin;
                    }
                }
                _ => {}
            }
        }
        "visual" => {
            *cur_vis = Some(VisualOrCollision {
                name: attrs.get("name").cloned(),
                ..Default::default()
            });
        }
        "collision" => {
            *cur_col = Some(VisualOrCollision {
                name: attrs.get("name").cloned(),
                ..Default::default()
            });
        }
        "geometry" => {
            // Geometry is a container — its concrete shape arrives as the
            // next opened element. Nothing to do here.
        }
        "box" => {
            if let Some(size) = attrs.get("size").and_then(|s| parse_vec3(s).ok()) {
                set_current_geometry(
                    stack_top.as_deref(),
                    cur_vis,
                    cur_col,
                    GeometryKind::Box { size },
                );
            }
        }
        "cylinder" => {
            let radius = attr_f64(&attrs, "radius", 0.0);
            let length = attr_f64(&attrs, "length", 0.0);
            set_current_geometry(
                stack_top.as_deref(),
                cur_vis,
                cur_col,
                GeometryKind::Cylinder { radius, length },
            );
        }
        "sphere" => {
            let radius = attr_f64(&attrs, "radius", 0.0);
            set_current_geometry(
                stack_top.as_deref(),
                cur_vis,
                cur_col,
                GeometryKind::Sphere { radius },
            );
        }
        "mesh" => {
            let filename = attrs.get("filename").cloned().unwrap_or_default();
            let scale = attrs
                .get("scale")
                .and_then(|s| parse_vec3(s).ok())
                .unwrap_or([1.0, 1.0, 1.0]);
            set_current_geometry(
                stack_top.as_deref(),
                cur_vis,
                cur_col,
                GeometryKind::Mesh { filename, scale },
            );
        }
        "material" => {
            *cur_material = Some(Material {
                name: attrs.get("name").cloned().unwrap_or_default(),
                color: None,
                texture: None,
            });
            *material_at_root = matches!(stack_top.as_deref(), Some("robot"));
        }
        "color" => {
            if let Some(m) = cur_material.as_mut() {
                if let Some(rgba) = attrs.get("rgba").and_then(|s| parse_vec4(s).ok()) {
                    m.color = Some(rgba);
                }
            }
        }
        "texture" => {
            if let Some(m) = cur_material.as_mut() {
                m.texture = attrs.get("filename").cloned();
            }
        }
        _ => {
            // Unknown elements (transmissions, ros_control plugins, gazebo
            // blocks, sensors…) are preserved by being silently ignored at
            // this stage. They are tracked as deferred items for #5215.
        }
    }

    stack.push(name);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn close_element(
    name: &str,
    stack: &mut Vec<String>,
    robot: &mut Robot,
    cur_link: &mut Option<Link>,
    cur_joint: &mut Option<Joint>,
    cur_inertial: &mut Option<Inertial>,
    cur_vis: &mut Option<VisualOrCollision>,
    cur_col: &mut Option<VisualOrCollision>,
    cur_material: &mut Option<Material>,
    material_at_root: &mut bool,
) -> UrdfResult<()> {
    // Pop matching tag off the stack. Allow tolerant pop if input is loose.
    if let Some(pos) = stack.iter().rposition(|x| x == name) {
        stack.truncate(pos);
    }
    match name {
        "link" => {
            if let Some(link) = cur_link.take() {
                robot.links.push(link);
            }
        }
        "joint" => {
            // Only commit joints that were opened at the robot level.
            if let Some(joint) = cur_joint.take() {
                robot.joints.push(joint);
            }
        }
        "inertial" => {
            if let (Some(inert), Some(link)) = (cur_inertial.take(), cur_link.as_mut()) {
                link.inertial = Some(inert);
            }
        }
        "visual" => {
            if let (Some(v), Some(link)) = (cur_vis.take(), cur_link.as_mut()) {
                link.visuals.push(v);
            }
        }
        "collision" => {
            if let (Some(c), Some(link)) = (cur_col.take(), cur_link.as_mut()) {
                link.collisions.push(c);
            }
        }
        "material" => {
            if let Some(m) = cur_material.take() {
                if *material_at_root {
                    robot.materials.push(m);
                } else if let Some(v) = cur_vis.as_mut() {
                    v.material = Some(m);
                }
                *material_at_root = false;
            }
        }
        _ => {}
    }
    Ok(())
}

fn set_current_geometry(
    stack_top: Option<&str>,
    cur_vis: &mut Option<VisualOrCollision>,
    cur_col: &mut Option<VisualOrCollision>,
    kind: GeometryKind,
) {
    // Geometry shapes appear inside <geometry>, which sits inside <visual>
    // or <collision>. Resolve from the second-from-top of the stack.
    let geom = Geometry { kind };
    // We don't have direct stack access here, but `stack_top` is the
    // immediate parent (<geometry>) — its grandparent is what we need. The
    // simple/reliable thing is to set on whichever container is currently
    // open: if both visual and collision are open we cannot tell (URDF
    // forbids that anyway), so prefer visual.
    let _ = stack_top;
    if let Some(v) = cur_vis.as_mut() {
        if v.geometry.is_none() {
            v.geometry = Some(geom);
            return;
        }
    }
    if let Some(c) = cur_col.as_mut() {
        if c.geometry.is_none() {
            c.geometry = Some(geom);
        }
    }
}

fn collect_attrs(e: &BytesStart<'_>) -> UrdfResult<std::collections::HashMap<String, String>> {
    let mut out = std::collections::HashMap::new();
    for a in e.attributes() {
        let a = a?;
        let key = std::str::from_utf8(a.key.as_ref())?.to_string();
        let val = a.unescape_value().map_err(UrdfError::from)?.into_owned();
        out.insert(key, val);
    }
    Ok(out)
}

fn attr_f64(map: &std::collections::HashMap<String, String>, key: &str, default: f64) -> f64 {
    map.get(key)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(default)
}

fn parse_vec3(s: &str) -> UrdfResult<[f64; 3]> {
    let parts: Vec<f64> = s
        .split_whitespace()
        .map(|p| {
            p.parse::<f64>()
                .map_err(|e| UrdfError::Parse(e.to_string()))
        })
        .collect::<UrdfResult<_>>()?;
    if parts.len() != 3 {
        return Err(UrdfError::Parse(format!(
            "expected 3 floats, got {}: {s:?}",
            parts.len()
        )));
    }
    Ok([parts[0], parts[1], parts[2]])
}

fn parse_vec4(s: &str) -> UrdfResult<[f64; 4]> {
    let parts: Vec<f64> = s
        .split_whitespace()
        .map(|p| {
            p.parse::<f64>()
                .map_err(|e| UrdfError::Parse(e.to_string()))
        })
        .collect::<UrdfResult<_>>()?;
    if parts.len() != 4 {
        return Err(UrdfError::Parse(format!(
            "expected 4 floats, got {}: {s:?}",
            parts.len()
        )));
    }
    Ok([parts[0], parts[1], parts[2], parts[3]])
}

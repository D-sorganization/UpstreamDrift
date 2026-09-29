//! Typed AST → URDF XML string.
//!
//! The writer's primary contract is *schema-equivalent* round-trip — parse,
//! write, re-parse, and the resulting `Robot` should equal the original.
//! It is not a byte-stable pretty printer; the historical Python writers
//! also normalise whitespace and attribute order.

use std::fmt::Write as _;

use crate::ast::{
    Geometry, GeometryKind, Inertial, Joint, JointDynamics, JointLimits, Link, Material, Origin,
    Robot, VisualOrCollision,
};
use crate::error::UrdfResult;

const INDENT: &str = "  ";

/// Render a [`Robot`] as a URDF XML document string.
pub fn write_urdf(robot: &Robot) -> UrdfResult<String> {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
    writeln!(out, "<robot name=\"{}\">", xml_escape(&robot.name)).ok();

    for mat in &robot.materials {
        write_material(&mut out, mat, 1);
    }
    for link in &robot.links {
        write_link(&mut out, link, 1);
    }
    for joint in &robot.joints {
        write_joint(&mut out, joint, 1);
    }

    out.push_str("</robot>\n");
    Ok(out)
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str(INDENT);
    }
}

fn write_link(out: &mut String, link: &Link, depth: usize) {
    indent(out, depth);
    writeln!(out, "<link name=\"{}\">", xml_escape(&link.name)).ok();
    if let Some(inert) = &link.inertial {
        write_inertial(out, inert, depth + 1);
    }
    for v in &link.visuals {
        write_visual(out, v, depth + 1, "visual");
    }
    for c in &link.collisions {
        write_visual(out, c, depth + 1, "collision");
    }
    indent(out, depth);
    out.push_str("</link>\n");
}

fn write_inertial(out: &mut String, inert: &Inertial, depth: usize) {
    indent(out, depth);
    out.push_str("<inertial>\n");
    write_origin(out, &inert.origin, depth + 1);
    indent(out, depth + 1);
    writeln!(out, "<mass value=\"{}\"/>", fmt_f(inert.mass)).ok();
    indent(out, depth + 1);
    writeln!(
        out,
        "<inertia ixx=\"{}\" ixy=\"{}\" ixz=\"{}\" iyy=\"{}\" iyz=\"{}\" izz=\"{}\"/>",
        fmt_f(inert.ixx),
        fmt_f(inert.ixy),
        fmt_f(inert.ixz),
        fmt_f(inert.iyy),
        fmt_f(inert.iyz),
        fmt_f(inert.izz),
    )
    .ok();
    indent(out, depth);
    out.push_str("</inertial>\n");
}

fn write_origin(out: &mut String, o: &Origin, depth: usize) {
    if o.xyz == [0.0; 3] && o.rpy == [0.0; 3] {
        return;
    }
    indent(out, depth);
    writeln!(
        out,
        "<origin xyz=\"{} {} {}\" rpy=\"{} {} {}\"/>",
        fmt_f(o.xyz[0]),
        fmt_f(o.xyz[1]),
        fmt_f(o.xyz[2]),
        fmt_f(o.rpy[0]),
        fmt_f(o.rpy[1]),
        fmt_f(o.rpy[2]),
    )
    .ok();
}

fn write_visual(out: &mut String, v: &VisualOrCollision, depth: usize, tag: &str) {
    indent(out, depth);
    match &v.name {
        Some(n) => writeln!(out, "<{tag} name=\"{}\">", xml_escape(n)).ok(),
        None => writeln!(out, "<{tag}>").ok(),
    };
    write_origin(out, &v.origin, depth + 1);
    if let Some(geom) = &v.geometry {
        write_geometry(out, geom, depth + 1);
    }
    if let Some(m) = &v.material {
        write_material(out, m, depth + 1);
    }
    indent(out, depth);
    writeln!(out, "</{tag}>").ok();
}

fn write_geometry(out: &mut String, geom: &Geometry, depth: usize) {
    indent(out, depth);
    out.push_str("<geometry>\n");
    indent(out, depth + 1);
    match &geom.kind {
        GeometryKind::Box { size } => {
            writeln!(
                out,
                "<box size=\"{} {} {}\"/>",
                fmt_f(size[0]),
                fmt_f(size[1]),
                fmt_f(size[2])
            )
            .ok();
        }
        GeometryKind::Cylinder { radius, length } => {
            writeln!(
                out,
                "<cylinder radius=\"{}\" length=\"{}\"/>",
                fmt_f(*radius),
                fmt_f(*length)
            )
            .ok();
        }
        GeometryKind::Sphere { radius } => {
            writeln!(out, "<sphere radius=\"{}\"/>", fmt_f(*radius)).ok();
        }
        GeometryKind::Mesh { filename, scale } => {
            writeln!(
                out,
                "<mesh filename=\"{}\" scale=\"{} {} {}\"/>",
                xml_escape(filename),
                fmt_f(scale[0]),
                fmt_f(scale[1]),
                fmt_f(scale[2])
            )
            .ok();
        }
    }
    indent(out, depth);
    out.push_str("</geometry>\n");
}

fn write_material(out: &mut String, m: &Material, depth: usize) {
    indent(out, depth);
    if m.color.is_none() && m.texture.is_none() {
        writeln!(out, "<material name=\"{}\"/>", xml_escape(&m.name)).ok();
        return;
    }
    writeln!(out, "<material name=\"{}\">", xml_escape(&m.name)).ok();
    if let Some(c) = m.color {
        indent(out, depth + 1);
        writeln!(
            out,
            "<color rgba=\"{} {} {} {}\"/>",
            fmt_f(c[0]),
            fmt_f(c[1]),
            fmt_f(c[2]),
            fmt_f(c[3])
        )
        .ok();
    }
    if let Some(t) = &m.texture {
        indent(out, depth + 1);
        writeln!(out, "<texture filename=\"{}\"/>", xml_escape(t)).ok();
    }
    indent(out, depth);
    out.push_str("</material>\n");
}

fn write_joint(out: &mut String, j: &Joint, depth: usize) {
    indent(out, depth);
    writeln!(
        out,
        "<joint name=\"{}\" type=\"{}\">",
        xml_escape(&j.name),
        j.kind.as_str()
    )
    .ok();
    indent(out, depth + 1);
    writeln!(out, "<parent link=\"{}\"/>", xml_escape(&j.parent)).ok();
    indent(out, depth + 1);
    writeln!(out, "<child link=\"{}\"/>", xml_escape(&j.child)).ok();
    write_origin(out, &j.origin, depth + 1);
    if j.axis != [0.0, 0.0, 1.0] {
        indent(out, depth + 1);
        writeln!(
            out,
            "<axis xyz=\"{} {} {}\"/>",
            fmt_f(j.axis[0]),
            fmt_f(j.axis[1]),
            fmt_f(j.axis[2])
        )
        .ok();
    }
    if let Some(l) = &j.limits {
        write_limits(out, l, depth + 1);
    }
    if j.dynamics != JointDynamics::default() {
        write_dynamics(out, &j.dynamics, depth + 1);
    }
    indent(out, depth);
    writeln!(out, "</joint>").ok();
}

fn write_limits(out: &mut String, l: &JointLimits, depth: usize) {
    indent(out, depth);
    writeln!(
        out,
        "<limit lower=\"{}\" upper=\"{}\" effort=\"{}\" velocity=\"{}\"/>",
        fmt_f(l.lower),
        fmt_f(l.upper),
        fmt_f(l.effort),
        fmt_f(l.velocity)
    )
    .ok();
}

fn write_dynamics(out: &mut String, d: &JointDynamics, depth: usize) {
    indent(out, depth);
    writeln!(
        out,
        "<dynamics damping=\"{}\" friction=\"{}\"/>",
        fmt_f(d.damping),
        fmt_f(d.friction)
    )
    .ok();
}

/// Compact floating-point format that matches the historical Python output
/// shape: integers as `1`, fractions as their shortest decimal repr.
fn fmt_f(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    if x == x.trunc() && x.abs() < 1e16 {
        return format!("{}", x as i64);
    }
    // Rust's default Display for f64 already produces a round-trip safe
    // representation that drops trailing zeros.
    format!("{x}")
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

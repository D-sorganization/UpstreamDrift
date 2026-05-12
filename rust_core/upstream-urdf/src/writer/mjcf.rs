//! Typed AST → MJCF (MuJoCo XML) string.
//!
//! Schema-equivalent round-trip with the parser: a parse → write →
//! parse cycle yields the same [`MujocoDocument`]. Like the URDF
//! writer, this is not a byte-stable pretty printer; it normalises
//! whitespace and attribute order while preserving the structural
//! content of the AST. Sections we have not modelled semantically are
//! emitted verbatim from [`mjcf_ast::RawSection::xml`].

use std::fmt::Write as _;

use crate::error::UrdfResult;
use crate::mjcf_ast::{
    Actuator, Asset, Body, Compiler, Geom, Inertial, Joint, MjOption, MujocoDocument, RawSection,
    Site, Worldbody,
};

const INDENT: &str = "  ";

pub fn write_mjcf(doc: &MujocoDocument) -> UrdfResult<String> {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
    if doc.model.is_empty() {
        out.push_str("<mujoco>\n");
    } else {
        writeln!(out, "<mujoco model=\"{}\">", xml_escape(&doc.model)).ok();
    }
    if let Some(c) = &doc.compiler {
        write_compiler(&mut out, c, 1);
    }
    if let Some(o) = &doc.option {
        write_option(&mut out, o, 1);
    }
    if let Some(d) = &doc.default_xml {
        write_indent(&mut out, 1);
        out.push_str(d);
        out.push('\n');
    }
    if !doc.assets.is_empty() {
        write_asset_section(&mut out, &doc.assets, 1);
    }
    write_worldbody(&mut out, &doc.worldbody, 1);
    if !doc.actuators.is_empty() {
        write_actuator_section(&mut out, &doc.actuators, 1);
    }
    for x in &doc.extras {
        write_raw_section(&mut out, x, 1);
    }
    out.push_str("</mujoco>\n");
    Ok(out)
}

fn write_indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str(INDENT);
    }
}

fn write_compiler(out: &mut String, c: &Compiler, depth: usize) {
    write_indent(out, depth);
    out.push_str("<compiler");
    if let Some(v) = &c.angle {
        write!(out, " angle=\"{}\"", xml_escape(v)).ok();
    }
    if let Some(v) = &c.coordinate {
        write!(out, " coordinate=\"{}\"", xml_escape(v)).ok();
    }
    if let Some(v) = &c.inertiafromgeom {
        write!(out, " inertiafromgeom=\"{}\"", xml_escape(v)).ok();
    }
    if let Some(v) = &c.meshdir {
        write!(out, " meshdir=\"{}\"", xml_escape(v)).ok();
    }
    if let Some(v) = &c.texturedir {
        write!(out, " texturedir=\"{}\"", xml_escape(v)).ok();
    }
    for (k, v) in &c.extra_attrs {
        write!(out, " {}=\"{}\"", k, xml_escape(v)).ok();
    }
    out.push_str("/>\n");
}

fn write_option(out: &mut String, o: &MjOption, depth: usize) {
    write_indent(out, depth);
    out.push_str("<option");
    if let Some(g) = o.gravity {
        write!(
            out,
            " gravity=\"{} {} {}\"",
            fmt_f(g[0]),
            fmt_f(g[1]),
            fmt_f(g[2])
        )
        .ok();
    }
    if let Some(t) = o.timestep {
        write!(out, " timestep=\"{}\"", fmt_f(t)).ok();
    }
    for (k, v) in &o.extra_attrs {
        write!(out, " {}=\"{}\"", k, xml_escape(v)).ok();
    }
    out.push_str("/>\n");
}

fn write_asset_section(out: &mut String, assets: &[Asset], depth: usize) {
    write_indent(out, depth);
    out.push_str("<asset>\n");
    for a in assets {
        write_indent(out, depth + 1);
        write_asset(out, a);
        out.push('\n');
    }
    write_indent(out, depth);
    out.push_str("</asset>\n");
}

fn write_asset(out: &mut String, a: &Asset) {
    match a {
        Asset::Material {
            name,
            rgba,
            specular,
            shininess,
            texture,
            extra_attrs,
        } => {
            write!(out, "<material name=\"{}\"", xml_escape(name)).ok();
            if let Some(c) = rgba {
                write!(
                    out,
                    " rgba=\"{} {} {} {}\"",
                    fmt_f(c[0]),
                    fmt_f(c[1]),
                    fmt_f(c[2]),
                    fmt_f(c[3])
                )
                .ok();
            }
            if let Some(s) = specular {
                write!(out, " specular=\"{}\"", fmt_f(*s)).ok();
            }
            if let Some(s) = shininess {
                write!(out, " shininess=\"{}\"", fmt_f(*s)).ok();
            }
            if let Some(t) = texture {
                write!(out, " texture=\"{}\"", xml_escape(t)).ok();
            }
            write_extras(out, extra_attrs);
            out.push_str("/>");
        }
        Asset::Mesh {
            name,
            file,
            scale,
            extra_attrs,
        } => {
            write!(out, "<mesh name=\"{}\"", xml_escape(name)).ok();
            if let Some(f) = file {
                write!(out, " file=\"{}\"", xml_escape(f)).ok();
            }
            if let Some(s) = scale {
                write!(
                    out,
                    " scale=\"{} {} {}\"",
                    fmt_f(s[0]),
                    fmt_f(s[1]),
                    fmt_f(s[2])
                )
                .ok();
            }
            write_extras(out, extra_attrs);
            out.push_str("/>");
        }
        Asset::Texture {
            name,
            file,
            type_,
            extra_attrs,
        } => {
            out.push_str("<texture");
            if let Some(n) = name {
                write!(out, " name=\"{}\"", xml_escape(n)).ok();
            }
            if let Some(t) = type_ {
                write!(out, " type=\"{}\"", xml_escape(t)).ok();
            }
            if let Some(f) = file {
                write!(out, " file=\"{}\"", xml_escape(f)).ok();
            }
            write_extras(out, extra_attrs);
            out.push_str("/>");
        }
    }
}

fn write_worldbody(out: &mut String, wb: &Worldbody, depth: usize) {
    write_indent(out, depth);
    out.push_str("<worldbody>\n");
    for g in &wb.geoms {
        write_geom(out, g, depth + 1);
    }
    for s in &wb.sites {
        write_site(out, s, depth + 1);
    }
    for b in &wb.bodies {
        write_body(out, b, depth + 1);
    }
    for x in &wb.extras {
        write_raw_section(out, x, depth + 1);
    }
    write_indent(out, depth);
    out.push_str("</worldbody>\n");
}

fn write_body(out: &mut String, b: &Body, depth: usize) {
    write_indent(out, depth);
    write!(out, "<body name=\"{}\"", xml_escape(&b.name)).ok();
    write!(
        out,
        " pos=\"{} {} {}\"",
        fmt_f(b.pos[0]),
        fmt_f(b.pos[1]),
        fmt_f(b.pos[2])
    )
    .ok();
    if let Some(q) = b.quat {
        write!(
            out,
            " quat=\"{} {} {} {}\"",
            fmt_f(q[0]),
            fmt_f(q[1]),
            fmt_f(q[2]),
            fmt_f(q[3])
        )
        .ok();
    }
    if let Some(e) = b.euler {
        write!(
            out,
            " euler=\"{} {} {}\"",
            fmt_f(e[0]),
            fmt_f(e[1]),
            fmt_f(e[2])
        )
        .ok();
    }
    if let Some(cc) = &b.childclass {
        write!(out, " childclass=\"{}\"", xml_escape(cc)).ok();
    }
    out.push_str(">\n");

    if let Some(inert) = &b.inertial {
        write_inertial(out, inert, depth + 1);
    }
    for j in &b.joints {
        write_joint(out, j, depth + 1);
    }
    for g in &b.geoms {
        write_geom(out, g, depth + 1);
    }
    for s in &b.sites {
        write_site(out, s, depth + 1);
    }
    for child in &b.bodies {
        write_body(out, child, depth + 1);
    }
    for x in &b.extras {
        write_raw_section(out, x, depth + 1);
    }
    write_indent(out, depth);
    out.push_str("</body>\n");
}

fn write_inertial(out: &mut String, i: &Inertial, depth: usize) {
    write_indent(out, depth);
    write!(out, "<inertial mass=\"{}\"", fmt_f(i.mass)).ok();
    write!(
        out,
        " pos=\"{} {} {}\"",
        fmt_f(i.pos[0]),
        fmt_f(i.pos[1]),
        fmt_f(i.pos[2])
    )
    .ok();
    if let Some(d) = i.diaginertia {
        write!(
            out,
            " diaginertia=\"{} {} {}\"",
            fmt_f(d[0]),
            fmt_f(d[1]),
            fmt_f(d[2])
        )
        .ok();
    }
    if let Some(f) = i.fullinertia {
        write!(
            out,
            " fullinertia=\"{} {} {} {} {} {}\"",
            fmt_f(f[0]),
            fmt_f(f[1]),
            fmt_f(f[2]),
            fmt_f(f[3]),
            fmt_f(f[4]),
            fmt_f(f[5])
        )
        .ok();
    }
    if let Some(q) = i.quat {
        write!(
            out,
            " quat=\"{} {} {} {}\"",
            fmt_f(q[0]),
            fmt_f(q[1]),
            fmt_f(q[2]),
            fmt_f(q[3])
        )
        .ok();
    }
    out.push_str("/>\n");
}

fn write_joint(out: &mut String, j: &Joint, depth: usize) {
    write_indent(out, depth);
    out.push_str("<joint");
    if let Some(n) = &j.name {
        write!(out, " name=\"{}\"", xml_escape(n)).ok();
    }
    write!(out, " type=\"{}\"", xml_escape(&j.type_)).ok();
    write!(
        out,
        " axis=\"{} {} {}\"",
        fmt_f(j.axis[0]),
        fmt_f(j.axis[1]),
        fmt_f(j.axis[2])
    )
    .ok();
    if let Some(p) = j.pos {
        write!(
            out,
            " pos=\"{} {} {}\"",
            fmt_f(p[0]),
            fmt_f(p[1]),
            fmt_f(p[2])
        )
        .ok();
    }
    if let Some(r) = j.range {
        write!(out, " range=\"{} {}\"", fmt_f(r[0]), fmt_f(r[1])).ok();
    }
    if let Some(d) = j.damping {
        write!(out, " damping=\"{}\"", fmt_f(d)).ok();
    }
    if let Some(f) = j.frictionloss {
        write!(out, " frictionloss=\"{}\"", fmt_f(f)).ok();
    }
    if let Some(a) = j.armature {
        write!(out, " armature=\"{}\"", fmt_f(a)).ok();
    }
    if let Some(s) = j.stiffness {
        write!(out, " stiffness=\"{}\"", fmt_f(s)).ok();
    }
    if let Some(c) = &j.class {
        write!(out, " class=\"{}\"", xml_escape(c)).ok();
    }
    if let Some(l) = &j.limited {
        write!(out, " limited=\"{}\"", xml_escape(l)).ok();
    }
    write_extras(out, &j.extra_attrs);
    out.push_str("/>\n");
}

fn write_geom(out: &mut String, g: &Geom, depth: usize) {
    write_indent(out, depth);
    out.push_str("<geom");
    if let Some(n) = &g.name {
        write!(out, " name=\"{}\"", xml_escape(n)).ok();
    }
    write!(out, " type=\"{}\"", xml_escape(&g.type_)).ok();
    if !g.size.is_empty() {
        out.push_str(" size=\"");
        for (i, v) in g.size.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&fmt_f(*v));
        }
        out.push('"');
    }
    if let Some(p) = g.pos {
        write!(
            out,
            " pos=\"{} {} {}\"",
            fmt_f(p[0]),
            fmt_f(p[1]),
            fmt_f(p[2])
        )
        .ok();
    }
    if let Some(q) = g.quat {
        write!(
            out,
            " quat=\"{} {} {} {}\"",
            fmt_f(q[0]),
            fmt_f(q[1]),
            fmt_f(q[2]),
            fmt_f(q[3])
        )
        .ok();
    }
    if let Some(ft) = g.fromto {
        write!(
            out,
            " fromto=\"{} {} {} {} {} {}\"",
            fmt_f(ft[0]),
            fmt_f(ft[1]),
            fmt_f(ft[2]),
            fmt_f(ft[3]),
            fmt_f(ft[4]),
            fmt_f(ft[5])
        )
        .ok();
    }
    if let Some(rgba) = g.rgba {
        write!(
            out,
            " rgba=\"{} {} {} {}\"",
            fmt_f(rgba[0]),
            fmt_f(rgba[1]),
            fmt_f(rgba[2]),
            fmt_f(rgba[3])
        )
        .ok();
    }
    if let Some(m) = &g.material {
        write!(out, " material=\"{}\"", xml_escape(m)).ok();
    }
    if let Some(m) = &g.mesh {
        write!(out, " mesh=\"{}\"", xml_escape(m)).ok();
    }
    if let Some(v) = g.mass {
        write!(out, " mass=\"{}\"", fmt_f(v)).ok();
    }
    if let Some(v) = g.density {
        write!(out, " density=\"{}\"", fmt_f(v)).ok();
    }
    if let Some(c) = &g.class {
        write!(out, " class=\"{}\"", xml_escape(c)).ok();
    }
    if let Some(v) = g.group {
        write!(out, " group=\"{v}\"").ok();
    }
    if let Some(v) = g.contype {
        write!(out, " contype=\"{v}\"").ok();
    }
    if let Some(v) = g.conaffinity {
        write!(out, " conaffinity=\"{v}\"").ok();
    }
    if let Some(f) = &g.friction {
        out.push_str(" friction=\"");
        for (i, v) in f.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&fmt_f(*v));
        }
        out.push('"');
    }
    write_extras(out, &g.extra_attrs);
    out.push_str("/>\n");
}

fn write_site(out: &mut String, s: &Site, depth: usize) {
    write_indent(out, depth);
    out.push_str("<site");
    if let Some(n) = &s.name {
        write!(out, " name=\"{}\"", xml_escape(n)).ok();
    }
    if let Some(t) = &s.type_ {
        write!(out, " type=\"{}\"", xml_escape(t)).ok();
    }
    if let Some(p) = s.pos {
        write!(
            out,
            " pos=\"{} {} {}\"",
            fmt_f(p[0]),
            fmt_f(p[1]),
            fmt_f(p[2])
        )
        .ok();
    }
    if !s.size.is_empty() {
        out.push_str(" size=\"");
        for (i, v) in s.size.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&fmt_f(*v));
        }
        out.push('"');
    }
    if let Some(r) = s.rgba {
        write!(
            out,
            " rgba=\"{} {} {} {}\"",
            fmt_f(r[0]),
            fmt_f(r[1]),
            fmt_f(r[2]),
            fmt_f(r[3])
        )
        .ok();
    }
    write_extras(out, &s.extra_attrs);
    out.push_str("/>\n");
}

fn write_actuator_section(out: &mut String, actuators: &[Actuator], depth: usize) {
    write_indent(out, depth);
    out.push_str("<actuator>\n");
    for a in actuators {
        write_indent(out, depth + 1);
        write!(out, "<{}", a.tag).ok();
        for (k, v) in &a.attrs {
            write!(out, " {}=\"{}\"", k, xml_escape(v)).ok();
        }
        out.push_str("/>\n");
    }
    write_indent(out, depth);
    out.push_str("</actuator>\n");
}

fn write_raw_section(out: &mut String, x: &RawSection, depth: usize) {
    write_indent(out, depth);
    out.push_str(&x.xml);
    out.push('\n');
}

fn write_extras(out: &mut String, extras: &[(String, String)]) {
    for (k, v) in extras {
        write!(out, " {}=\"{}\"", k, xml_escape(v)).ok();
    }
}

fn fmt_f(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    if x == x.trunc() && x.abs() < 1e16 {
        return format!("{}", x as i64);
    }
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

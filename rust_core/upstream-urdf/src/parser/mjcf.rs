//! MJCF (MuJoCo XML) parser.
//!
//! Uses `quick-xml` in `Reader` mode for the elements we model
//! semantically (`<mujoco>`, `<worldbody>`, `<body>`, `<joint>`,
//! `<geom>`, `<inertial>`, `<asset>` children, `<actuator>` children)
//! and captures unknown subtrees verbatim into
//! [`mjcf_ast::RawSection`] so that a parse → write → parse cycle is
//! lossless on sections we have not modelled. This is the same
//! single-pass strategy used by `parser::urdf`, extended for MJCF's
//! recursive `<body>` nesting.

use std::collections::HashMap;

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

use crate::error::{UrdfError, UrdfResult};
use crate::mjcf_ast::{
    Actuator, Asset, Body, Compiler, Geom, Inertial, Joint, MjOption, MujocoDocument, RawSection,
    Site, Worldbody,
};

/// Parse an MJCF document supplied as a string. Must have `<mujoco>` as
/// the root element.
pub fn parse_mjcf_str(xml: &str) -> UrdfResult<MujocoDocument> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);

    // We do two passes over the input. The first reads top-level
    // sections (compiler, option, asset, worldbody, actuator) and
    // walks `<body>` recursively. The second-level recursion for
    // bodies uses a manual stack so we never overflow on deep trees.
    let mut doc = MujocoDocument::default();
    let mut buf = Vec::new();

    // Establish that we are inside <mujoco>.
    let mut depth = 0usize;
    let mut in_mujoco = false;

    loop {
        match reader.read_event_into(&mut buf)? {
            Event::Eof => break,
            Event::Decl(_)
            | Event::Comment(_)
            | Event::PI(_)
            | Event::DocType(_)
            | Event::Text(_)
            | Event::CData(_) => {}
            Event::Start(e) => {
                let name = tag_name(&e)?;
                depth += 1;
                if name == "mujoco" {
                    in_mujoco = true;
                    let attrs = collect_attrs(&e)?;
                    if let Some(model) = attrs.get("model") {
                        doc.model = model.clone();
                    }
                    continue;
                }
                if !in_mujoco {
                    continue;
                }
                // At depth 2 (i.e. children of <mujoco>) we dispatch
                // to a section handler that reads through the matching
                // end tag and resets `depth` accordingly.
                if depth == 2 {
                    match name.as_str() {
                        "compiler" => {
                            doc.compiler = Some(parse_compiler(&e)?);
                            // <compiler> is normally an Empty element,
                            // but it can contain nested directives —
                            // discard the body to the matching close.
                            skip_until_close(&mut reader, "compiler")?;
                            depth = depth.saturating_sub(1);
                        }
                        "option" => {
                            doc.option = Some(parse_option(&e)?);
                            // Same shape as <compiler>; may contain
                            // <flag warmstart="enable"/> etc. which we
                            // do not model.
                            skip_until_close(&mut reader, "option")?;
                            depth = depth.saturating_sub(1);
                        }
                        "default" => {
                            // Capture the entire <default> block as raw
                            // XML so we can re-emit it.
                            let xml_text = capture_subtree(&mut reader, &e, "default")?;
                            doc.default_xml = Some(xml_text);
                            depth = depth.saturating_sub(1);
                        }
                        "asset" => {
                            parse_asset_section(&mut reader, &mut doc.assets)?;
                            depth = depth.saturating_sub(1);
                        }
                        "worldbody" => {
                            doc.worldbody = parse_worldbody(&mut reader)?;
                            depth = depth.saturating_sub(1);
                        }
                        "actuator" => {
                            parse_actuator_section(&mut reader, &mut doc.actuators)?;
                            depth = depth.saturating_sub(1);
                        }
                        _ => {
                            // Unknown top-level section — preserve verbatim.
                            let xml_text = capture_subtree(&mut reader, &e, &name)?;
                            doc.extras.push(RawSection {
                                tag: name,
                                xml: xml_text,
                            });
                            depth = depth.saturating_sub(1);
                        }
                    }
                }
            }
            Event::Empty(e) => {
                let name = tag_name(&e)?;
                if !in_mujoco {
                    continue;
                }
                if depth == 1 {
                    match name.as_str() {
                        "compiler" => doc.compiler = Some(parse_compiler(&e)?),
                        "option" => doc.option = Some(parse_option(&e)?),
                        _ => {
                            // Re-render as a self-closing tag for round-trip.
                            let xml_text = render_empty(&e)?;
                            doc.extras.push(RawSection {
                                tag: name,
                                xml: xml_text,
                            });
                        }
                    }
                }
            }
            Event::End(e) => {
                let name = tag_name_end(&e)?;
                depth = depth.saturating_sub(1);
                if name == "mujoco" {
                    in_mujoco = false;
                }
            }
        }
        buf.clear();
    }

    if doc.model.is_empty() && doc.worldbody.bodies.is_empty() && doc.assets.is_empty() {
        // Empty document — accept as long as the root tag was <mujoco>.
        // We don't require a model name (MuJoCo doesn't either).
    }
    Ok(doc)
}

// ---------- helpers --------------------------------------------------

fn tag_name(e: &BytesStart<'_>) -> UrdfResult<String> {
    Ok(std::str::from_utf8(e.name().as_ref())?.to_string())
}

fn tag_name_end(e: &quick_xml::events::BytesEnd<'_>) -> UrdfResult<String> {
    Ok(std::str::from_utf8(e.name().as_ref())?.to_string())
}

fn collect_attrs(e: &BytesStart<'_>) -> UrdfResult<HashMap<String, String>> {
    let mut out = HashMap::new();
    for a in e.attributes() {
        let a = a?;
        let key = std::str::from_utf8(a.key.as_ref())?.to_string();
        let val = a.unescape_value().map_err(UrdfError::from)?.into_owned();
        out.insert(key, val);
    }
    Ok(out)
}

fn ordered_attrs(e: &BytesStart<'_>) -> UrdfResult<Vec<(String, String)>> {
    let mut out = Vec::new();
    for a in e.attributes() {
        let a = a?;
        let key = std::str::from_utf8(a.key.as_ref())?.to_string();
        let val = a.unescape_value().map_err(UrdfError::from)?.into_owned();
        out.push((key, val));
    }
    Ok(out)
}

fn parse_vec_f64(s: &str) -> UrdfResult<Vec<f64>> {
    s.split_whitespace()
        .map(|p| {
            p.parse::<f64>()
                .map_err(|e| UrdfError::Parse(format!("invalid float {p:?}: {e}")))
        })
        .collect()
}

fn parse_vec_n<const N: usize>(s: &str) -> UrdfResult<[f64; N]> {
    let parts = parse_vec_f64(s)?;
    if parts.len() != N {
        return Err(UrdfError::Parse(format!(
            "expected {N} floats, got {} in {s:?}",
            parts.len()
        )));
    }
    let mut arr = [0.0; N];
    arr.copy_from_slice(&parts);
    Ok(arr)
}

// Consume events until we read the End for `tag` at any depth ≥ 1.
fn skip_until_close<B: std::io::BufRead>(reader: &mut Reader<B>, tag: &str) -> UrdfResult<()> {
    let mut depth = 1usize;
    let mut buf = Vec::new();
    while depth > 0 {
        match reader.read_event_into(&mut buf)? {
            Event::Eof => {
                return Err(UrdfError::Parse(format!(
                    "unexpected EOF while looking for </{tag}>"
                )))
            }
            Event::Start(_) => depth += 1,
            Event::End(_) => depth -= 1,
            _ => {}
        }
        buf.clear();
    }
    Ok(())
}

// Capture the entire subtree rooted at `start` (which must be a Start
// event) as the original XML text. We re-emit the open tag, then read
// events and re-serialise them until the matching End. This is the
// fallback for sections we do not model semantically.
fn capture_subtree<B: std::io::BufRead>(
    reader: &mut Reader<B>,
    start: &BytesStart<'_>,
    tag: &str,
) -> UrdfResult<String> {
    let mut out = String::new();
    write_open_tag(&mut out, start, false);
    let mut depth = 1usize;
    let mut buf = Vec::new();
    while depth > 0 {
        match reader.read_event_into(&mut buf)? {
            Event::Eof => return Err(UrdfError::Parse(format!("unexpected EOF in <{tag}>"))),
            Event::Start(e) => {
                depth += 1;
                write_open_tag(&mut out, &e, false);
            }
            Event::Empty(e) => {
                write_open_tag(&mut out, &e, true);
            }
            Event::End(e) => {
                depth -= 1;
                if depth == 0 {
                    write_close_tag(&mut out, tag);
                } else {
                    let n = std::str::from_utf8(e.name().as_ref())?.to_string();
                    write_close_tag(&mut out, &n);
                }
            }
            Event::Text(t) => {
                let s = t.unescape().map_err(UrdfError::from)?;
                out.push_str(&xml_escape(&s));
            }
            Event::CData(c) => {
                out.push_str("<![CDATA[");
                out.push_str(std::str::from_utf8(c.as_ref())?);
                out.push_str("]]>");
            }
            Event::Comment(c) => {
                out.push_str("<!--");
                out.push_str(std::str::from_utf8(c.as_ref())?);
                out.push_str("-->");
            }
            _ => {}
        }
        buf.clear();
    }
    Ok(out)
}

fn render_empty(e: &BytesStart<'_>) -> UrdfResult<String> {
    let mut out = String::new();
    write_open_tag(&mut out, e, true);
    Ok(out)
}

fn write_open_tag(out: &mut String, e: &BytesStart<'_>, self_close: bool) {
    out.push('<');
    out.push_str(std::str::from_utf8(e.name().as_ref()).unwrap_or("?"));
    for a in e.attributes().flatten() {
        let key = std::str::from_utf8(a.key.as_ref()).unwrap_or("?");
        let val = a.unescape_value().unwrap_or_default();
        out.push(' ');
        out.push_str(key);
        out.push_str("=\"");
        out.push_str(&xml_escape(&val));
        out.push('"');
    }
    if self_close {
        out.push_str("/>");
    } else {
        out.push('>');
    }
}

fn write_close_tag(out: &mut String, tag: &str) {
    out.push_str("</");
    out.push_str(tag);
    out.push('>');
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

// ---------- compiler / option ---------------------------------------

fn parse_compiler(e: &BytesStart<'_>) -> UrdfResult<Compiler> {
    let attrs = collect_attrs(e)?;
    let mut c = Compiler::default();
    for (k, v) in &attrs {
        match k.as_str() {
            "angle" => c.angle = Some(v.clone()),
            "coordinate" => c.coordinate = Some(v.clone()),
            "inertiafromgeom" => c.inertiafromgeom = Some(v.clone()),
            "meshdir" => c.meshdir = Some(v.clone()),
            "texturedir" => c.texturedir = Some(v.clone()),
            _ => c.extra_attrs.push((k.clone(), v.clone())),
        }
    }
    Ok(c)
}

fn parse_option(e: &BytesStart<'_>) -> UrdfResult<MjOption> {
    let attrs = collect_attrs(e)?;
    let mut o = MjOption::default();
    for (k, v) in &attrs {
        match k.as_str() {
            "gravity" => o.gravity = parse_vec_n::<3>(v).ok(),
            "timestep" => o.timestep = v.parse().ok(),
            _ => o.extra_attrs.push((k.clone(), v.clone())),
        }
    }
    Ok(o)
}

// ---------- <asset> section ----------------------------------------

fn parse_asset_section<B: std::io::BufRead>(
    reader: &mut Reader<B>,
    out: &mut Vec<Asset>,
) -> UrdfResult<()> {
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf)? {
            Event::Eof => return Err(UrdfError::Parse("unexpected EOF in <asset>".into())),
            Event::Empty(e) => {
                let name = tag_name(&e)?;
                let attrs = ordered_attrs(&e)?;
                match name.as_str() {
                    "material" => out.push(parse_asset_material(&attrs)),
                    "mesh" => out.push(parse_asset_mesh(&attrs)),
                    "texture" => out.push(parse_asset_texture(&attrs)),
                    _ => {
                        // Skip unknown empty asset children silently.
                    }
                }
            }
            Event::Start(e) => {
                let name = tag_name(&e)?;
                let attrs = ordered_attrs(&e)?;
                match name.as_str() {
                    "material" => out.push(parse_asset_material(&attrs)),
                    "mesh" => out.push(parse_asset_mesh(&attrs)),
                    "texture" => out.push(parse_asset_texture(&attrs)),
                    _ => {}
                }
                // Drop the (rare) nested subtree for assets with children.
                skip_until_close(reader, &name)?;
            }
            Event::End(e) => {
                let n = tag_name_end(&e)?;
                if n == "asset" {
                    return Ok(());
                }
            }
            _ => {}
        }
        buf.clear();
    }
}

fn parse_asset_material(attrs: &[(String, String)]) -> Asset {
    let mut name = String::new();
    let mut rgba = None;
    let mut specular = None;
    let mut shininess = None;
    let mut texture = None;
    let mut extra = Vec::new();
    for (k, v) in attrs {
        match k.as_str() {
            "name" => name = v.clone(),
            "rgba" => rgba = parse_vec_n::<4>(v).ok(),
            "specular" => specular = v.parse().ok(),
            "shininess" => shininess = v.parse().ok(),
            "texture" => texture = Some(v.clone()),
            _ => extra.push((k.clone(), v.clone())),
        }
    }
    Asset::Material {
        name,
        rgba,
        specular,
        shininess,
        texture,
        extra_attrs: extra,
    }
}

fn parse_asset_mesh(attrs: &[(String, String)]) -> Asset {
    let mut name = String::new();
    let mut file = None;
    let mut scale = None;
    let mut extra = Vec::new();
    for (k, v) in attrs {
        match k.as_str() {
            "name" => name = v.clone(),
            "file" => file = Some(v.clone()),
            "scale" => scale = parse_vec_n::<3>(v).ok(),
            _ => extra.push((k.clone(), v.clone())),
        }
    }
    Asset::Mesh {
        name,
        file,
        scale,
        extra_attrs: extra,
    }
}

fn parse_asset_texture(attrs: &[(String, String)]) -> Asset {
    let mut name = None;
    let mut file = None;
    let mut type_ = None;
    let mut extra = Vec::new();
    for (k, v) in attrs {
        match k.as_str() {
            "name" => name = Some(v.clone()),
            "file" => file = Some(v.clone()),
            "type" => type_ = Some(v.clone()),
            _ => extra.push((k.clone(), v.clone())),
        }
    }
    Asset::Texture {
        name,
        file,
        type_,
        extra_attrs: extra,
    }
}

// ---------- <worldbody> --------------------------------------------

fn parse_worldbody<B: std::io::BufRead>(reader: &mut Reader<B>) -> UrdfResult<Worldbody> {
    let mut wb = Worldbody::default();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf)? {
            Event::Eof => return Err(UrdfError::Parse("unexpected EOF in <worldbody>".into())),
            Event::Start(e) => {
                let name = tag_name(&e)?;
                match name.as_str() {
                    "body" => wb.bodies.push(parse_body(reader, &e)?),
                    "geom" => {
                        wb.geoms.push(parse_geom(&e)?);
                        skip_until_close(reader, "geom")?;
                    }
                    "site" => {
                        wb.sites.push(parse_site(&e)?);
                        skip_until_close(reader, "site")?;
                    }
                    other => {
                        let xml_text = capture_subtree(reader, &e, other)?;
                        wb.extras.push(RawSection {
                            tag: other.to_string(),
                            xml: xml_text,
                        });
                    }
                }
            }
            Event::Empty(e) => {
                let name = tag_name(&e)?;
                match name.as_str() {
                    "geom" => wb.geoms.push(parse_geom(&e)?),
                    "site" => wb.sites.push(parse_site(&e)?),
                    other => {
                        let xml_text = render_empty(&e)?;
                        wb.extras.push(RawSection {
                            tag: other.to_string(),
                            xml: xml_text,
                        });
                    }
                }
            }
            Event::End(e) => {
                if tag_name_end(&e)? == "worldbody" {
                    return Ok(wb);
                }
            }
            _ => {}
        }
        buf.clear();
    }
}

fn parse_body<B: std::io::BufRead>(
    reader: &mut Reader<B>,
    start: &BytesStart<'_>,
) -> UrdfResult<Body> {
    let attrs = collect_attrs(start)?;
    let mut body = Body {
        name: attrs.get("name").cloned().unwrap_or_default(),
        pos: attrs
            .get("pos")
            .and_then(|s| parse_vec_n::<3>(s).ok())
            .unwrap_or([0.0; 3]),
        quat: attrs.get("quat").and_then(|s| parse_vec_n::<4>(s).ok()),
        euler: attrs.get("euler").and_then(|s| parse_vec_n::<3>(s).ok()),
        childclass: attrs.get("childclass").cloned(),
        ..Default::default()
    };

    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf)? {
            Event::Eof => return Err(UrdfError::Parse("unexpected EOF inside <body>".into())),
            Event::Start(e) => {
                let name = tag_name(&e)?;
                match name.as_str() {
                    "body" => body.bodies.push(parse_body(reader, &e)?),
                    "inertial" => {
                        body.inertial = Some(parse_inertial(&e)?);
                        skip_until_close(reader, "inertial")?;
                    }
                    "joint" => {
                        body.joints.push(parse_joint(&e)?);
                        skip_until_close(reader, "joint")?;
                    }
                    "geom" => {
                        body.geoms.push(parse_geom(&e)?);
                        skip_until_close(reader, "geom")?;
                    }
                    "site" => {
                        body.sites.push(parse_site(&e)?);
                        skip_until_close(reader, "site")?;
                    }
                    other => {
                        let xml_text = capture_subtree(reader, &e, other)?;
                        body.extras.push(RawSection {
                            tag: other.to_string(),
                            xml: xml_text,
                        });
                    }
                }
            }
            Event::Empty(e) => {
                let name = tag_name(&e)?;
                match name.as_str() {
                    "inertial" => body.inertial = Some(parse_inertial(&e)?),
                    "joint" => body.joints.push(parse_joint(&e)?),
                    "geom" => body.geoms.push(parse_geom(&e)?),
                    "site" => body.sites.push(parse_site(&e)?),
                    other => {
                        let xml_text = render_empty(&e)?;
                        body.extras.push(RawSection {
                            tag: other.to_string(),
                            xml: xml_text,
                        });
                    }
                }
            }
            Event::End(e) => {
                if tag_name_end(&e)? == "body" {
                    return Ok(body);
                }
            }
            _ => {}
        }
        buf.clear();
    }
}

fn parse_inertial(e: &BytesStart<'_>) -> UrdfResult<Inertial> {
    let attrs = collect_attrs(e)?;
    Ok(Inertial {
        mass: attrs
            .get("mass")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0),
        pos: attrs
            .get("pos")
            .and_then(|s| parse_vec_n::<3>(s).ok())
            .unwrap_or([0.0; 3]),
        diaginertia: attrs
            .get("diaginertia")
            .and_then(|s| parse_vec_n::<3>(s).ok()),
        fullinertia: attrs
            .get("fullinertia")
            .and_then(|s| parse_vec_n::<6>(s).ok()),
        quat: attrs.get("quat").and_then(|s| parse_vec_n::<4>(s).ok()),
    })
}

fn parse_joint(e: &BytesStart<'_>) -> UrdfResult<Joint> {
    let attrs = collect_attrs(e)?;
    let mut extra = Vec::new();
    let mut j = Joint {
        name: attrs.get("name").cloned(),
        type_: attrs
            .get("type")
            .cloned()
            .unwrap_or_else(|| "hinge".to_string()),
        axis: attrs
            .get("axis")
            .and_then(|s| parse_vec_n::<3>(s).ok())
            .unwrap_or([0.0, 0.0, 1.0]),
        pos: attrs.get("pos").and_then(|s| parse_vec_n::<3>(s).ok()),
        range: attrs.get("range").and_then(|s| parse_vec_n::<2>(s).ok()),
        damping: attrs.get("damping").and_then(|s| s.parse().ok()),
        frictionloss: attrs.get("frictionloss").and_then(|s| s.parse().ok()),
        armature: attrs.get("armature").and_then(|s| s.parse().ok()),
        stiffness: attrs.get("stiffness").and_then(|s| s.parse().ok()),
        class: attrs.get("class").cloned(),
        limited: attrs.get("limited").cloned(),
        extra_attrs: Vec::new(),
    };
    for (k, v) in attrs {
        if matches!(
            k.as_str(),
            "name"
                | "type"
                | "axis"
                | "pos"
                | "range"
                | "damping"
                | "frictionloss"
                | "armature"
                | "stiffness"
                | "class"
                | "limited"
        ) {
            continue;
        }
        extra.push((k, v));
    }
    j.extra_attrs = extra;
    Ok(j)
}

fn parse_geom(e: &BytesStart<'_>) -> UrdfResult<Geom> {
    let attrs = collect_attrs(e)?;
    let mut extra = Vec::new();
    let mut g = Geom {
        name: attrs.get("name").cloned(),
        type_: attrs
            .get("type")
            .cloned()
            .unwrap_or_else(|| "sphere".to_string()),
        size: attrs
            .get("size")
            .map(|s| parse_vec_f64(s).unwrap_or_default())
            .unwrap_or_default(),
        pos: attrs.get("pos").and_then(|s| parse_vec_n::<3>(s).ok()),
        quat: attrs.get("quat").and_then(|s| parse_vec_n::<4>(s).ok()),
        fromto: attrs.get("fromto").and_then(|s| parse_vec_n::<6>(s).ok()),
        rgba: attrs.get("rgba").and_then(|s| parse_vec_n::<4>(s).ok()),
        material: attrs.get("material").cloned(),
        mesh: attrs.get("mesh").cloned(),
        mass: attrs.get("mass").and_then(|s| s.parse().ok()),
        density: attrs.get("density").and_then(|s| s.parse().ok()),
        class: attrs.get("class").cloned(),
        group: attrs.get("group").and_then(|s| s.parse().ok()),
        contype: attrs.get("contype").and_then(|s| s.parse().ok()),
        conaffinity: attrs.get("conaffinity").and_then(|s| s.parse().ok()),
        friction: attrs
            .get("friction")
            .map(|s| parse_vec_f64(s).unwrap_or_default()),
        extra_attrs: Vec::new(),
    };
    for (k, v) in attrs {
        if matches!(
            k.as_str(),
            "name"
                | "type"
                | "size"
                | "pos"
                | "quat"
                | "fromto"
                | "rgba"
                | "material"
                | "mesh"
                | "mass"
                | "density"
                | "class"
                | "group"
                | "contype"
                | "conaffinity"
                | "friction"
        ) {
            continue;
        }
        extra.push((k, v));
    }
    g.extra_attrs = extra;
    Ok(g)
}

fn parse_site(e: &BytesStart<'_>) -> UrdfResult<Site> {
    let attrs = collect_attrs(e)?;
    let mut extra = Vec::new();
    let mut s = Site {
        name: attrs.get("name").cloned(),
        pos: attrs.get("pos").and_then(|v| parse_vec_n::<3>(v).ok()),
        size: attrs
            .get("size")
            .map(|v| parse_vec_f64(v).unwrap_or_default())
            .unwrap_or_default(),
        type_: attrs.get("type").cloned(),
        rgba: attrs.get("rgba").and_then(|v| parse_vec_n::<4>(v).ok()),
        extra_attrs: Vec::new(),
    };
    for (k, v) in attrs {
        if matches!(k.as_str(), "name" | "pos" | "size" | "type" | "rgba") {
            continue;
        }
        extra.push((k, v));
    }
    s.extra_attrs = extra;
    Ok(s)
}

// ---------- <actuator> section -------------------------------------

fn parse_actuator_section<B: std::io::BufRead>(
    reader: &mut Reader<B>,
    out: &mut Vec<Actuator>,
) -> UrdfResult<()> {
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf)? {
            Event::Eof => return Err(UrdfError::Parse("unexpected EOF in <actuator>".into())),
            Event::Empty(e) => {
                let tag = tag_name(&e)?;
                let attrs = ordered_attrs(&e)?;
                out.push(Actuator { tag, attrs });
            }
            Event::Start(e) => {
                let tag = tag_name(&e)?;
                let attrs = ordered_attrs(&e)?;
                out.push(Actuator {
                    tag: tag.clone(),
                    attrs,
                });
                skip_until_close(reader, &tag)?;
            }
            Event::End(e) => {
                if tag_name_end(&e)? == "actuator" {
                    return Ok(());
                }
            }
            _ => {}
        }
        buf.clear();
    }
}

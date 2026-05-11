//! BVH (BioVision Hierarchy) parser.
//!
//! Two sections:
//!   * `HIERARCHY` — joint tree (ROOT / JOINT / End Site, with OFFSET and
//!     CHANNELS, brace-nested)
//!   * `MOTION` — `Frames: N`, `Frame Time: dt`, then N rows of channel
//!     values (rotations in degrees, translations in source units)
//!
//! The Python facade owns construction of the `SkeletonRig` and the
//! degree→radian conversion of rotational channels (the Python adapter
//! converts *all* channels via `np.deg2rad`, which is the existing legacy
//! behaviour we must preserve byte-for-byte). The Rust side therefore
//! returns the raw motion matrix unchanged (still in degrees) and lets
//! the facade do the deg→rad pass.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::{JointData, JointInfo, ParseError};

pub fn parse_bvh_file(path: &Path) -> Result<JointData, ParseError> {
    let mut file = File::open(path)?;
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    parse_bvh_text(&text)
}

pub fn parse_bvh_text(text: &str) -> Result<JointData, ParseError> {
    // Split HIERARCHY / MOTION sections.
    let upper = text.to_ascii_uppercase();
    let motion_idx = upper
        .find("\nMOTION")
        .ok_or_else(|| ParseError::Format("BVH file missing MOTION section".into()))?;
    let hierarchy = &text[..motion_idx];
    let motion = &text[motion_idx + 1..]; // skip leading newline

    let joints = parse_hierarchy(hierarchy)?;

    // Motion section header.
    let mut lines = motion.lines();
    // First line should be "MOTION"
    let first = lines
        .next()
        .ok_or_else(|| ParseError::Format("BVH MOTION section empty".into()))?;
    if !first.trim().eq_ignore_ascii_case("MOTION") {
        return Err(ParseError::Format(
            "BVH MOTION section header missing".into(),
        ));
    }

    let mut frames_count: usize = 0;
    let mut frame_time: f32 = 1.0 / 30.0;
    let mut data_lines: Vec<&str> = Vec::new();
    let mut in_data = false;
    for line in lines {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        if !in_data {
            let upper = s.to_ascii_uppercase();
            if let Some(rest) = upper.strip_prefix("FRAMES:") {
                frames_count = rest.trim().parse().unwrap_or(0);
                continue;
            }
            if let Some(rest) = upper.strip_prefix("FRAME TIME:") {
                frame_time = rest.trim().parse().unwrap_or(1.0 / 30.0);
                if frame_time <= 0.0 {
                    frame_time = 1.0 / 30.0;
                }
                in_data = true;
                continue;
            }
            // First non-header numeric-looking line implies header missing/done.
            in_data = true;
        }
        data_lines.push(s);
    }

    let num_dofs: usize = joints.iter().map(|j| j.channels.len()).sum();
    // If skeleton has no channels (degenerate), fall back to 3 (root rot) so
    // downstream Python contract doesn't crash. The Python facade detects
    // this case via `skeleton.num_dofs == 0` and skips frame construction.
    let num_dofs_eff = num_dofs.max(1);
    let mut motion_buf = Vec::with_capacity(data_lines.len() * num_dofs_eff);

    for raw in &data_lines {
        let mut values: Vec<f32> = raw
            .split_whitespace()
            .filter_map(|tok| tok.parse::<f32>().ok())
            .collect();
        // Python facade pads or truncates to num_dofs; mirror that behavior.
        if values.len() < num_dofs {
            values.resize(num_dofs, 0.0);
        } else if values.len() > num_dofs {
            values.truncate(num_dofs);
        }
        motion_buf.extend_from_slice(&values);
    }

    let n_frames_actual = if num_dofs == 0 {
        0
    } else {
        motion_buf.len() / num_dofs_eff
    };
    // Prefer "Frames:" header count when consistent; otherwise use what we parsed.
    let n_frames = if frames_count > 0 && frames_count <= n_frames_actual {
        frames_count
    } else {
        n_frames_actual
    };
    // Truncate the buffer to match.
    motion_buf.truncate(n_frames * num_dofs_eff);

    let fps = 1.0 / frame_time;

    Ok(JointData {
        joints,
        motion: motion_buf,
        n_frames,
        num_dofs,
        fps,
    })
}

fn parse_hierarchy(text: &str) -> Result<Vec<JointInfo>, ParseError> {
    let mut joints: Vec<JointInfo> = Vec::new();
    // Stack of indices into `joints` for the current parent chain.
    let mut stack: Vec<usize> = Vec::new();
    // Whether the next token closing `{` belongs to an End Site (skip joint).
    let mut pending_end_site = false;
    let mut current_idx: Option<usize> = None;

    let mut tokens = text.split_whitespace().peekable();

    while let Some(tok) = tokens.next() {
        match tok.to_ascii_uppercase().as_str() {
            "HIERARCHY" => continue,
            "ROOT" | "JOINT" => {
                let name = tokens
                    .next()
                    .ok_or_else(|| ParseError::Format("Missing joint name".into()))?
                    .to_string();
                let parent = stack.last().copied();
                joints.push(JointInfo {
                    name,
                    parent,
                    channels: Vec::new(),
                });
                current_idx = Some(joints.len() - 1);
            }
            "END" => {
                // "End Site" — anonymous leaf
                if let Some(next) = tokens.next() {
                    if next.eq_ignore_ascii_case("Site") {
                        pending_end_site = true;
                    }
                }
            }
            "{" => {
                if pending_end_site {
                    // Will pop a depth on matching '}', but we did not push a real joint.
                    // Push a sentinel value to track the brace depth without a joint.
                    pending_end_site = false;
                    // We track end-site depth with usize::MAX sentinel.
                    stack.push(usize::MAX);
                } else if let Some(idx) = current_idx {
                    stack.push(idx);
                    current_idx = None;
                }
            }
            "}" => {
                stack.pop();
            }
            "OFFSET" => {
                // Skip three floats.
                for _ in 0..3 {
                    let _ = tokens.next();
                }
            }
            "CHANNELS" => {
                let n: usize = tokens
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| ParseError::Format("Bad CHANNELS count".into()))?;
                let mut chans = Vec::with_capacity(n);
                for _ in 0..n {
                    let c = tokens
                        .next()
                        .ok_or_else(|| ParseError::Format("Truncated CHANNELS".into()))?
                        .to_string();
                    chans.push(c);
                }
                // CHANNELS belongs to the most-recently-opened real joint.
                if let Some(&top) = stack.iter().rev().find(|&&i| i != usize::MAX) {
                    joints[top].channels = chans;
                }
            }
            _ => {
                // Unknown token; ignore.
            }
        }
    }

    if joints.is_empty() {
        return Err(ParseError::Format("BVH file has no joints".into()));
    }

    Ok(joints)
}

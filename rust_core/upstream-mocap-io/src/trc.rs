//! TRC (OpenSim Track Row Column) parser.
//!
//! Header layout (tab-separated):
//!   line 0: `PathFileType  4  (X/Y/Z)  <filename>`
//!   line 1: keys (DataRate, CameraRate, NumFrames, NumMarkers, Units, ...)
//!   line 2: values for those keys
//!   line 3: marker name row — `Frame#\tTime\tMarker1\t\t\tMarker2\t\t\t...`
//!   line 4: `\t\tX1\tY1\tZ1\tX2\tY2\tZ2 ...`
//!   line 5+: data rows `frame_idx, time, x1, y1, z1, x2, y2, z2, ...`
//!
//! Data may use whitespace or tab separators. We accept any whitespace
//! per the existing Python adapter's behaviour.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::{MarkerData, ParseError};

pub fn parse_trc_file(path: &Path) -> Result<MarkerData, ParseError> {
    let mut file = File::open(path)?;
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    parse_trc_text(&text)
}

pub fn parse_trc_text(text: &str) -> Result<MarkerData, ParseError> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 5 {
        return Err(ParseError::Format(
            "TRC file truncated; need at least 5 header rows".into(),
        ));
    }

    // Header keys + values (line 1 + line 2).
    let keys: Vec<&str> = lines[1].split_whitespace().collect();
    let values: Vec<&str> = lines[2].split_whitespace().collect();

    let mut fps: f32 = 0.0;
    let mut camera_rate: f32 = 0.0;
    let mut num_frames: usize = 0;
    let mut units_str = String::from("mm");
    for (k, v) in keys.iter().zip(values.iter()) {
        match k.to_ascii_lowercase().as_str() {
            "datarate" => fps = v.parse().unwrap_or(0.0),
            "camerarate" => camera_rate = v.parse().unwrap_or(0.0),
            "numframes" => num_frames = v.parse().unwrap_or(0),
            "units" => units_str = v.trim().to_string(),
            _ => {}
        }
    }
    if fps <= 0.0 {
        fps = if camera_rate > 0.0 { camera_rate } else { 30.0 };
    }
    let _ = num_frames; // Trust actual data row count.

    // Marker names — split line 3 by TAB and strip empties.
    let marker_tokens: Vec<&str> = lines[3].split('\t').collect();
    // First two tokens are "Frame#" and "Time"; the rest are marker names
    // with empty spacer tokens between them.
    let marker_names: Vec<String> = marker_tokens
        .iter()
        .skip(2)
        .filter_map(|t| {
            let s = t.trim();
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        })
        .collect();
    let n_markers = marker_names.len();

    let units_lc = units_str.to_ascii_lowercase();
    let scale: f32 = if units_lc.starts_with("mm") {
        0.001
    } else {
        1.0
    };

    let mut positions: Vec<f32> = Vec::new();
    let mut n_frames_actual: usize = 0;

    // Data rows start at line index 5.
    for line in lines.iter().skip(5) {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = s.split_whitespace().collect();
        if tokens.len() < 2 {
            continue;
        }
        // Skip frame_idx (tokens[0]) and time (tokens[1]); parse n_markers * 3 floats.
        let coords = &tokens[2..];
        for m_i in 0..n_markers {
            let base = m_i * 3;
            if base + 2 >= coords.len() {
                positions.push(f32::NAN);
                positions.push(f32::NAN);
                positions.push(f32::NAN);
                continue;
            }
            let x = coords[base].parse::<f32>();
            let y = coords[base + 1].parse::<f32>();
            let z = coords[base + 2].parse::<f32>();
            match (x, y, z) {
                (Ok(x), Ok(y), Ok(z)) => {
                    positions.push(x * scale);
                    positions.push(y * scale);
                    positions.push(z * scale);
                }
                _ => {
                    positions.push(f32::NAN);
                    positions.push(f32::NAN);
                    positions.push(f32::NAN);
                }
            }
        }
        n_frames_actual += 1;
    }

    if n_frames_actual == 0 {
        return Err(ParseError::Format("TRC file has no data rows".into()));
    }

    Ok(MarkerData {
        names: marker_names,
        positions,
        n_frames: n_frames_actual,
        n_markers,
        fps,
        units: units_str,
    })
}

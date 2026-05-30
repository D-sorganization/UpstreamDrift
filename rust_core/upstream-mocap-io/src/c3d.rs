//! C3D binary parser (markers only).
//!
//! C3D is a 512-byte-block binary format with three sections:
//!   1. Header block (block 1, 512 bytes)
//!   2. Parameter section (blocks starting at `header[0]`)
//!   3. Data section (blocks starting at `header[16]` aka `start_block`)
//!
//! We support the most common encodings:
//!   * Intel little-endian (processor type 0x54)
//!   * IEEE big-endian   (processor type 0x55)
//!   * DEC Vax (processor type 0x56) — basic support, no actual DEC float
//!     conversion. Files written in this mode are nearly extinct.
//!   * Float (scale < 0) and int16 (scale > 0) point storage
//!
//! Out of scope (deferred follow-up to issue #5213):
//!   * Analog channels
//!   * Event labels / event times
//!   * Custom parameter groups beyond POINT:RATE/LABELS/UNITS/SCALE/FRAMES
//!
//! Reference: <https://www.c3d.org/docs/C3D_User_Guide.pdf>

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use byteorder::{BigEndian, ByteOrder, LittleEndian};

use crate::{MarkerData, ParseError};

const BLOCK_SIZE: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

impl Endian {
    fn read_i16(self, bytes: &[u8]) -> i16 {
        match self {
            Endian::Little => LittleEndian::read_i16(bytes),
            Endian::Big => BigEndian::read_i16(bytes),
        }
    }
    fn read_u16(self, bytes: &[u8]) -> u16 {
        match self {
            Endian::Little => LittleEndian::read_u16(bytes),
            Endian::Big => BigEndian::read_u16(bytes),
        }
    }
    fn read_f32(self, bytes: &[u8]) -> f32 {
        match self {
            Endian::Little => LittleEndian::read_f32(bytes),
            Endian::Big => BigEndian::read_f32(bytes),
        }
    }
}

/// Parse a C3D file from disk. Returns marker positions in **meters**
/// (mm-scaled if UNITS starts with "mm", otherwise treated as already
/// in meters). Occluded markers (negative residual in int16 mode or
/// NaN in float mode) become `f32::NAN`.
pub fn parse_c3d_file(path: &Path) -> Result<MarkerData, ParseError> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    parse_c3d_bytes(&buf)
}

/// Parse a C3D file from an in-memory buffer.
pub fn parse_c3d_bytes(buf: &[u8]) -> Result<MarkerData, ParseError> {
    if buf.len() < BLOCK_SIZE {
        return Err(ParseError::Format("C3D file shorter than one block".into()));
    }

    // Determine endian by inspecting the parameter section's processor-type byte.
    // Header itself contains 16-bit values; the C3D format places the parameter
    // section pointer at byte 0 (1-based block) and the magic 0x50 at byte 1.
    if buf[1] != 0x50 {
        return Err(ParseError::Format(
            "C3D magic byte 0x50 not found at byte 1".into(),
        ));
    }

    // Parameter section first byte is a "reserved" zero; second is the magic
    // 0x50; third (offset +2) is the number of parameter blocks; fourth
    // (offset +3) is the processor type: 0x54=Intel LE, 0x55=DEC, 0x56=MIPS BE.
    let param_block_1based = buf[0] as usize;
    if param_block_1based == 0 {
        return Err(ParseError::Format("C3D header points to block 0".into()));
    }
    let param_offset = (param_block_1based - 1) * BLOCK_SIZE;
    if buf.len() < param_offset + 4 {
        return Err(ParseError::Format("Parameter section truncated".into()));
    }
    let processor = buf[param_offset + 3];
    let endian = match processor {
        0x54 => Endian::Little,     // Intel
        0x55 | 0x56 => Endian::Big, // DEC / SGI MIPS — both treated as BE for our purposes
        _ => Endian::Little,        // Default to LE; most modern C3D writers use Intel
    };

    // ── Header (block 1) ────────────────────────────────────────────────
    // Layout (16-bit words, 1-based as in the spec):
    //   word 1 (byte 0): parameter section first block (u8 lo) | 0x50 (u8 hi)
    //   word 2 (byte 2): number of 3D points (markers)
    //   word 3 (byte 4): number of analog measurements per 3D frame
    //   word 4 (byte 6): first 3D frame number (1-based)
    //   word 5 (byte 8): last 3D frame number (1-based, inclusive)
    //   word 6 (byte 10): max interpolation gap
    //   word 7 (byte 12, two words): scaling factor as 32-bit float
    //                                (sign indicates encoding)
    //   word 9 (byte 16): start of data section block (1-based)
    //   word 10 (byte 18): analog samples per 3D frame
    //   word 11 (byte 20, two words): frame rate (32-bit float)
    let n_points = endian.read_u16(&buf[2..4]) as usize;
    let first_frame = endian.read_u16(&buf[6..8]) as i32;
    let last_frame = endian.read_u16(&buf[8..10]) as i32;
    let scale = endian.read_f32(&buf[12..16]);
    let data_block_1based = endian.read_u16(&buf[16..18]) as usize;
    let fps_header = endian.read_f32(&buf[20..24]);

    if data_block_1based == 0 {
        return Err(ParseError::Format(
            "C3D data section pointer is zero".into(),
        ));
    }
    let n_frames_header = if last_frame >= first_frame {
        (last_frame - first_frame + 1) as usize
    } else {
        0
    };

    // ── Parameter section ──────────────────────────────────────────────
    // Walk the parameters; we only need POINT:LABELS, POINT:UNITS,
    // POINT:RATE, and optionally POINT:FRAMES (for long files where header
    // n_frames overflows u16).
    let (labels, units, fps_param, frames_param) = parse_point_params(buf, param_offset, endian)?;

    let fps = if fps_param > 0.0 {
        fps_param
    } else if fps_header > 0.0 {
        fps_header
    } else {
        30.0
    };

    let n_frames = if frames_param > 0 {
        frames_param
    } else {
        n_frames_header
    };

    // Use parameter-derived label count when available (header n_points can
    // include extra slots beyond labelled markers).
    let n_markers = if !labels.is_empty() {
        labels.len().min(n_points)
    } else {
        n_points
    };

    // ── Data section ────────────────────────────────────────────────────
    let data_offset = (data_block_1based - 1) * BLOCK_SIZE;
    let float_mode = scale < 0.0;
    let abs_scale = if scale == 0.0 { 1.0 } else { scale.abs() };
    let bytes_per_value = 4; // both int16 + residual and float use 4 bytes * 4 values = 16 bytes/marker
    let stride = n_points * bytes_per_value * 4 / bytes_per_value; // = n_points * 4 values
    let bytes_per_frame = n_points * 4 * 4; // 4 values (x, y, z, residual) * 4 bytes each (float) or 2*2 (int16+residual)
    let int_bytes_per_frame = n_points * 4 * 2; // 4 i16 per marker
    let _ = stride;

    let positions_capacity = n_frames * n_markers * 3;
    let mut positions = Vec::with_capacity(positions_capacity);

    // Match the existing Python adapter convention: an empty/missing UNITS
    // parameter defaults to millimetres (the de-facto C3D convention).
    let units_for_scale = if units.is_empty() {
        "mm"
    } else {
        units.as_str()
    };
    let unit_scale: f32 = if units_for_scale.to_ascii_lowercase().starts_with("mm") {
        0.001
    } else {
        1.0
    };

    for fi in 0..n_frames {
        let frame_start = data_offset
            + fi * if float_mode {
                bytes_per_frame
            } else {
                int_bytes_per_frame
            };
        for mi in 0..n_points {
            if mi >= n_markers {
                // Skip extra unlabeled marker slots
                continue;
            }
            let (x, y, z, residual) = if float_mode {
                let off = frame_start + mi * 16;
                if buf.len() < off + 16 {
                    return Err(ParseError::Format("C3D data truncated (float)".into()));
                }
                (
                    endian.read_f32(&buf[off..off + 4]),
                    endian.read_f32(&buf[off + 4..off + 8]),
                    endian.read_f32(&buf[off + 8..off + 12]),
                    endian.read_f32(&buf[off + 12..off + 16]),
                )
            } else {
                let off = frame_start + mi * 8;
                if buf.len() < off + 8 {
                    return Err(ParseError::Format("C3D data truncated (int)".into()));
                }
                let xi = endian.read_i16(&buf[off..off + 2]) as f32;
                let yi = endian.read_i16(&buf[off + 2..off + 4]) as f32;
                let zi = endian.read_i16(&buf[off + 4..off + 6]) as f32;
                let resid = endian.read_i16(&buf[off + 6..off + 8]) as f32;
                (xi * abs_scale, yi * abs_scale, zi * abs_scale, resid)
            };
            // Occlusion: residual < 0 (int mode) or NaN (float mode)
            if residual < 0.0 || !x.is_finite() || !y.is_finite() || !z.is_finite() {
                positions.push(f32::NAN);
                positions.push(f32::NAN);
                positions.push(f32::NAN);
            } else {
                positions.push(x * unit_scale);
                positions.push(y * unit_scale);
                positions.push(z * unit_scale);
            }
        }
    }

    Ok(MarkerData {
        names: labels.into_iter().take(n_markers).collect(),
        positions,
        n_frames,
        n_markers,
        fps,
        units,
    })
}

/// Walk the C3D parameter section pulling out POINT:LABELS, POINT:UNITS,
/// POINT:RATE, and optionally POINT:FRAMES.
fn parse_point_params(
    buf: &[u8],
    param_section_offset: usize,
    endian: Endian,
) -> Result<(Vec<String>, String, f32, usize), ParseError> {
    // After the 4-byte header (reserved, 0x50, n_param_blocks, processor),
    // parameter groups + items follow as a linked list with relative offsets.
    let mut pos = param_section_offset + 4;
    let mut labels: Vec<String> = Vec::new();
    let mut units = String::new();
    let mut fps: f32 = 0.0;
    let mut frames: usize = 0;

    // Group ID → name map. Group definitions have id < 0; items have id > 0.
    let mut group_names: std::collections::HashMap<i8, String> = std::collections::HashMap::new();

    // Safety cap: at most 128 blocks of params.
    let max_pos = (param_section_offset + 128 * BLOCK_SIZE).min(buf.len());

    while pos + 2 < max_pos {
        let name_len_signed = buf[pos] as i8;
        let group_id_signed = buf[pos + 1] as i8;
        if name_len_signed == 0 && group_id_signed == 0 {
            break;
        }
        let name_len = name_len_signed.unsigned_abs() as usize;
        pos += 2;
        if pos + name_len > buf.len() {
            break;
        }
        let name = String::from_utf8_lossy(&buf[pos..pos + name_len])
            .trim()
            .to_uppercase();
        pos += name_len;
        if pos + 2 > buf.len() {
            break;
        }
        let next_offset = endian.read_i16(&buf[pos..pos + 2]) as i32;
        let next_pos_in_record_start = pos; // offset is relative to here
        pos += 2;

        if group_id_signed < 0 {
            // Group definition. Skip description (length-prefixed byte).
            if pos >= buf.len() {
                break;
            }
            let _desc_len = buf[pos] as usize;
            // pos jump is governed by next_offset below; don't advance here.
            group_names.insert(-group_id_signed, name);
        } else {
            // Parameter item.
            let group_name = group_names
                .get(&group_id_signed)
                .cloned()
                .unwrap_or_default();
            if pos + 1 > buf.len() {
                break;
            }
            let element_size = buf[pos] as i8;
            pos += 1;
            if pos + 1 > buf.len() {
                break;
            }
            let n_dims = buf[pos] as usize;
            pos += 1;
            if pos + n_dims > buf.len() {
                break;
            }
            let mut dims = Vec::with_capacity(n_dims);
            let mut total: usize = 1;
            for _ in 0..n_dims {
                let d = buf[pos] as usize;
                dims.push(d);
                total = total.saturating_mul(d.max(1));
                pos += 1;
            }
            // Read raw data block.
            let elem_bytes = match element_size {
                -1 => 1,
                1 => 1,
                2 => 2,
                4 => 4,
                _ => 1,
            };
            let total_bytes = total.saturating_mul(elem_bytes as usize);
            if pos + total_bytes > buf.len() {
                break;
            }
            let data = &buf[pos..pos + total_bytes];
            pos += total_bytes;
            // Skip description (length-prefixed byte). pos is overwritten
            // by the next_offset jump at the bottom of the loop, so no
            // explicit advancement here.
            if pos < buf.len() {
                let _desc_len = buf[pos] as usize;
            }

            if group_name == "POINT" {
                match name.as_str() {
                    "LABELS"
                        // 2D char array: dims = [string_len, n_strings]
                        if element_size == -1 && dims.len() == 2 => {
                            let slen = dims[0];
                            let n_str = dims[1];
                            for i in 0..n_str {
                                let off = i * slen;
                                if off + slen > data.len() {
                                    break;
                                }
                                let s = String::from_utf8_lossy(&data[off..off + slen])
                                    .trim()
                                    .to_string();
                                labels.push(s);
                            }
                        }
                    "UNITS"
                        if element_size == -1 && dims.len() == 2 => {
                            let slen = dims[0];
                            // First string is the unit.
                            if slen <= data.len() {
                                units = String::from_utf8_lossy(&data[..slen]).trim().to_string();
                            }
                        }
                    "RATE"
                        if element_size == 4 && data.len() >= 4 => {
                            fps = endian.read_f32(&data[..4]);
                        }
                    "FRAMES" => {
                        if element_size == 2 && data.len() >= 2 {
                            frames = endian.read_u16(&data[..2]) as usize;
                        } else if element_size == 4 && data.len() >= 4 {
                            frames = endian.read_f32(&data[..4]) as usize;
                        }
                    }
                    _ => {}
                }
            }
        }

        if next_offset <= 0 {
            // End of parameter linked list.
            break;
        }
        // Advance pos to the position recorded by next_offset (relative to
        // the offset field itself).
        pos = next_pos_in_record_start + next_offset as usize;
    }

    Ok((labels, units, fps, frames))
}

#[allow(dead_code)]
fn read_block<R: Read + Seek>(r: &mut R, block_1based: usize) -> std::io::Result<[u8; BLOCK_SIZE]> {
    let mut blk = [0u8; BLOCK_SIZE];
    r.seek(SeekFrom::Start(((block_1based - 1) * BLOCK_SIZE) as u64))?;
    r.read_exact(&mut blk)?;
    Ok(blk)
}

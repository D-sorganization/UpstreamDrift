//! C3D binary parser (markers, event metadata, analog channels, force plates).
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
//!   * Custom parameter groups beyond POINT, EVENT, ANALOG, and FORCE_PLATFORM
//!
//! Reference: <https://www.c3d.org/docs/C3D_User_Guide.pdf>

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use byteorder::{BigEndian, ByteOrder, LittleEndian};

use crate::{C3dAnalogData, C3dEvent, C3dForcePlatform, MarkerData, ParseError};

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
    let analog_measurements_per_frame = endian.read_u16(&buf[4..6]) as usize;
    let first_frame = endian.read_u16(&buf[6..8]) as i32;
    let last_frame = endian.read_u16(&buf[8..10]) as i32;
    let scale = endian.read_f32(&buf[12..16]);
    let data_block_1based = endian.read_u16(&buf[16..18]) as usize;
    let analog_samples_per_frame_header = endian.read_u16(&buf[18..20]) as usize;
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
    // Walk the parameter section. Marker data depends on POINT metadata; event
    // metadata is optional and never affects marker parsing.
    let params = parse_c3d_params(buf, param_offset, endian)?;
    let labels = params.labels;
    let units = params.units;
    let events = params.events;
    let force_platforms = params.force_platforms;

    let fps = if params.fps > 0.0 {
        params.fps
    } else if fps_header > 0.0 {
        fps_header
    } else {
        30.0
    };

    let n_frames = if params.frames > 0 {
        params.frames
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
    let point_bytes_per_frame = if float_mode {
        n_points * 4 * 4
    } else {
        n_points * 4 * 2
    };
    let analog_samples_per_frame = if analog_measurements_per_frame == 0 {
        0
    } else if analog_samples_per_frame_header > 0 {
        analog_samples_per_frame_header
    } else if params.analog.rate > 0.0 && fps > 0.0 {
        (params.analog.rate / fps).round().max(1.0) as usize
    } else if analog_measurements_per_frame > 0 {
        1
    } else {
        0
    };
    let n_analog_channels = if analog_samples_per_frame > 0 {
        let from_header = analog_measurements_per_frame.checked_div(analog_samples_per_frame).unwrap_or(0);
        params.analog.channel_count().max(from_header)
    } else {
        0
    };
    let analog_values_per_frame = analog_samples_per_frame * n_analog_channels;
    let analog_bytes_per_frame = analog_values_per_frame * if float_mode { 4 } else { 2 };
    let bytes_per_frame = point_bytes_per_frame + analog_bytes_per_frame;

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
        let frame_start = data_offset + fi * bytes_per_frame;
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

    let analog = if n_analog_channels > 0 && analog_samples_per_frame > 0 {
        Some(read_analog_data(
            buf,
            data_offset,
            n_frames,
            point_bytes_per_frame,
            bytes_per_frame,
            n_analog_channels,
            analog_samples_per_frame,
            float_mode,
            endian,
            &params.analog,
            fps,
        )?)
    } else {
        None
    };

    Ok(MarkerData {
        names: labels.into_iter().take(n_markers).collect(),
        positions,
        n_frames,
        n_markers,
        fps,
        units,
        events,
        analog,
        force_platforms,
    })
}

#[derive(Debug, Default)]
struct C3dParams {
    labels: Vec<String>,
    units: String,
    fps: f32,
    frames: usize,
    events: Vec<C3dEvent>,
    analog: AnalogParams,
    force_platforms: Vec<C3dForcePlatform>,
}

#[derive(Debug)]
struct AnalogParams {
    labels: Vec<String>,
    units: Vec<String>,
    rate: f32,
    scales: Vec<f32>,
    offsets: Vec<i16>,
    gen_scale: f32,
}

impl Default for AnalogParams {
    fn default() -> Self {
        Self {
            labels: Vec::new(),
            units: Vec::new(),
            rate: 0.0,
            scales: Vec::new(),
            offsets: Vec::new(),
            gen_scale: 1.0,
        }
    }
}

impl AnalogParams {
    fn channel_count(&self) -> usize {
        self.labels
            .len()
            .max(self.units.len())
            .max(self.scales.len())
            .max(self.offsets.len())
    }
}

#[derive(Debug, Default)]
struct EventParams {
    labels: Vec<String>,
    contexts: Vec<String>,
    times: Vec<f32>,
    used: Option<usize>,
}

#[derive(Debug, Default)]
struct ForcePlatformParams {
    types: Vec<i16>,
    channels: Vec<Vec<i16>>,
    corners: Vec<Vec<[f32; 3]>>,
    origins: Vec<[f32; 3]>,
    used: Option<usize>,
}

impl ForcePlatformParams {
    fn into_force_platforms(self) -> Vec<C3dForcePlatform> {
        let inferred = self
            .types
            .len()
            .max(self.channels.len())
            .max(self.corners.len())
            .max(self.origins.len());
        let count = self.used.unwrap_or(inferred).min(inferred);
        let mut platforms = Vec::with_capacity(count);
        for idx in 0..count {
            platforms.push(C3dForcePlatform {
                platform_type: self.types.get(idx).copied().unwrap_or_default(),
                channels: self.channels.get(idx).cloned().unwrap_or_default(),
                corners: self.corners.get(idx).cloned().unwrap_or_default(),
                origin: self.origins.get(idx).copied().unwrap_or([0.0; 3]),
            });
        }
        platforms
    }
}

impl EventParams {
    fn into_events(self) -> Vec<C3dEvent> {
        let count = self
            .used
            .unwrap_or_else(|| self.labels.len().min(self.times.len()));
        if count == 0 {
            return Vec::new();
        }

        let mut events = Vec::with_capacity(count);
        for idx in 0..count {
            let Some(label) = self
                .labels
                .get(idx)
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            else {
                continue;
            };
            let Some(time_s) = self.times.get(idx).copied().filter(|v| v.is_finite()) else {
                continue;
            };
            events.push(C3dEvent {
                label: label.to_string(),
                context: self
                    .contexts
                    .get(idx)
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default(),
                time_s,
            });
        }
        events
    }
}

/// Walk the C3D parameter section pulling out POINT marker metadata plus
/// optional EVENT, ANALOG, and FORCE_PLATFORM metadata.
fn parse_c3d_params(
    buf: &[u8],
    param_section_offset: usize,
    endian: Endian,
) -> Result<C3dParams, ParseError> {
    // After the 4-byte header (reserved, 0x50, n_param_blocks, processor),
    // parameter groups + items follow as a linked list with relative offsets.
    let mut pos = param_section_offset + 4;
    let mut params = C3dParams::default();
    let mut event_params = EventParams::default();
    let mut force_platform_params = ForcePlatformParams::default();

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
                                params.labels.push(s);
                            }
                        }
                    "UNITS"
                        // POINT:UNITS is most commonly a 1-D char array
                        // (dims = [string_len]) holding a single unit
                        // string such as "mm" or "m"; some writers emit a
                        // 2-D char array (dims = [string_len, n_strings]).
                        // Accept both — ignoring the 1-D form silently
                        // defaulted meter-based files to mm and shrank
                        // their coordinates 1000x.
                        if element_size == -1 && (dims.len() == 1 || dims.len() == 2) => {
                            let slen = dims[0];
                            // First string is the unit.
                            if slen <= data.len() {
                                params.units =
                                    String::from_utf8_lossy(&data[..slen]).trim().to_string();
                            }
                        }
                    "RATE"
                        if element_size == 4 && data.len() >= 4 => {
                            params.fps = endian.read_f32(&data[..4]);
                        }
                    "FRAMES" => {
                        if element_size == 2 && data.len() >= 2 {
                            params.frames = endian.read_u16(&data[..2]) as usize;
                        } else if element_size == 4 && data.len() >= 4 {
                            params.frames = endian.read_f32(&data[..4]) as usize;
                        }
                    }
                    _ => {}
                }
            } else if group_name == "EVENT" {
                match name.as_str() {
                    "LABELS" if element_size == -1 && dims.len() == 2 => {
                        event_params.labels = read_string_table(data, dims[0], dims[1]);
                    }
                    "CONTEXTS" if element_size == -1 && dims.len() == 2 => {
                        event_params.contexts = read_string_table(data, dims[0], dims[1]);
                    }
                    "TIMES" if element_size == 4 => {
                        event_params.times = read_event_times(data, &dims, endian);
                    }
                    "USED" => {
                        event_params.used = read_usize_scalar(data, element_size, endian);
                    }
                    _ => {}
                }
            } else if group_name == "ANALOG" {
                match name.as_str() {
                    "LABELS" if element_size == -1 && dims.len() == 2 => {
                        params.analog.labels = read_string_table(data, dims[0], dims[1]);
                    }
                    "UNITS" if element_size == -1 && dims.len() == 2 => {
                        params.analog.units = read_string_table(data, dims[0], dims[1]);
                    }
                    "RATE" if element_size == 4 && data.len() >= 4 => {
                        params.analog.rate = endian.read_f32(&data[..4]);
                    }
                    "SCALE" if element_size == 4 => {
                        params.analog.scales = read_f32_values(data, endian);
                    }
                    "OFFSET" if element_size == 2 => {
                        params.analog.offsets = read_i16_values(data, endian);
                    }
                    "GEN_SCALE" if element_size == 4 && data.len() >= 4 => {
                        params.analog.gen_scale = endian.read_f32(&data[..4]);
                    }
                    _ => {}
                }
            } else if group_name == "FORCE_PLATFORM" {
                match name.as_str() {
                    "USED" => {
                        force_platform_params.used = read_usize_scalar(data, element_size, endian);
                    }
                    "TYPE" if element_size == 2 => {
                        force_platform_params.types = read_i16_values(data, endian);
                    }
                    "CHANNEL" if element_size == 2 && dims.len() == 2 => {
                        force_platform_params.channels =
                            read_i16_matrix_columns(data, dims[0], dims[1], endian);
                    }
                    "CORNERS" if element_size == 4 && dims.len() == 3 && dims[0] == 3 => {
                        force_platform_params.corners =
                            read_force_platform_corners(data, dims[1], dims[2], endian);
                    }
                    "ORIGIN" if element_size == 4 && dims.len() == 2 && dims[0] == 3 => {
                        force_platform_params.origins = read_xyz_columns(data, dims[1], endian);
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

    params.events = event_params.into_events();
    params.force_platforms = force_platform_params.into_force_platforms();
    Ok(params)
}

fn read_string_table(data: &[u8], string_len: usize, count: usize) -> Vec<String> {
    let mut values = Vec::with_capacity(count);
    for idx in 0..count {
        let off = idx * string_len;
        if off + string_len > data.len() {
            break;
        }
        values.push(
            String::from_utf8_lossy(&data[off..off + string_len])
                .trim()
                .to_string(),
        );
    }
    values
}

fn read_event_times(data: &[u8], dims: &[usize], endian: Endian) -> Vec<f32> {
    let values = read_f32_values(data, endian);
    if dims.len() == 2 && dims[0] == 2 {
        let count = dims[1].min(values.len() / 2);
        (0..count)
            .map(|idx| values[idx * 2] * 60.0 + values[idx * 2 + 1])
            .collect()
    } else if dims.len() == 2 && dims[1] == 2 {
        let count = dims[0].min(values.len() / 2);
        (0..count)
            .map(|idx| values[idx] * 60.0 + values[idx + count])
            .collect()
    } else {
        values
    }
}

fn read_f32_values(data: &[u8], endian: Endian) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| endian.read_f32(chunk))
        .collect()
}

fn read_i16_values(data: &[u8], endian: Endian) -> Vec<i16> {
    data.chunks_exact(2)
        .map(|chunk| endian.read_i16(chunk))
        .collect()
}

fn read_i16_matrix_columns(data: &[u8], rows: usize, cols: usize, endian: Endian) -> Vec<Vec<i16>> {
    let values = read_i16_values(data, endian);
    (0..cols)
        .map(|col| {
            (0..rows)
                .filter_map(|row| values.get(col * rows + row).copied())
                .collect()
        })
        .collect()
}

fn read_xyz_columns(data: &[u8], count: usize, endian: Endian) -> Vec<[f32; 3]> {
    let values = read_f32_values(data, endian);
    (0..count)
        .filter_map(|idx| {
            let base = idx * 3;
            Some([
                *values.get(base)?,
                *values.get(base + 1)?,
                *values.get(base + 2)?,
            ])
        })
        .collect()
}

fn read_force_platform_corners(
    data: &[u8],
    corner_count: usize,
    platform_count: usize,
    endian: Endian,
) -> Vec<Vec<[f32; 3]>> {
    let values = read_f32_values(data, endian);
    (0..platform_count)
        .map(|platform| {
            (0..corner_count)
                .filter_map(|corner| {
                    let base = platform * corner_count * 3 + corner * 3;
                    Some([
                        *values.get(base)?,
                        *values.get(base + 1)?,
                        *values.get(base + 2)?,
                    ])
                })
                .collect()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn read_analog_data(
    buf: &[u8],
    data_offset: usize,
    n_frames: usize,
    point_bytes_per_frame: usize,
    bytes_per_frame: usize,
    n_channels: usize,
    samples_per_frame: usize,
    float_mode: bool,
    endian: Endian,
    params: &AnalogParams,
    point_rate: f32,
) -> Result<C3dAnalogData, ParseError> {
    let mut values = Vec::with_capacity(n_frames * samples_per_frame * n_channels);
    for frame_idx in 0..n_frames {
        let analog_start = data_offset + frame_idx * bytes_per_frame + point_bytes_per_frame;
        for sample_idx in 0..samples_per_frame {
            for channel_idx in 0..n_channels {
                let linear_idx = sample_idx * n_channels + channel_idx;
                let raw_value = if float_mode {
                    let off = analog_start + linear_idx * 4;
                    if buf.len() < off + 4 {
                        return Err(ParseError::Format(
                            "C3D analog data truncated (float)".into(),
                        ));
                    }
                    endian.read_f32(&buf[off..off + 4])
                } else {
                    let off = analog_start + linear_idx * 2;
                    if buf.len() < off + 2 {
                        return Err(ParseError::Format("C3D analog data truncated (int)".into()));
                    }
                    let raw = endian.read_i16(&buf[off..off + 2]);
                    let offset = params.offsets.get(channel_idx).copied().unwrap_or(0);
                    let scale = params.scales.get(channel_idx).copied().unwrap_or(1.0);
                    (raw as f32 - offset as f32) * scale * params.gen_scale
                };
                values.push(raw_value);
            }
        }
    }

    let labels = (0..n_channels)
        .map(|idx| {
            params
                .labels
                .get(idx)
                .filter(|label| !label.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("ANALOG{}", idx + 1))
        })
        .collect();
    let units = (0..n_channels)
        .map(|idx| params.units.get(idx).cloned().unwrap_or_default())
        .collect();
    let rate = if params.rate > 0.0 {
        params.rate
    } else {
        point_rate * samples_per_frame as f32
    };

    Ok(C3dAnalogData {
        labels,
        units,
        values,
        n_frames,
        samples_per_frame,
        n_channels,
        rate,
    })
}

fn read_usize_scalar(data: &[u8], element_size: i8, endian: Endian) -> Option<usize> {
    match element_size {
        1 if !data.is_empty() => Some(data[0] as usize),
        2 if data.len() >= 2 => Some(endian.read_u16(&data[..2]) as usize),
        4 if data.len() >= 4 => Some(endian.read_f32(&data[..4]) as usize),
        _ => None,
    }
}

#[allow(dead_code)]
fn read_block<R: Read + Seek>(r: &mut R, block_1based: usize) -> std::io::Result<[u8; BLOCK_SIZE]> {
    let mut blk = [0u8; BLOCK_SIZE];
    r.seek(SeekFrom::Start(((block_1based - 1) * BLOCK_SIZE) as u64))?;
    r.read_exact(&mut blk)?;
    Ok(blk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{ByteOrder, LittleEndian};

    fn write_record(buf: &mut [u8], pos: &mut usize, name: &str, group_id: i8, payload: &[u8]) {
        let record_start = *pos;
        buf[*pos] = name.len() as u8;
        buf[*pos + 1] = group_id as u8;
        *pos += 2;
        buf[*pos..*pos + name.len()].copy_from_slice(name.as_bytes());
        *pos += name.len();
        let offset_pos = *pos;
        *pos += 2;
        buf[*pos..*pos + payload.len()].copy_from_slice(payload);
        *pos += payload.len();
        let next_offset = (*pos - offset_pos) as i16;
        LittleEndian::write_i16(&mut buf[offset_pos..offset_pos + 2], next_offset);
        assert!(*pos > record_start);
    }

    fn group_payload(description: &str) -> Vec<u8> {
        let mut payload = vec![description.len() as u8];
        payload.extend_from_slice(description.as_bytes());
        payload
    }

    fn string_param_payload(string_len: usize, values: &[&str]) -> Vec<u8> {
        let mut payload = vec![-1_i8 as u8, 2, string_len as u8, values.len() as u8];
        for value in values {
            let mut padded = vec![b' '; string_len];
            let raw = value.as_bytes();
            padded[..raw.len().min(string_len)].copy_from_slice(&raw[..raw.len().min(string_len)]);
            payload.extend_from_slice(&padded);
        }
        payload.push(0);
        payload
    }

    fn f32_param_payload(dims: &[u8], values: &[f32]) -> Vec<u8> {
        let mut payload = vec![4, dims.len() as u8];
        payload.extend_from_slice(dims);
        for value in values {
            let mut raw = [0; 4];
            LittleEndian::write_f32(&mut raw, *value);
            payload.extend_from_slice(&raw);
        }
        payload.push(0);
        payload
    }

    fn i16_param_payload(dims: &[u8], values: &[i16]) -> Vec<u8> {
        let mut payload = vec![2, dims.len() as u8];
        payload.extend_from_slice(dims);
        for value in values {
            let mut raw = [0; 2];
            LittleEndian::write_i16(&mut raw, *value);
            payload.extend_from_slice(&raw);
        }
        payload.push(0);
        payload
    }

    fn minimal_c3d_with_event_params() -> Vec<u8> {
        let mut buf = vec![0_u8; BLOCK_SIZE * 4];

        buf[0] = 2;
        buf[1] = 0x50;
        LittleEndian::write_u16(&mut buf[2..4], 1);
        LittleEndian::write_u16(&mut buf[6..8], 1);
        LittleEndian::write_u16(&mut buf[8..10], 1);
        LittleEndian::write_f32(&mut buf[12..16], -1.0);
        LittleEndian::write_u16(&mut buf[16..18], 4);
        LittleEndian::write_f32(&mut buf[20..24], 100.0);

        let param_offset = BLOCK_SIZE;
        buf[param_offset] = 0;
        buf[param_offset + 1] = 0x50;
        buf[param_offset + 2] = 2;
        buf[param_offset + 3] = 0x54;
        let mut pos = param_offset + 4;

        write_record(&mut buf, &mut pos, "POINT", -1, &group_payload("point"));
        write_record(&mut buf, &mut pos, "EVENT", -2, &group_payload("event"));
        write_record(
            &mut buf,
            &mut pos,
            "LABELS",
            1,
            &string_param_payload(3, &["M01"]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "UNITS",
            1,
            &string_param_payload(2, &["m"]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "RATE",
            1,
            &f32_param_payload(&[1], &[100.0]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "LABELS",
            2,
            &string_param_payload(10, &["FootStrike", "ToeOff"]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "CONTEXTS",
            2,
            &string_param_payload(5, &["Left", "Right"]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "TIMES",
            2,
            &f32_param_payload(&[2, 2], &[0.0, 0.5, 0.0, 1.25]),
        );
        buf[pos] = 0;
        buf[pos + 1] = 0;

        let data_offset = BLOCK_SIZE * 3;
        LittleEndian::write_f32(&mut buf[data_offset..data_offset + 4], 1.0);
        LittleEndian::write_f32(&mut buf[data_offset + 4..data_offset + 8], 2.0);
        LittleEndian::write_f32(&mut buf[data_offset + 8..data_offset + 12], 3.0);
        LittleEndian::write_f32(&mut buf[data_offset + 12..data_offset + 16], 0.0);
        buf
    }

    fn minimal_c3d_with_int_analog_and_force_platform() -> Vec<u8> {
        let mut buf = vec![0_u8; BLOCK_SIZE * 4];

        buf[0] = 2;
        buf[1] = 0x50;
        LittleEndian::write_u16(&mut buf[2..4], 1);
        LittleEndian::write_u16(&mut buf[4..6], 4);
        LittleEndian::write_u16(&mut buf[6..8], 1);
        LittleEndian::write_u16(&mut buf[8..10], 1);
        LittleEndian::write_f32(&mut buf[12..16], 1.0);
        LittleEndian::write_u16(&mut buf[16..18], 4);
        LittleEndian::write_u16(&mut buf[18..20], 2);
        LittleEndian::write_f32(&mut buf[20..24], 100.0);

        let param_offset = BLOCK_SIZE;
        buf[param_offset] = 0;
        buf[param_offset + 1] = 0x50;
        buf[param_offset + 2] = 2;
        buf[param_offset + 3] = 0x54;
        let mut pos = param_offset + 4;

        write_record(&mut buf, &mut pos, "POINT", -1, &group_payload("point"));
        write_record(&mut buf, &mut pos, "ANALOG", -2, &group_payload("analog"));
        write_record(
            &mut buf,
            &mut pos,
            "FORCE_PLATFORM",
            -3,
            &group_payload("force"),
        );
        write_record(
            &mut buf,
            &mut pos,
            "LABELS",
            1,
            &string_param_payload(3, &["M01"]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "UNITS",
            1,
            &string_param_payload(2, &["m"]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "RATE",
            1,
            &f32_param_payload(&[1], &[100.0]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "LABELS",
            2,
            &string_param_payload(2, &["Fx", "Fy"]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "UNITS",
            2,
            &string_param_payload(1, &["N", "N"]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "RATE",
            2,
            &f32_param_payload(&[1], &[200.0]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "SCALE",
            2,
            &f32_param_payload(&[2], &[0.5, 2.0]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "OFFSET",
            2,
            &i16_param_payload(&[2], &[10, -5]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "GEN_SCALE",
            2,
            &f32_param_payload(&[1], &[2.0]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "USED",
            3,
            &i16_param_payload(&[1], &[1]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "TYPE",
            3,
            &i16_param_payload(&[1], &[2]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "CHANNEL",
            3,
            &i16_param_payload(&[6, 1], &[1, 2, 3, 4, 5, 6]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "CORNERS",
            3,
            &f32_param_payload(
                &[3, 4, 1],
                &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            ),
        );
        write_record(
            &mut buf,
            &mut pos,
            "ORIGIN",
            3,
            &f32_param_payload(&[3, 1], &[0.0, 0.0, -0.05]),
        );
        buf[pos] = 0;
        buf[pos + 1] = 0;

        let data_offset = BLOCK_SIZE * 3;
        LittleEndian::write_i16(&mut buf[data_offset..data_offset + 2], 1);
        LittleEndian::write_i16(&mut buf[data_offset + 2..data_offset + 4], 2);
        LittleEndian::write_i16(&mut buf[data_offset + 4..data_offset + 6], 3);
        LittleEndian::write_i16(&mut buf[data_offset + 6..data_offset + 8], 0);
        for (idx, value) in [12_i16, 7, 14, 9].iter().enumerate() {
            let off = data_offset + 8 + idx * 2;
            LittleEndian::write_i16(&mut buf[off..off + 2], *value);
        }
        buf
    }

    fn minimal_c3d_with_float_analog() -> Vec<u8> {
        let mut buf = vec![0_u8; BLOCK_SIZE * 4];

        buf[0] = 2;
        buf[1] = 0x50;
        LittleEndian::write_u16(&mut buf[2..4], 1);
        LittleEndian::write_u16(&mut buf[4..6], 2);
        LittleEndian::write_u16(&mut buf[6..8], 1);
        LittleEndian::write_u16(&mut buf[8..10], 1);
        LittleEndian::write_f32(&mut buf[12..16], -1.0);
        LittleEndian::write_u16(&mut buf[16..18], 4);
        LittleEndian::write_u16(&mut buf[18..20], 1);
        LittleEndian::write_f32(&mut buf[20..24], 100.0);

        let param_offset = BLOCK_SIZE;
        buf[param_offset] = 0;
        buf[param_offset + 1] = 0x50;
        buf[param_offset + 2] = 2;
        buf[param_offset + 3] = 0x54;
        let mut pos = param_offset + 4;

        write_record(&mut buf, &mut pos, "POINT", -1, &group_payload("point"));
        write_record(&mut buf, &mut pos, "ANALOG", -2, &group_payload("analog"));
        write_record(
            &mut buf,
            &mut pos,
            "LABELS",
            1,
            &string_param_payload(3, &["M01"]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "UNITS",
            1,
            &string_param_payload(2, &["m"]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "LABELS",
            2,
            &string_param_payload(2, &["Fx", "Fy"]),
        );
        write_record(
            &mut buf,
            &mut pos,
            "RATE",
            2,
            &f32_param_payload(&[1], &[100.0]),
        );
        buf[pos] = 0;
        buf[pos + 1] = 0;

        let data_offset = BLOCK_SIZE * 3;
        for (idx, value) in [1.0_f32, 2.0, 3.0, 0.0, 10.25, -2.5].iter().enumerate() {
            let off = data_offset + idx * 4;
            LittleEndian::write_f32(&mut buf[off..off + 4], *value);
        }
        buf
    }

    #[test]
    fn parse_c3d_extracts_event_labels_contexts_and_times() {
        let data = parse_c3d_bytes(&minimal_c3d_with_event_params()).expect("parse c3d");

        assert_eq!(data.names, vec!["M01"]);
        assert_eq!(data.events.len(), 2);
        assert_eq!(data.events[0].label, "FootStrike");
        assert_eq!(data.events[0].context, "Left");
        assert_eq!(data.events[0].time_s, 0.5);
        assert_eq!(data.events[1].label, "ToeOff");
        assert_eq!(data.events[1].context, "Right");
        assert_eq!(data.events[1].time_s, 1.25);
    }

    #[test]
    fn parse_c3d_decodes_int16_analog_and_force_platform_params() {
        let data =
            parse_c3d_bytes(&minimal_c3d_with_int_analog_and_force_platform()).expect("parse c3d");

        assert_eq!(data.names, vec!["M01"]);
        assert_eq!(data.positions, vec![1.0, 2.0, 3.0]);
        let analog = data.analog.expect("analog channels");
        assert_eq!(analog.labels, vec!["Fx", "Fy"]);
        assert_eq!(analog.units, vec!["N", "N"]);
        assert_eq!(analog.n_frames, 1);
        assert_eq!(analog.samples_per_frame, 2);
        assert_eq!(analog.n_channels, 2);
        assert_eq!(analog.rate, 200.0);
        assert_eq!(analog.values, vec![2.0, 48.0, 4.0, 56.0]);

        assert_eq!(data.force_platforms.len(), 1);
        let platform = &data.force_platforms[0];
        assert_eq!(platform.platform_type, 2);
        assert_eq!(platform.channels, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(platform.corners.len(), 4);
        assert_eq!(platform.corners[2], [1.0, 1.0, 0.0]);
        assert_eq!(platform.origin, [0.0, 0.0, -0.05]);
    }

    #[test]
    fn parse_c3d_decodes_float_analog_without_int_scaling() {
        let data = parse_c3d_bytes(&minimal_c3d_with_float_analog()).expect("parse c3d");

        let analog = data.analog.expect("analog channels");
        assert_eq!(analog.labels, vec!["Fx", "Fy"]);
        assert_eq!(analog.samples_per_frame, 1);
        assert_eq!(analog.n_channels, 2);
        assert_eq!(analog.values, vec![10.25, -2.5]);
    }
}
